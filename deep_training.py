import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm
from accelerate import Accelerator
import argparse
import logging
from datetime import datetime
import json
from pathlib import Path

# 导入自定义模块
from data_prepare import MultiFormatDataLoader
from deep_models import ModelFactory, BaseModelConfig


def setup_logging(log_dir: str, model_name: str):
    """设置日志记录"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{model_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_metrics(metrics: dict, save_path: str):
    """保存训练指标"""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def train_and_evaluate_accelerated(
    model, 
    train_loader, 
    val_loader, 
    test_loader,
    accelerator: Accelerator,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    model_name: str = 'model',
    patience: int = 10,
    save_dir: str = './models',
    logger = None
):
    """
    使用Accelerate的训练和评估函数
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器  
        test_loader: 测试数据加载器
        accelerator: Accelerate加速器
        num_epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        model_name: 模型名称
        patience: 早停耐心值
        save_dir: 模型保存目录
        logger: 日志记录器
    """
    
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 使用accelerator准备模型、优化器和数据加载器
    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler
    )
    
    # 训练历史记录
    best_val_acc = 0.0
    best_model_path = save_dir / f'best_{model_name}.pth'
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []
    learning_rates = []
    patience_counter = 0
    
    if logger and accelerator.is_local_main_process:
        logger.info(f"Starting training for {model_name}")
        logger.info(f"Total epochs: {num_epochs}, Learning rate: {lr}, Weight decay: {weight_decay}")
        logger.info(f"Patience: {patience}, Save directory: {save_dir}")
        logger.info(f"Device: {accelerator.device}")
    
    # 训练循环
    for epoch in trange(num_epochs, desc=f"Training {model_name}", disable=not accelerator.is_local_main_process):
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        total_train = 0
        correct_train = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                         disable=not accelerator.is_local_main_process, leave=False)
        
        for batch_idx, (xb, yb) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(xb)
            loss = criterion(outputs, yb)
            
            # 反向传播
            accelerator.backward(loss)
            
            # 梯度裁剪
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            running_loss += loss.item() * xb.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += yb.size(0)
            correct_train += (predicted == yb).sum().item()
            
            # 更新进度条
            # if batch_idx % 10 == 0:
            #     train_pbar.set_postfix({
            #         'loss': f'{loss.item():.4f}',
            #         'acc': f'{100 * correct_train / total_train:.2f}%'
            #     })
        
        # 同步所有进程的训练统计
        running_loss = accelerator.gather_for_metrics(torch.tensor(running_loss, device=accelerator.device)).sum().item()
        total_train = accelerator.gather_for_metrics(torch.tensor(total_train, device=accelerator.device)).sum().item()
        correct_train = accelerator.gather_for_metrics(torch.tensor(correct_train, device=accelerator.device)).sum().item()
        
        avg_train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                       disable=not accelerator.is_local_main_process, leave=False)
        
        with torch.no_grad():
            for xb, yb in val_pbar:
                outputs = model(xb)
                loss = criterion(outputs, yb)
                
                val_running_loss += loss.item() * xb.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += yb.size(0)
                val_correct += (predicted == yb).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        # 同步所有进程的验证统计
        val_running_loss = accelerator.gather_for_metrics(torch.tensor(val_running_loss, device=accelerator.device)).sum().item()
        val_total = accelerator.gather_for_metrics(torch.tensor(val_total, device=accelerator.device)).sum().item()
        val_correct = accelerator.gather_for_metrics(torch.tensor(val_correct, device=accelerator.device)).sum().item()
        
        avg_val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 日志输出
        if accelerator.is_local_main_process:
            log_msg = (
                f"[{model_name}] Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            print(log_msg)
            if logger:
                logger.info(log_msg)
        
        # 早停和模型保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # 只在主进程保存模型
            if accelerator.is_local_main_process:
                # accelerator.save(accelerator.unwrap_model(model).state_dict(), best_model_path)
                if logger:
                    logger.info(f"New best model saved with val_acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if accelerator.is_local_main_process:
                print(f"Early stopping at epoch {epoch+1}")
                if logger:
                    logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # 测试阶段
    # 加载最佳模型
    # if accelerator.is_local_main_process and best_model_path.exists():
    #     model_state = torch.load(best_model_path, map_location=accelerator.device)
    #     accelerator.unwrap_model(model).load_state_dict(model_state)
    
    # 等待所有进程同步
    accelerator.wait_for_everyone()
    
    model.eval()
    test_correct = 0
    test_total = 0
    test_predictions = []
    test_targets = []
    
    test_pbar = tqdm(test_loader, desc="Testing", 
                    disable=not accelerator.is_local_main_process)
    
    with torch.no_grad():
        for xb, yb in test_pbar:
            outputs = model(xb)
            _, predicted = torch.max(outputs, 1)
            
            test_total += yb.size(0)
            test_correct += (predicted == yb).sum().item()
            
            # 收集预测结果用于后续分析
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(yb.cpu().numpy())
    
    # 同步测试统计
    test_total = accelerator.gather_for_metrics(torch.tensor(test_total, device=accelerator.device)).sum().item()
    test_correct = accelerator.gather_for_metrics(torch.tensor(test_correct, device=accelerator.device)).sum().item()
    
    test_acc = test_correct / test_total
    
    if accelerator.is_local_main_process:
        print(f"[{model_name}] Test Accuracy: {test_acc:.4f}")
        if logger:
            logger.info(f"[{model_name}] Final Test Accuracy: {test_acc:.4f}")
        
        # 保存训练指标
        metrics = {
            'model_name': model_name,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'total_epochs': epoch + 1,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'train_accuracies': train_accuracies,
            'learning_rates': learning_rates,
            'hyperparameters': {
                'lr': lr,
                'weight_decay': weight_decay,
                'patience': patience,
                'num_epochs': num_epochs
            }
        }
        
        metrics_path = save_dir / f"{model_name}_metrics.json"
        save_metrics(metrics, metrics_path)
        if logger:
            logger.info(f"Metrics saved to: {metrics_path}")
    
    return best_val_acc, test_acc, test_predictions, test_targets


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ICA Binary Classification Training')
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default='./data', 
                       help='Data directory containing pickle files')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    
    # 模型相关参数
    parser.add_argument('--model_type', type=str, default='mlp',
                       choices=['mlp', 'cnn', 'rnn', 'gru', 'resnet18', 'resnet50', 
                               'densenet121', 'vit'],
                       help='Type of model to train')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # 特定模型参数
    parser.add_argument('--hidden_dims', type=str, default='128,64',
                       help='Hidden dimensions for MLP (comma-separated)')
    parser.add_argument('--channels', type=str, default='32,64',
                          help='Channels for CNN (comma-separated)')
    parser.add_argument('--kernel_size', type=int, default=3,
                          help='Kernel size for CNN')
    parser.add_argument('--hidden_size', type=int, default=64,
                          help='Hidden size for RNN/GRU')
    parser.add_argument('--num_layers', type=int, default=2,
                            help='Number of layers for RNN/GRU')
    parser.add_argument('--bidirectional', action='store_true',
                          help='Use bidirectional RNN/GRU')

    # 训练相关参数
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # 交叉验证参数
    parser.add_argument('--use_kfold', action='store_true',
                       help='Use K-fold cross validation')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='Number of K-fold splits')
    parser.add_argument('--stratified', action='store_true', default=True,
                       help='Use stratified K-fold')
    parser.add_argument('--fold', type=int, default=0,
                       help='The num of this fold')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='The ratio of the test set in each fold, only used when use_kfold is False')    

    # 保存和日志参数
    parser.add_argument('--save_dir', type=str, default='./models',
                       help='Directory to save models and results')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory to save logs')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for logging')
    
    args = parser.parse_args()
    
    # 设置实验名称
    if args.experiment_name is None:
        args.experiment_name = f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 初始化Accelerator
    accelerator = Accelerator()

    # 设置日志
    logger = setup_logging(args.log_dir, args.experiment_name)
    
    # 设置模型参数
    model_config = ModelFactory.DEFAULT_CONFIGS.get(args.model_type, BaseModelConfig())
    model_config.num_classes = args.num_classes
    model_config.dropout = args.dropout
    # 确保特定模型设置齐全，生成对应config
    try:
        if args.model_type == 'mlp':
            model_config.hidden_dims = list(map(int, args.hidden_dims.split(',')))
        elif args.model_type == 'cnn':
            model_config.channels = list(map(int, args.channels.split(',')))
            model_config.kernel_size = int(args.kernel_size)
        elif args.model_type == 'rnn' or args.model_type == 'gru':
            model_config.hidden_size = int(args.hidden_size)
            model_config.num_layers = int(args.num_layers)
            model_config.bidirectional = args.bidirectional
    except Exception as e:
        # if accelerator.is_local_main_process:
        logger.error(f"Error parsing model parameters: {str(e)}")
        raise

    if accelerator.is_local_main_process:
        logger.info("=" * 60)
        logger.info(f"Starting experiment: {args.experiment_name}")
        logger.info(f"Arguments: {vars(args)}")
        # logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info("=" * 60)
        # logger.info(f"Model configuration: {model_config.hidden_dims}")

    try:
        # 加载数据
        if accelerator.is_local_main_process:
            logger.info("Loading data...")
        
        data_loader = MultiFormatDataLoader(
            data_dir=args.data_dir, 
            batch_size=args.batch_size
        )
        data_loader.load_data()
            
        data_loader.setup_kfold(
            n_splits=args.n_splits,
            stratified=args.stratified,
            random_state=42,
            test_ratio=args.test_ratio if hasattr(args, 'test_ratio') else 0.1
        )

        if accelerator.is_local_main_process:
            logger.info(f"\n{'='*20} FOLD {args.fold + 1}/{args.n_splits} {'='*20}")
        
        # 获取当前折的数据加载器
        train_loader = data_loader.get_dataloader('train', args.model_type, fold=args.fold)
        val_loader = data_loader.get_dataloader('val', args.model_type, fold=args.fold)
        test_loader = data_loader.get_dataloader('test', args.model_type, fold=args.fold)

        # 创建模型
        model = ModelFactory.create_model(
            args.model_type,
            config=model_config
        )
        
        if accelerator.is_local_main_process:
            model_info = ModelFactory.get_model_info(model)
            logger.info(f"Model info: {model_info}")
        
        # 训练模型
        fold_model_name = f"{args.experiment_name}_fold_{args.fold + 1}"
        
        val_acc, test_acc, _, _ = train_and_evaluate_accelerated(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            accelerator=accelerator,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            model_name=fold_model_name,
            patience=args.patience,
            save_dir=args.save_dir,
            logger=logger
        )
        
        if accelerator.is_local_main_process:
            logger.info(f"Fold {args.fold + 1} - Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    except Exception as e:
        if accelerator.is_local_main_process:
            logger.error(f"Error during training: {str(e)}")
            logger.exception("Detailed error traceback:")
        raise
    
    finally:
        if accelerator.is_local_main_process:
            logger.info("Training completed successfully!")
            logger.info("=" * 60)


if __name__ == "__main__":
    main()