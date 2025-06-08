# run_training.py - 主运行脚本
import yaml
import argparse
from pathlib import Path
import subprocess
import sys
import time
import logging

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training_log.txt'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件格式错误: {e}")

def validate_config(config):
    """验证配置文件的完整性"""
    required_keys = ['data', 'model', 'training', 'paths']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置文件缺少必要的键: {key}")
    
    # 验证数据目录是否存在
    if not Path(config['data']['data_dir']).exists():
        raise FileNotFoundError(f"数据目录不存在: {config['data']['data_dir']}")

def run_single_fold(config, model_type, experiment_name=None, logger=None, fold=0):
    """运行单个模型的训练"""

    cmd = [
        "accelerate", "launch", 
        "--config_file", "configs/accelerate_config.yaml",
        "--num_cpu_threads_per_process", "6",
        "deep_training.py"
    ]
    
    # 添加数据参数
    cmd.extend([
        "--data_dir", config['data']['data_dir'],
        "--batch_size", str(config['data']['batch_size'])
    ])
    
    # 添加模型参数
    cmd.extend([
        "--model_type", model_type,
        "--num_classes", str(config['model']['num_classes']),
        "--dropout", str(config['model']['dropout'])
    ])
    
    # 添加模型特定配置
    if model_type == 'mlp':
        cmd.extend([
            "--hidden_dims", ','.join(map(str, config['model']['mlp']['hidden_dims']))
        ])
    elif model_type == 'cnn':
        cmd.extend([
            "--channels", ','.join(map(str, config['model']['cnn']['channels'])),
            "--kernel_size", str(config['model']['cnn']['kernel_size'])
        ])
    elif model_type == 'rnn':
        cmd.extend([
            "--hidden_size", str(config['model']['rnn']['hidden_size']),
            "--num_layers", str(config['model']['rnn']['num_layers']),
            "--bidirectional" if config['model']['rnn']['bidirectional'] else ""
        ])

    # 添加训练参数
    cmd.extend([
        "--num_epochs", str(config['training']['num_epochs']),
        "--lr", str(config['training']['lr']),
        "--weight_decay", str(config['training']['weight_decay']),
        "--patience", str(config['training']['patience'])
    ])

    # 添加交叉验证参数
    if config['cross_validation']['use_kfold']:
        cmd.extend([
            "--use_kfold",
            "--n_splits", str(config['cross_validation']['n_splits'])
        ])
        if config['cross_validation']['stratified']:
            cmd.append("--stratified")
        cmd.extend([
            "test_ratio", str(config['cross_validation']['test_ratio']),
        ])
    
    # 添加折数参数
    cmd.extend([
        "--fold", str(fold)
    ])

    # 添加保存路径参数
    cmd.extend([
        "--save_dir", config['paths']['save_dir'],
        "--log_dir", config['paths']['log_dir']
    ])

    if experiment_name:
        cmd.extend(["--experiment_name", experiment_name])
    
    if logger and fold==0:
        logger.info(f"执行命令: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        if logger:
            logger.info(f"模型 {model_type} 训练完成，耗时: {end_time - start_time:.2f}秒")
            if result.stdout:
                logger.info(f"输出: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"模型 {model_type} 训练失败: {e}")
            if e.stderr:
                logger.error(f"错误信息: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run ICA training experiments')
    parser.add_argument('--config', type=str, default='configs/test.yaml',
                       help='Path to configuration file')
    parser.add_argument('--single_model', type=str,
                       help='Train only one specific model', required=True)
    parser.add_argument('--experiment_prefix', type=str, default='ica_exp',
                       help='Prefix for experiment names')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    
    try:
        # 加载和验证配置
        logger.info(f"加载配置文件: {args.config}")
        config = load_config(args.config)
        validate_config(config)
        logger.info("配置文件验证通过")
        
        # 创建必要的目录
        save_dir = Path(config['paths']['save_dir'])
        log_dir = Path(config['paths']['log_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建保存目录: {save_dir}")
        logger.info(f"创建日志目录: {log_dir}")
        
        # 训练单个模型
        experiment_name = f"{args.experiment_prefix}_{args.single_model}"
        logger.info(f"开始训练模型: {args.single_model}")
        for fold in range(config['cross_validation']['n_splits']):
            logger.info(f"开始第 {fold + 1} 折训练")
            success = run_single_fold(config, args.single_model, experiment_name, logger, fold=fold)
            if success:
                logger.info(f"第 {fold + 1} 折训练成功")
            else:
                logger.error(f"第 {fold + 1} 折训练失败")
                sys.exit(1)
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()