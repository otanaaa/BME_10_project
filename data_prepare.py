import os
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import KFold, StratifiedKFold

class MultiFormatDataLoader:
    """多格式数据加载器，支持多种机器学习模型"""
    
    def __init__(self, data_dir='./data', batch_size=2048):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.raw_data = {}
        self.data = {}
        self.splits = {}
        self.kfold_splits = {}
        
    def load_data(self):
        """加载和预处理数据"""
        print("Loading data...")
        
        # 并行加载pickle文件
        fnames = [f for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(self._load_single_file, fnames), 
                              total=len(fnames), desc="Loading files"))
        
        # 整理数据
        arrays, labels = zip(*[r for r in results if r is not None])
        data_3d = np.stack(arrays).astype(np.float32)  # (N, 64, 64, 20)
        labels = np.array(labels)
        
        # 数据归一化
        mean, std = data_3d.mean(), data_3d.std() + 1e-8
        data_3d = (data_3d - mean) / std
        
        # 生成多种格式
        self.raw_data = {
            'flat': data_3d.reshape(len(data_3d), -1),  # (N, 81920)
            '3d': data_3d,                              # (N, 64, 64, 20)
            'sequence': data_3d.reshape(len(data_3d), 64*64, 20),  # (N, 4096, 20)
            'labels': labels
        }
        
        # 保持原有data接口兼容性
        self.data = self.raw_data.copy()
        
        print(f"Loaded {len(labels)} samples")
        return self
    
    def _load_single_file(self, fname):
        """加载单个pickle文件"""
        try:
            with open(os.path.join(self.data_dir, fname), 'rb') as f:
                tup = pickle.load(f)
                return tup[0], tup[2]  # 数组和标签
        except:
            return None
    
    def setup_kfold(self, n_splits=5, stratified=True, random_state=42, test_ratio=0.2):
        """设置K折交叉验证
        
        Args:
            n_splits: K折数量
            stratified: 是否使用分层K折（保持各类别比例）
            random_state: 随机种子
            test_ratio: 测试集比例（先分出测试集，再对剩余数据做K折）
        """
        n_total = len(self.raw_data['labels'])
        
        # 先分出测试集
        if test_ratio > 0:
            n_test = int(test_ratio * n_total)
            # 随机选择测试集索引
            np.random.seed(random_state)
            all_indices = np.arange(n_total)
            np.random.shuffle(all_indices)
            
            test_indices = all_indices[:n_test]
            train_val_indices = all_indices[n_test:]
            
            # 保存测试集
            self.test_indices = test_indices
            self.train_val_indices = train_val_indices
            
            # 用于K折的数据和标签
            kfold_labels = self.raw_data['labels'][train_val_indices]
        else:
            # 全部数据用于K折
            self.train_val_indices = np.arange(n_total)
            self.test_indices = np.array([])
            kfold_labels = self.raw_data['labels']
        
        # 设置K折交叉验证
        if stratified:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = kf.split(self.train_val_indices, kfold_labels)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = kf.split(self.train_val_indices)
        
        # 保存K折划分
        self.kfold_splits = {}
        for fold, (train_idx, val_idx) in enumerate(splits):
            # train_idx和val_idx是相对于train_val_indices的索引
            actual_train_idx = self.train_val_indices[train_idx]
            actual_val_idx = self.train_val_indices[val_idx]
            
            self.kfold_splits[fold] = {
                'train_indices': actual_train_idx,
                'val_indices': actual_val_idx,
                'test_indices': self.test_indices
            }
        
        self.n_splits = n_splits
        # If this is the main torch process, print the conclusion
        if torch.distributed.get_rank() == 0:
            print(f"K-Fold CV setup: {n_splits} folds")
            if test_ratio > 0:
                print(f"Test set: {len(self.test_indices)} samples ({test_ratio:.1%})")
            print(f"Train+Val per fold: ~{len(self.train_val_indices)//n_splits} val, ~{len(self.train_val_indices)*(n_splits-1)//n_splits} train samples")
        
        return self
    
    def split_data(self, train_ratio=0.8, val_ratio=0.1):
        """传统的固定划分方式（与K折交叉验证互斥）"""
        n = len(self.raw_data['labels'])
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)
        
        self.splits = {}
        for split, (start, end) in [('train', (0, train_end)), 
                                   ('val', (train_end, val_end)), 
                                   ('test', (val_end, n))]:
            self.splits[split] = {
                'flat': self.raw_data['flat'][start:end],
                '3d': self.raw_data['3d'][start:end],
                'sequence': self.raw_data['sequence'][start:end],
                'labels': self.raw_data['labels'][start:end]
            }
        return self
    
    def get_fold_data(self, fold, split_type='train'):
        """获取指定折数和类型的数据
        
        Args:
            fold: 折数 (0 to n_splits-1)
            split_type: 'train', 'val', 'test'
        """
        if fold not in self.kfold_splits:
            raise ValueError(f"Fold {fold} not found. Available folds: {list(self.kfold_splits.keys())}")
        
        indices_key = f'{split_type}_indices'
        if indices_key not in self.kfold_splits[fold]:
            raise ValueError(f"Split type '{split_type}' not available")
            
        indices = self.kfold_splits[fold][indices_key]
        
        if len(indices) == 0:
            # 返回空数据
            return {
                'flat': np.array([]).reshape(0, -1),
                '3d': np.array([]).reshape(0, 64, 64, 20),
                'sequence': np.array([]).reshape(0, 4096, 20),
                'labels': np.array([])
            }
        
        return {
            'flat': self.raw_data['flat'][indices],
            '3d': self.raw_data['3d'][indices],
            'sequence': self.raw_data['sequence'][indices],
            'labels': self.raw_data['labels'][indices]
        }
    
    def get_dataloader(self, split='train', model_type='mlp', shuffle=None, fold=None):
        """获取指定模型类型和数据集的DataLoader
        
        Args:
            split: 'train', 'val', 'test'
            model_type: 模型类型
            shuffle: 是否打乱数据
            fold: K折交叉验证的折数，如果为None则使用传统划分
        """
        if shuffle is None:
            shuffle = (split == 'train')
        
        # 选择数据源
        if fold is not None:
            # 使用K折数据
            data_dict = self.get_fold_data(fold, split)
        else:
            # 使用传统划分数据
            if not self.splits:
                raise ValueError("No data splits found. Call split_data() or setup_kfold() first.")
            data_dict = self.splits[split]
        
        # 数据格式映射
        format_map = {
            'mlp': 'flat', 'logistic': 'flat', 'svm': 'flat', 
            'knn': 'flat', 'rf': 'flat',
            'cnn': '3d',
            'resnet18': '3d', 'resnet50': '3d', 'densenet121': '3d', 'vit': '3d',
            'rnn': 'sequence', 'gru': 'sequence', 'lstm': 'sequence'
        }
        
        data_format = format_map.get(model_type, 'flat')
        X = data_dict[data_format]
        y = data_dict['labels']
        
        # 如果数据为空，返回空的DataLoader
        if len(X) == 0:
            dataset = TensorDataset(torch.empty(0), torch.empty(0, dtype=torch.long))
            return DataLoader(dataset, batch_size=1)
        
        # 转换为tensor
        X_tensor = self._format_tensor(X, model_type)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # 调整batch_size
        batch_size = self.batch_size // 64 if model_type in ['rnn', 'gru', 'lstm'] else self.batch_size
        
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                         num_workers=4, pin_memory=True)
    
    def _format_tensor(self, X, model_type):
        """根据模型类型格式化tensor"""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        if model_type == 'cnn':
            # (B,64,64,20) -> (B,1,64,64,20)
            return X_tensor.unsqueeze(1)
        elif model_type in ['resnet18', 'resnet50', 'densenet121', 'vit']:
            # (B,64,64,20) -> (B,20,64,64)
            return X_tensor.permute(0, 3, 1, 2)
        else:
            return X_tensor
    
    def get_sklearn_data(self, split='train', fold=None):
        """获取sklearn格式的数据
        
        Args:
            split: 'train', 'val', 'test'
            fold: K折交叉验证的折数，如果为None则使用传统划分
        """
        if fold is not None:
            data_dict = self.get_fold_data(fold, split)
        else:
            data_dict = self.splits[split]
        return data_dict['flat'], data_dict['labels']
    
    def cross_validate(self, model_func, model_type='mlp', metrics_func=None):
        """执行K折交叉验证
        
        Args:
            model_func: 模型创建函数，接收(train_loader, val_loader)参数
            model_type: 模型类型
            metrics_func: 评估函数，接收(model, test_loader)参数，返回评估结果
            
        Returns:
            list: 每折的评估结果
        """
        if not self.kfold_splits:
            raise ValueError("K-fold not set up. Call setup_kfold() first.")
        
        results = []
        
        for fold in range(self.n_splits):
            print(f"\n=== Fold {fold + 1}/{self.n_splits} ===")
            
            # 获取数据加载器
            train_loader = self.get_dataloader('train', model_type, fold=fold)
            val_loader = self.get_dataloader('val', model_type, fold=fold)
            
            # 训练模型
            model = model_func(train_loader, val_loader)
            
            # 评估模型
            if metrics_func:
                result = metrics_func(model, val_loader)
                results.append(result)
                print(f"Fold {fold + 1} result: {result}")
        
        return results