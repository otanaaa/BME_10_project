import torch
import torch.nn as nn
import torchvision.models as models
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

class BaseModelConfig:
    """基础模型配置类"""
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        self.num_classes = num_classes
        self.dropout = dropout

class MLPConfig(BaseModelConfig):
    def __init__(self, hidden_dims: list = [1024, 256, 64], **kwargs):
        super().__init__(**kwargs)
        self.hidden_dims = hidden_dims

class CNNConfig(BaseModelConfig):
    def __init__(self, channels: list = [16, 32, 64], kernel_size: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size

class RNNConfig(BaseModelConfig):
    def __init__(self, hidden_size: int = 128, num_layers: int = 2, 
                 bidirectional: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

class BaseModel(nn.Module, ABC):
    """基础模型类"""
    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.config = config
        self._build_model()
    
    @abstractmethod
    def _build_model(self):
        """构建模型结构"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2),  # 假设float32
            'config': self.config.__dict__
        }

class MLPModel(BaseModel):
    """多层感知机"""
    def _build_model(self):
        layers = []
        prev_dim:int # 将在forward中动态设置
        
        for i, hidden_dim in enumerate(self.config.hidden_dims):
            if i == 0:
                # 第一层需要input_dim，在forward中处理
                self.first_layer_dim = hidden_dim
            else:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout if i < len(self.config.hidden_dims) - 1 else 0)
                ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, self.config.num_classes))
        self.layers = nn.ModuleList(layers)
        self.first_layer = None
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # 动态创建第一层
        if self.first_layer is None:
            input_dim = x.size(1)
            self.first_layer = nn.Linear(input_dim, self.first_layer_dim).to(x.device)
        
        x = nn.functional.relu(self.first_layer(x))
        x = nn.functional.dropout(x, p=self.config.dropout, training=self.training)
        
        for layer in self.layers:
            x = layer(x)
        return x

class CNNModel(BaseModel):
    """3D卷积神经网络"""
    def _build_model(self):
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        in_channels = 1
        for out_channels in self.config.channels:
            self.conv_layers.append(
                nn.Conv3d(in_channels, out_channels, self.config.kernel_size, padding=1)
            )
            self.bn_layers.append(nn.BatchNorm3d(out_channels))
            in_channels = out_channels
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(2)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        final_channels = self.config.channels[-1]
        self.classifier = nn.Sequential(
            nn.Linear(final_channels, final_channels // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(final_channels // 2, self.config.num_classes)
        )
    
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 64, 64, 20) -> (B, 1, 64, 64, 20)
        
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.relu(bn(conv(x)))
            x = self.pool(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class RNNModel(BaseModel):
    """RNN/GRU模型"""
    def __init__(self, config: RNNConfig, rnn_type: str = 'GRU'):
        self.rnn_type = rnn_type.upper()
        super().__init__(config)
    
    def _build_model(self):
        rnn_cls = getattr(nn, self.rnn_type)
        self.rnn = rnn_cls(
            input_size=20,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=self.config.bidirectional,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0
        )
        
        rnn_output_size = self.config.hidden_size * (2 if self.config.bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_size, 64),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(64, self.config.num_classes)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), 4096, 20)
        
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # 取最后一个时间步
        return self.classifier(out)

class PretrainedModelWrapper(BaseModel):
    """预训练模型包装器"""
    def __init__(self, config: BaseModelConfig, model_name: str):
        self.model_name = model_name
        super().__init__(config)
    
    def _build_model(self):
        if 'resnet' in self.model_name:
            if self.model_name == 'resnet18':
                self.model = models.resnet18(weights=None)
            elif self.model_name == 'resnet50':
                self.model = models.resnet50(weights=None)
            
            # 修改输入层和输出层
            self.model.conv1 = nn.Conv2d(20, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.config.num_classes)
            
        elif self.model_name == 'densenet121':
            self.model = models.densenet121(weights=None)
            self.model.features.conv0 = nn.Conv2d(20, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, self.config.num_classes)
    
    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == 20, f"Expected (B, 20, H, W), got {x.shape}"
        return self.model(x)

class ViTModel(BaseModel):
    """Vision Transformer模型"""
    def _build_model(self):
        try:
            from transformers import ViTModel as HFViTModel, ViTConfig
            config = ViTConfig(
                image_size=64, 
                num_labels=self.config.num_classes,
                num_channels=20, 
                hidden_size=256, 
                num_hidden_layers=6, 
                num_attention_heads=8, 
                intermediate_size=512
            )
            self.vit = HFViTModel(config)
            self.classifier = nn.Linear(config.hidden_size, self.config.num_classes)
        except ImportError:
            raise ImportError("transformers library required for ViT model")
    
    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == 20, f"Expected (B, 20, H, W), got {x.shape}"
        outputs = self.vit(pixel_values=x)
        pooled = outputs.pooler_output
        return self.classifier(pooled)

class ModelFactory:
    """模型工厂类"""
    
    # 默认配置
    DEFAULT_CONFIGS = {
        'mlp': MLPConfig(),
        'logistic': MLPConfig(hidden_dims=[]),  # 无隐藏层
        'cnn': CNNConfig(),
        'rnn': RNNConfig(),
        'gru': RNNConfig(),
        'resnet18': BaseModelConfig(),
        'resnet50': BaseModelConfig(),
        'densenet121': BaseModelConfig(),
        'vit': BaseModelConfig()
    }
    
    @classmethod
    def create_model(cls, model_type: str, config: Optional[BaseModelConfig] = None, 
                    device: Optional[torch.device] = None, **kwargs) -> nn.Module:
        """创建模型
        
        Args:
            model_type: 模型类型
            config: 模型配置，如果为None则使用默认配置
            device: 设备
            **kwargs: 额外参数会更新到config中
        """
        model_type = model_type.lower()
        
        # 获取配置
        if config is None:
            config = cls.DEFAULT_CONFIGS.get(model_type, BaseModelConfig())
        
        # 更新配置
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # 创建模型
        if model_type == 'mlp':
            model = MLPModel(config)
        elif model_type == 'logistic':
            # 逻辑回归就是没有隐藏层的MLP
            logistic_config = MLPConfig(hidden_dims=[], **config.__dict__)
            model = nn.Linear(kwargs.get('input_dim', 81920), config.num_classes)
        elif model_type == 'cnn':
            model = CNNModel(config)
        elif model_type in ['rnn', 'gru']:
            model = RNNModel(config, model_type)
        elif model_type in ['resnet18', 'resnet50', 'densenet121']:
            model = PretrainedModelWrapper(config, model_type)
        elif model_type == 'vit':
            model = ViTModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 移动到指定设备
        if device is not None:
            model = model.to(device)
        
        return model
    
    @classmethod
    def get_model_info(cls, model: nn.Module) -> Dict[str, Any]:
        """获取模型信息"""
        if isinstance(model, BaseModel):
            return model.get_model_info()
        else:
            # 通用模型信息
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            return {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'model_size_mb': total_params * 4 / (1024 ** 2),
                'model_type': model.__class__.__name__
            }
    
    @classmethod
    def list_available_models(cls) -> list:
        """列出可用的模型类型"""
        return list(cls.DEFAULT_CONFIGS.keys())

# 兼容原接口的便捷函数
def get_model(model_type: str, input_dim: Optional[int] = None, **kwargs) -> nn.Module:
    """兼容原接口的模型创建函数"""
    return ModelFactory.create_model(model_type, input_dim=input_dim, **kwargs)

# 使用示例
# if __name__ == "__main__":
#     # 创建不同类型的模型
    
#     # 1. 使用默认配置
#     mlp_model = ModelFactory.create_model('mlp')
#     cnn_model = ModelFactory.create_model('cnn')
    
#     # 2. 使用自定义配置
#     custom_config = MLPConfig(hidden_dims=[512, 128, 32], dropout=0.5)
#     custom_mlp = ModelFactory.create_model('mlp', custom_config)
    
#     # 3. 使用kwargs快速配置
#     rnn_model = ModelFactory.create_model('gru', hidden_size=256, num_layers=3)
    
#     # 4. 指定设备
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     gpu_model = ModelFactory.create_model('resnet18', device=device)
    
#     # 获取模型信息
#     for name, model in [('MLP', mlp_model), ('CNN', cnn_model), ('RNN', rnn_model)]:
#         info = ModelFactory.get_model_info(model)
#         print(f"\n{name} Model Info:")
#         print(f"  Parameters: {info['total_params']:,}")
#         print(f"  Model Size: {info['model_size_mb']:.2f} MB")
    
#     # 列出可用模型
#     print(f"\nAvailable models: {ModelFactory.list_available_models()}")
    
#     # 兼容原接口
#     legacy_model = get_model('mlp', input_dim=81920)
#     print(f"\nLegacy interface works: {type(legacy_model)}")