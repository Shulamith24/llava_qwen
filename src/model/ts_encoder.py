"""
时序编码器包装类
包装官方PatchTST编码器，提供统一的接口和权重加载功能
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

# 添加PatchTST路径
current_dir = os.path.dirname(os.path.abspath(__file__))
patchtst_path = os.path.join(current_dir, '..', 'PatchTST_supervised')
if patchtst_path not in sys.path:
    sys.path.insert(0, patchtst_path)

from layers.PatchTST_backbone import PatchTST_backbone


class PatchTSTEncoderWrapper(nn.Module):
    """
    PatchTST编码器包装类
    
    基于官方PatchTST_backbone，提取backbone的特征而非预测输出
    
    提供统一的接口，支持：
    - 预训练权重加载
    - 灵活的冻结/解冻控制
    - 兼容LLaVA架构的属性
    
    Args:
        context_window: 输入序列长度
        patch_len: patch长度
        stride: patch步长
        d_model: 模型维度
        n_layers: Transformer层数
        n_heads: 注意力头数
        d_ff: 前馈网络维度
        dropout: dropout率
        use_revin: 是否使用RevIN归一化
        **kwargs: 其他PatchTST参数
    """
    
    def __init__(
        self,
        context_window: int = 256,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 16,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_revin: bool = False,  # 默认不用RevIN，避免归一化问题
        **kwargs
    ):
        super().__init__()
        
        # 保存配置
        self.context_window = context_window
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.use_revin = use_revin
        
        # 计算patch数量
        self.patch_num = int((context_window - patch_len) / stride + 1)
        
        # 兼容LLaVA架构的属性
        self.is_loaded = False
        self.hidden_size = d_model  # 每个patch的特征维度
        
        # 延迟初始化backbone（因为c_in需要在forward时确定）
        self.backbone = None
        self._c_in = None  # 记录当前backbone的变量数
        
        # 存储配置用于创建backbone
        self.backbone_config = {
            'context_window': context_window,
            'target_window': 0,  # 我们不需要预测，只需要特征
            'patch_len': patch_len,
            'stride': stride,
            'd_model': d_model,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'd_ff': d_ff,
            'dropout': dropout,
            'attn_dropout': dropout,
            'head_dropout': 0,
            'individual': False,
            'revin': use_revin,
            'affine': True,
            'subtract_last': False,
            'pretrain_head': False,
            'head_type': 'flatten',
            **kwargs
        }
    
    def _get_or_create_backbone(self, c_in: int):
        """
        获取或创建对应变量数的backbone
        
        Args:
            c_in: 变量数
            
        Returns:
            backbone: PatchTST_backbone实例
        """
        # 如果backbone不存在或变量数改变，重新创建
        if self.backbone is None or self._c_in != c_in:
            self.backbone = PatchTST_backbone(
                c_in=c_in,
                **self.backbone_config
            )
            self._c_in = c_in
            
            # 如果已经加载过权重，需要重新加载
            if hasattr(self, '_cached_checkpoint') and self._cached_checkpoint is not None:
                self._load_weights_to_backbone(self._cached_checkpoint)
        
        return self.backbone
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（单样本）
        
        提取PatchTST backbone的特征，跳过预测head
        
        Args:
            x: [n_vars, seq_len] 单个样本的时间序列
            
        Returns:
            features: [n_vars, patch_num, d_model] 时序特征
        """
        # 确保输入是2D: [n_vars, seq_len]
        if x.dim() == 3:
            assert x.size(0) == 1, "Wrapper只支持单样本编码"
            x = x.squeeze(0)
        
        n_vars, seq_len = x.shape
        assert seq_len == self.context_window, \
            f"输入序列长度{seq_len}与context_window{self.context_window}不匹配"
        
        # 获取或创建backbone
        backbone = self._get_or_create_backbone(n_vars)
        
        # 添加batch维度: [1, n_vars, seq_len]
        z = x.unsqueeze(0)
        
        # === 以下代码复制自官方PatchTST_backbone.forward，但跳过head和denorm ===
        
        # RevIN normalization
        if backbone.revin:
            z = z.permute(0, 2, 1)  # [bs, seq_len, n_vars]
            z = backbone.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)  # [bs, n_vars, seq_len]
        
        # Patching
        if backbone.padding_patch == 'end':
            z = backbone.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=backbone.patch_len, step=backbone.stride)  # [bs, n_vars, patch_num, patch_len]
        z = z.permute(0, 1, 3, 2)  # [bs, n_vars, patch_len, patch_num]
        
        # Backbone encoder
        z = backbone.backbone(z)  # [bs, n_vars, d_model, patch_num]
        
        # 转换为 [bs, n_vars, patch_num, d_model]
        z = z.permute(0, 1, 3, 2)  # [bs, n_vars, patch_num, d_model]
        
        # 注意：我们跳过了head和denorm，因为：
        # 1. head是用于预测的，我们只需要特征
        # 2. denorm用于还原预测值，但特征不需要还原
        
        # 移除batch维度
        z = z.squeeze(0)  # [n_vars, patch_num, d_model]
        
        return z
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载预训练权重
        
        Args:
            checkpoint_path: checkpoint文件路径
        """
        if not os.path.exists(checkpoint_path):
            print(f"警告：checkpoint文件不存在: {checkpoint_path}")
            return
        
        print(f"从 {checkpoint_path} 加载PatchTST权重...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 缓存checkpoint以便后续使用
        self._cached_checkpoint = checkpoint
        
        # 如果已经初始化了backbone，立即加载
        if self.backbone is not None:
            self._load_weights_to_backbone(checkpoint)
        
        self.is_loaded = True
    
    def _load_weights_to_backbone(self, checkpoint):
        """
        将权重加载到backbone
        
        Args:
            checkpoint: checkpoint字典
        """
        # 提取模型权重
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 加载权重（只加载匹配的层）
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict, strict=False)
        print(f"成功加载 {len(pretrained_dict)}/{len(state_dict)} 个参数")
    
    @property
    def device(self):
        """获取设备"""
        if self.backbone is not None:
            return next(self.backbone.parameters()).device
        return torch.device('cpu')
    
    @property
    def dtype(self):
        """获取数据类型"""
        if self.backbone is not None:
            return next(self.backbone.parameters()).dtype
        return torch.float32
    
    @property
    def dummy_feature(self):
        """返回dummy特征（用于非多模态样本）"""
        return torch.zeros(1, self.d_model, device=self.device, dtype=self.dtype)
    
    def freeze(self):
        """冻结所有参数"""
        self.requires_grad_(False)
        print("已冻结PatchTST编码器")
    
    def unfreeze(self):
        """解冻所有参数"""
        self.requires_grad_(True)
        print("已解冻PatchTST编码器")


def build_ts_encoder(
    checkpoint_path: Optional[str] = None,
    freeze: bool = True,
    **config
) -> PatchTSTEncoderWrapper:
    """
    构建时序编码器
    
    Args:
        checkpoint_path: 预训练权重路径
        freeze: 是否冻结权重
        **config: 编码器配置参数
        
    Returns:
        encoder: 时序编码器实例
    """
    # 创建编码器
    encoder = PatchTSTEncoderWrapper(**config)
    
    # 加载权重
    if checkpoint_path is not None:
        encoder.load_checkpoint(checkpoint_path)
    
    # 冻结参数
    if freeze:
        encoder.freeze()
    
    return encoder
