"""
时序编码器
基于 PatchTST 思想的精简实现

关键设计：
- 整个编码器在 float32 下运行，避免 bf16 精度问题
- 使用标准 PyTorch 组件
- 输出在传入投影层前再转换精度
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SimplePatchTSTEncoder(nn.Module):
    """
    精简版 PatchTST 时序编码器
    
    将时间序列分割成 patches，经过 Transformer 编码后输出特征。
    整个编码过程在 float32 下进行，确保数值稳定性。
    
    Args:
        context_window: 输入序列长度
        patch_len: patch 长度
        stride: patch 步长
        d_model: Transformer 模型维度
        n_layers: Transformer 层数
        n_heads: 注意力头数
        d_ff: 前馈网络维度
        dropout: dropout 率
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
        self.dropout_rate = dropout
        
        # 计算 patch 数量
        self.n_patches = int((context_window - patch_len) / stride + 1)
        
        # 兼容属性
        self.is_loaded = False
        self.hidden_size = d_model
        
        # === 核心组件 ===
        
        # 2. 位置编码 (Sinusoidal - Fixed & Stable)
        # 使用固定正弦位置编码，无需训练，数值稳定性极高
        pe = torch.zeros(self.n_patches, d_model)
        position = torch.arange(0, self.n_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为 buffer (不更新，随模型保存)
        self.register_buffer('pos_embed', pe.unsqueeze(0))
        
        # 3. Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 4. Transformer 编码器层（使用 PyTorch 原生组件）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN 更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 5. 最终归一化
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [n_vars, seq_len] 单个样本的时间序列
            
        Returns:
            features: [n_vars, n_patches, d_model] 时序特征 (float32)
        """
        # 处理输入维度
        if x.dim() == 3:
            assert x.size(0) == 1, "只支持单样本编码"
            x = x.squeeze(0)
        
        n_vars, seq_len = x.shape
        assert seq_len == self.context_window, \
            f"输入序列长度 {seq_len} 与 context_window {self.context_window} 不匹配"
            
        # 强制使用 float32 进行计算，避免 bf16 下的精度问题
        # 使用 autocast 确保算子在 float32 下运行，而不是手动转换权重
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            # === 1. Patching ===
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            # x: [n_vars, n_patches, patch_len]
            
            # === 2. Patch 投影 ===
            x = self.patch_projection(x)
            # x: [n_vars, n_patches, d_model]
            
            # === 3. 位置编码 ===
            # pos_embed is now a buffer [1, n_patches, d_model]
            x = x + self.pos_embed
            
            # === 4. Dropout ===
            x = self.dropout(x)
            
            # === 5. Transformer 层 ===
            x = self.transformer(x)
            
            # === 6. 最终归一化 ===
            x = self.norm(x)
        
        return x
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载预训练权重（可选）"""
        if checkpoint_path is None:
            return
        
        import os
        if not os.path.exists(checkpoint_path):
            print(f"警告：checkpoint 不存在: {checkpoint_path}")
            return
        
        print(f"尝试加载权重: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 尝试兼容加载
            loaded = 0
            model_dict = self.state_dict()
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    model_dict[k] = v
                    loaded += 1
            
            self.load_state_dict(model_dict, strict=False)
            print(f"加载了 {loaded} 个参数")
            self.is_loaded = True
        except Exception as e:
            print(f"加载失败: {e}")
    
    @property
    def device(self):
        return self.patch_projection.weight.device
    
    @property
    def dtype(self):
        return self.patch_projection.weight.dtype
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.d_model, device=self.device, dtype=self.dtype)
    
    def freeze(self):
        self.requires_grad_(False)
        print("已冻结时序编码器")
    
    def unfreeze(self):
        self.requires_grad_(True)
        print("已解冻时序编码器")


# 兼容别名
PatchTSTEncoderWrapper = SimplePatchTSTEncoder


def build_ts_encoder(
    checkpoint_path: Optional[str] = None,
    freeze: bool = True,
    **config
) -> SimplePatchTSTEncoder:
    """构建时序编码器"""
    encoder = SimplePatchTSTEncoder(**config)
    
    if checkpoint_path is not None:
        encoder.load_checkpoint(checkpoint_path)
    
    if freeze:
        encoder.freeze()
    
    return encoder
