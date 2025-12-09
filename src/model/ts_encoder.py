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
        
        # 1. Patch 投影层
        self.patch_projection = nn.Linear(patch_len, d_model)
        
        # 2. 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # 标准初始化
        
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
               输入应为 float32，编码全程保持 float32
            
        Returns:
            features: [n_vars, n_patches, d_model] 时序特征 (float32)
        """
        # 确保输入为 float32
        original_dtype = x.dtype
        if x.dtype != torch.float32:
            x = x.float()
        
        # 处理输入维度
        if x.dim() == 3:
            assert x.size(0) == 1, "只支持单样本编码"
            x = x.squeeze(0)
        
        n_vars, seq_len = x.shape
        assert seq_len == self.context_window, \
            f"输入序列长度 {seq_len} 与 context_window {self.context_window} 不匹配"
        
        # === 1. Patching ===
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [n_vars, n_patches, patch_len]
        
        # === 2. Patch 投影（确保权重也是 float32）===
        x = self.patch_projection(x)
        # x: [n_vars, n_patches, d_model]
        
        # === 3. 位置编码 ===
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
        # 始终返回 float32，确保输入转换正确
        return torch.float32
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.d_model, device=self.device, dtype=torch.float32)
    
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
