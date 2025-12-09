"""
时序编码器
基于 PatchTST 思想的精简实现，专门针对 bfloat16 优化数值稳定性

关键设计：
- 使用 float32 进行注意力计算，避免 bfloat16 溢出
- 使用较小的初始化值
- 添加数值稳定性保护
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class StableMultiheadAttention(nn.Module):
    """
    数值稳定的多头注意力
    
    关键：在 bfloat16 下，先转为 float32 计算 softmax，再转回 bfloat16
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} 必须能被 n_heads {n_heads} 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV 投影
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化：使用较小的方差
        self._init_weights()
    
    def _init_weights(self):
        # Xavier 初始化，但使用较小的 gain
        nn.init.xavier_uniform_(self.qkv.weight, gain=0.5)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        original_dtype = x.dtype
        
        # QKV 投影
        qkv = self.qkv(x)  # [batch, seq_len, 3 * d_model]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, n_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # === 关键：使用 float32 计算 attention ===
        q = q.float()
        k = k.float()
        v_float = v.float()
        
        # 缩放点积注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 数值稳定性：减去最大值
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True).values
        
        # Softmax（float32）
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        output = torch.matmul(attn_weights, v_float)
        
        # 转回原始 dtype
        output = output.to(original_dtype)
        
        # 重塑并投影
        output = output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        return output


class StableTransformerEncoderLayer(nn.Module):
    """
    数值稳定的 Transformer 编码器层
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = StableMultiheadAttention(d_model, n_heads, dropout)
        
        # FFN
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight, gain=0.5)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight, gain=0.5)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention + residual
        attn_out = self.self_attn(x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # FFN + residual
        ffn_out = self.linear2(F.gelu(self.linear1(x)))
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)
        
        return x


class SimplePatchTSTEncoder(nn.Module):
    """
    精简版 PatchTST 时序编码器（数值稳定版）
    
    将时间序列分割成 patches，经过 Transformer 编码后输出特征。
    
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
        
        # 2. 位置编码（较小的初始值）
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.01)  # 较小的 std
        
        # 3. Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 4. Transformer 层
        self.layers = nn.ModuleList([
            StableTransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 5. 最终归一化
        self.norm = nn.LayerNorm(d_model)
        
        # 初始化 patch projection
        nn.init.xavier_uniform_(self.patch_projection.weight, gain=0.5)
        if self.patch_projection.bias is not None:
            nn.init.zeros_(self.patch_projection.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [n_vars, seq_len] 单个样本的时间序列（已归一化）
            
        Returns:
            features: [n_vars, n_patches, d_model] 时序特征
        """
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
        
        # === 2. Patch 投影 ===
        x = self.patch_projection(x)
        # x: [n_vars, n_patches, d_model]       
        # === 3. 位置编码 ===
        x = x + self.pos_embed
        
        # === 4. Dropout ===
        x = self.dropout(x)
        
        # === 5. Transformer 层 ===
        for layer in self.layers:
            x = layer(x)
        
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
