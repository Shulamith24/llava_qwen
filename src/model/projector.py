"""
MLP投影层
将PatchTST的时序特征投影到Qwen3的embedding空间
"""

import torch
import torch.nn as nn


class MLP2xGELU(nn.Module):
    """
    两层MLP投影层，使用GELU激活函数
    
    架构: Linear -> GELU -> Linear -> GELU -> Linear
    类似LLaVA的投影层设计
    
    Args:
        input_dim: 输入维度（PatchTST的d_model）
        output_dim: 输出维度（Qwen3的hidden_size）
        hidden_multiplier: 隐藏层维度倍数（默认4倍）
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_multiplier: int = 4):
        super().__init__()
        
        hidden_dim = input_dim * hidden_multiplier
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch, seq_len, input_dim] 或 [seq_len, input_dim]
            
        Returns:
            projected: [batch, seq_len, output_dim] 或 [seq_len, output_dim]
        """
        return self.projector(x)


class IdentityProjector(nn.Module):
    """
    恒等投影层（用于调试）
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        assert input_dim == output_dim, "Identity projector requires input_dim == output_dim"
    
    def forward(self, x):
        return x


class ScaleEncoder(nn.Module):
    """
    尺度编码器
    
    将时间序列的尺度信息(mean, std)编码为隐藏向量，
    作为额外的token注入到LLM中，使模型能够感知原始数据的尺度。
    
    Args:
        output_dim: 输出维度（与LLM hidden_size一致）
        hidden_dim: 中间隐藏层维度      #这里使用一个超参数
    """
    
    def __init__(self, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 输入: (mean, std)
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),  #从超参数--> llm_hidden_size
        )
    
    def forward(self, scale_stats: torch.Tensor) -> torch.Tensor:
        """
        编码尺度信息
        
        Args:
            scale_stats: [n_vars, 2] 或 [2]，每个变量的(mean, std)
            
        Returns:
            scale_embed: [n_vars, output_dim] 或 [output_dim]，尺度嵌入向量
        """
        return self.encoder(scale_stats)


def build_projector(projector_type: str, input_dim: int, output_dim: int, **kwargs):
    """
    构建投影层
    
    Args:
        projector_type: 投影层类型 ('mlp2x_gelu', 'identity', 'linear')
        input_dim: 输入维度
        output_dim: 输出维度
        **kwargs: 其他参数
        
    Returns:
        projector: 投影层实例
    """
    if projector_type == 'mlp2x_gelu':
        hidden_multiplier = kwargs.get('hidden_multiplier', 4)
        return MLP2xGELU(input_dim, output_dim, hidden_multiplier)
    
    elif projector_type == 'linear':
        return nn.Linear(input_dim, output_dim)
    
    elif projector_type == 'identity':
        return IdentityProjector(input_dim, output_dim)
    
    else:
        raise ValueError(f"Unknown projector type: {projector_type}")
