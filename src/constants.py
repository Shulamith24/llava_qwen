"""
全局常量定义
类似LLaVA的constants模块，定义多模态训练所需的常量
"""

# ======================== 训练相关常量 ========================

# 损失计算中的忽略索引（padding和时序token位置不计算loss）
IGNORE_INDEX = -100

# ======================== 特殊Token ========================

# 时序占位符（类似LLaVA的<image>）
DEFAULT_TS_TOKEN = "<ts>"
DEFAULT_TS_START_TOKEN = "<ts>"
DEFAULT_TS_END_TOKEN = "</ts>"

# Token索引（在tokenizer初始化后会被动态设置）
# 使用-200作为占位，实际值在添加token后更新
TS_TOKEN_INDEX = -200

# ======================== 模型配置 ========================

# 默认的图像/时序patch token（用于LLaVA兼容）
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# 兼容性：IMAGE_TOKEN_INDEX指向TS_TOKEN_INDEX
IMAGE_TOKEN_INDEX = TS_TOKEN_INDEX
