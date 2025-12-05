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
DEFAULT_TS_END_TOKEN = "<ts/>"

# Token索引（在tokenizer初始化后会被动态设置）
# 使用-200作为占位，实际值在添加token后更新
# 
# 重要：所有模块应通过 constants.TS_TOKEN_INDEX 访问（而非 from constants import TS_TOKEN_INDEX）
# 这样才能读取到运行时更新后的值。更新方式示例：
#   import constants as GLOBAL_CONSTANTS
#   GLOBAL_CONSTANTS.TS_TOKEN_INDEX = tokenizer.convert_tokens_to_ids(DEFAULT_TS_TOKEN)
TS_TOKEN_INDEX = -200



