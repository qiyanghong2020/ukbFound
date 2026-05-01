__version__ = "0.2.1"

import logging
import sys

# -----------------------------------------------------------------------------
# 基础 logger 配置
# -----------------------------------------------------------------------------
logger = logging.getLogger("ukbfound")

# 只在第一次 import 时初始化 handler，避免重复添加
if not logger.hasHandlers() or len(logger.handlers) == 0:
    logger.propagate = False
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# -----------------------------------------------------------------------------
# 这里不再在包初始化阶段导入 trainer，避免循环依赖：
#   ukbfound -> trainer -> ukbfound
#
# 以后如需使用训练相关函数，直接：
#   from ukbfound.trainer import train, prepare_data, ...
#
# 推理 / demo 用到的模块（model、tokenizer）建议在各自脚本中显式导入：
#   from ukbfound.model import TransformerModel
#   from ukbfound.tokenizer import tokenize_and_pad_batch
# -----------------------------------------------------------------------------
