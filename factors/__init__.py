"""
因子领域

提供因子计算、合成、分析的完整解决方案。

模块结构：
- library/           因子库（单文件单因子）
- pipeline/          因子处理流水线（合成、标准化、滚动）
- analysis/          因子分析（计算器、报告、指标）
"""

from . import library
from . import pipeline
from . import analysis

__all__ = [
    'library',
    'pipeline',
    'analysis',
]
