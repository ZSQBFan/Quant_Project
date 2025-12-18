"""
滚动计算器模块

提供基于滚动窗口的动态因子权重计算器。
"""

from .base import RollingCalculatorBase
from .rolling_icir import RollingICIRCalculator
from .rolling_regression import RollingRegressionCalculator
from .ai_combiner import RollingAITrainer, AdversarialLLMCombiner

__all__ = [
    'RollingCalculatorBase',
    'RollingICIRCalculator',
    'RollingRegressionCalculator',
    'RollingAITrainer',
    'AdversarialLLMCombiner',
]
