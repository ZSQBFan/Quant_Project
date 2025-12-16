"""
滚动计算器模块

提供基于滚动窗口的动态因子权重计算器。
"""

from .icir import RollingICIRCalculator
from .regression import RollingRegressionCalculator
from .ai_trainer import RollingAITrainer
from .adversarial_llm import AdversarialLLMCombiner

__all__ = [
    'RollingICIRCalculator',
    'RollingRegressionCalculator',
    'RollingAITrainer',
    'AdversarialLLMCombiner',
]
