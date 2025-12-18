"""
AI Combiner 模块

提供基于机器学习的因子合成器。
"""

from .base import AITrainerBase, RollingCalculatorBase
from .ai_lightgbm import RollingAITrainer
from .ai_adversarial_llm import AdversarialLLMCombiner

__all__ = [
    'AITrainerBase',
    'RollingCalculatorBase',
    'RollingAITrainer',
    'AdversarialLLMCombiner',
]
