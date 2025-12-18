"""
AI Combiner 基类

提供 AI 模型训练器的基础抽象。
实际基类定义在 core.abstractions.AITrainerBase
"""

from core.abstractions import AITrainerBase, RollingCalculatorBase

__all__ = ['AITrainerBase', 'RollingCalculatorBase']
