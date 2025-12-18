"""Combiners - 因子合成器"""
from .base import BaseFactorCombiner
from .equal_weight import EqualWeightCombiner
from .fixed_weight import FixedWeightCombiner
from .dynamic_weight import DynamicWeightCombiner
from .dynamic_significance import DynamicSignificanceCombiner
from .ai_combiner import AICombiner
from .rolling import (
    RollingCalculatorBase,
    RollingICIRCalculator,
    RollingRegressionCalculator,
    RollingAITrainer,
    AdversarialLLMCombiner,
)

__all__ = [
    'BaseFactorCombiner',
    'EqualWeightCombiner',
    'FixedWeightCombiner',
    'DynamicWeightCombiner',
    'DynamicSignificanceCombiner',
    'AICombiner',
    # Rolling combiners
    'RollingCalculatorBase',
    'RollingICIRCalculator',
    'RollingRegressionCalculator',
    'RollingAITrainer',
    'AdversarialLLMCombiner',
]
