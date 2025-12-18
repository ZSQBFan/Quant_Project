"""Combiners - 因子合成器"""
from .base import BaseFactorCombiner
from .equal_weight import EqualWeightCombiner
from .fixed_weight import FixedWeightCombiner
from .dynamic_weight import DynamicWeightCombiner
from .dynamic_significance import DynamicSignificanceCombiner
from .rolling import (
    RollingCalculatorBase,
    RollingICIRCalculator,
    RollingRegressionCalculator,
    RollingAITrainer,
    AdversarialLLMCombiner,
    StaticAICombiner,
)

__all__ = [
    'BaseFactorCombiner',
    'EqualWeightCombiner',
    'FixedWeightCombiner',
    'DynamicWeightCombiner',
    'DynamicSignificanceCombiner',
    'StaticAICombiner',
    # Rolling combiners
    'RollingCalculatorBase',
    'RollingICIRCalculator',
    'RollingRegressionCalculator',
    'RollingAITrainer',
    'AdversarialLLMCombiner',
]
