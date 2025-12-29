"""Allocators"""
from .equal_weight import EqualWeightAllocator, AllocatorBase
from .risk_parity import RiskParityAllocator
from .max_sharpe import MaxSharpeAllocator
from .min_variance import MinimumVarianceAllocator

# 别名：波动率倒数加权就是风险平价
InverseVolatilityAllocator = RiskParityAllocator

__all__ = [
    'EqualWeightAllocator',
    'AllocatorBase',
    'RiskParityAllocator',
    'InverseVolatilityAllocator',
    'MaxSharpeAllocator',
    'MinimumVarianceAllocator'
]
