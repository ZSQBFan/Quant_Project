"""
Industry Neutral Factors

行业中性因子 - 通过行业中性化处理消除行业偏差

所有行业中性因子共享一个中性化方法
"""

from .base import IndustryNeutralFactorBase

# 导入所有行业中性因子
from .momentum import IndNeuMomentumFactor
from .reversal import IndNeuReversalFactor
from .volume_cv import IndNeuVolumeCVFactor
from .ep import IndNeuEPFactor
from .bp import IndNeuBPFactor
from .sales_growth import IndNeuSalesGrowthFactor
from .cfop import IndNeuCFOPFactor
from .gpm import IndNeuGPMFactor

__all__ = [
    'IndustryNeutralFactorBase',
    'IndNeuMomentumFactor',
    'IndNeuReversalFactor',
    'IndNeuVolumeCVFactor',
    'IndNeuEPFactor',
    'IndNeuBPFactor',
    'IndNeuSalesGrowthFactor',
    'IndNeuCFOPFactor',
    'IndNeuGPMFactor',
]
