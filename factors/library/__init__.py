"""
Factors Library

因子库 - 所有因子的集合

自动导入所有因子以触发注册装饰器
"""

from .base import BaseFactor
from .complex_factor_base import ComplexFactorBase

# 简单因子（Simple Factors）
from .momentum import MomentumFactor
from .rsi import RSIFactor
from .kdj import KDJFactor
from .bollinger_bands import BollingerBandsFactor
from .moving_average_cross import MovingAverageCrossFactor
from .macd import MACDFactor
from .adx_dmi import ADXDMIFactor
from .volume_spike import VolumeSpikeFactor
from .reversal_20d import Reversal20DFactor

# 复合因子 - 非行业中性化（Complex Factors - Non-Industry-Neutral）
from .ep import EPFactor
from .bp import BPFactor
from .roe import ROEFactor
from .gpm import GPMFactor
from .sales_growth import SalesGrowthFactor
from .cfop import CFOPFactor
from .asset_turnover import AssetTurnoverFactor
from .current_ratio import CurrentRatioFactor

__all__ = [
    'BaseFactor',
    'ComplexFactorBase',
    'MomentumFactor',
    'RSIFactor',
    'KDJFactor',
    'BollingerBandsFactor',
    'MovingAverageCrossFactor',
    'MACDFactor',
    'ADXDMIFactor',
    'VolumeSpikeFactor',
    'Reversal20DFactor',
    'EPFactor',
    'BPFactor',
    'ROEFactor',
    'GPMFactor',
    'SalesGrowthFactor',
    'CFOPFactor',
    'AssetTurnoverFactor',
    'CurrentRatioFactor',
]
