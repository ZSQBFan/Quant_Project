"""
Factors Library

因子库 - 所有因子的集合

自动导入所有因子以触发注册装饰器
"""

from .base import BaseFactor

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

__all__ = [
    'BaseFactor',
    'MomentumFactor',
    'RSIFactor',
    'KDJFactor',
    'BollingerBandsFactor',
    'MovingAverageCrossFactor',
    'MACDFactor',
    'ADXDMIFactor',
    'VolumeSpikeFactor',
    'Reversal20DFactor',
]
