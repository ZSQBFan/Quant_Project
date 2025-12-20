"""中位数绝对偏差（MAD）标准化器"""
import pandas as pd
import numpy as np
from .base import BaseStandardizer
from core.registry import register_standardizer

@register_standardizer('MAD', description='中位数绝对偏差标准化（稳健）')
class CrossSectionalMADStandardizer(BaseStandardizer):
    def __init__(self, scale_factor: float = 1.4826, **kwargs):
        """初始化MAD标准化器
        
        Args:
            scale_factor: 与正态分布标准差的换算系数，默认为1.4826
            **kwargs: 其他配置参数
        """
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
    
    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        """使用MAD方法进行稳健标准化"""
        data = raw_signals_df.copy()
        
        # 计算中位数
        median = data.median(axis=0)
        
        # 计算中位数绝对偏差（MAD）
        mad = (data - median).abs().median(axis=0)
        
        # 避免除零错误，如果MAD为0则使用小的正数替代
        mad = mad.replace(0, 1e-8)
        
        # MAD标准化： (x - median) / (MAD * scale_factor)
        # scale_factor 确保与正态分布的标准差一致
        standardized_data = (data - median) / (mad * self.scale_factor)
        
        return standardized_data