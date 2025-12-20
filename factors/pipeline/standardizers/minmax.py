"""MinMax标准化器"""
import pandas as pd
from .base import BaseStandardizer
from core.registry import register_standardizer

@register_standardizer('MinMax', description='MinMax标准化')
class CrossSectionalMinMaxStandardizer(BaseStandardizer):
    """截面MinMax标准化器
    
    将每个截面（时间点）的数据线性映射到指定区间（默认为[0, 1]）
    公式：(x - min) / (max - min) * (max_range - min_range) + min_range
    """
    
    def __init__(self, feature_range=(0, 1), **kwargs):
        """初始化MinMax标准化器
        
        Args:
            feature_range: 目标范围，格式为(min, max)
            **kwargs: 忽略其他参数
        """
        self.feature_range = feature_range
        super().__init__()

    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        """执行MinMax标准化
        
        Args:
            raw_signals_df: 原始因子信号DataFrame，行为时间，列为股票
            
        Returns:
            标准化后的DataFrame
        """
        # 对每个截面（每行）进行MinMax标准化
        # 首先计算每行的最小值和最大值
        min_vals = raw_signals_df.min(axis=1)
        max_vals = raw_signals_df.max(axis=1)
        
        # 避免除零错误：当max == min时，标准化结果为0
        # 这种情况下通常表示该截面所有值都相同
        range_vals = max_vals - min_vals
        
        # 处理边界情况：当range为0时，使用0作为标准化结果
        standardized = raw_signals_df.subtract(min_vals, axis=0)
        
        # 只对非零范围进行除法运算
        mask = range_vals > 0
        standardized.loc[mask] = standardized.loc[mask].div(range_vals[mask], axis=0)
        
        # 对于范围为0的情况，标准化结果保持为0（表示所有值相同）
        standardized = standardized.fillna(0)
        
        # 缩放到指定范围
        min_range, max_range = self.feature_range
        if min_range != 0 or max_range != 1:
            scale = max_range - min_range
            standardized = standardized * scale + min_range
            
        return standardized