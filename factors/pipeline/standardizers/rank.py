"""排名标准化器"""
import pandas as pd
import numpy as np
from .base import BaseStandardizer
from core.registry import register_standardizer

@register_standardizer('Rank', description='排名标准化')
class CrossSectionalRankStandardizer(BaseStandardizer):
    def __init__(self, ascending: bool = True, pct: bool = True, **kwargs):
        """初始化排名标准化器
        
        Args:
            ascending: 是否升序排名，默认为True
            pct: 是否返回百分位排名，默认为True（缩放到[0,1]区间）
            **kwargs: 其他配置参数
        """
        super().__init__(**kwargs)
        self.ascending = ascending
        self.pct = pct
    
    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        """将数据转换为截面排名并缩放到[0,1]区间"""
        if self.pct:
            # 返回百分位排名，已经在[0,1]区间
            return raw_signals_df.rank(ascending=self.ascending, method='average') / len(raw_signals_df)
        else:
            # 返回原始排名，然后缩放到[0,1]区间
            ranks = raw_signals_df.rank(ascending=self.ascending, method='average')
            return (ranks - 1) / (len(raw_signals_df) - 1)