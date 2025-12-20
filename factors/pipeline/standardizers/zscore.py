"""Z-Score标准化器"""
import pandas as pd
from .base import BaseStandardizer
from core.registry import register_standardizer

@register_standardizer('ZScore', description='Z-Score标准化')
class CrossSectionalZScoreStandardizer(BaseStandardizer):
    def __init__(self, winsorize: bool = True, winsorize_limits: list = None, **kwargs):
        """初始化Z-Score标准化器
        
        Args:
            winsorize: 是否进行极值处理，默认为True
            winsorize_limits: 极值处理百分位，默认为[0.01, 0.99]
            **kwargs: 其他配置参数
        """
        super().__init__(**kwargs)
        self.winsorize = winsorize
        self.winsorize_limits = winsorize_limits or [0.01, 0.99]
    
    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        # 极值处理
        if self.winsorize:
            data = raw_signals_df.copy()
            lower_limit = data.quantile(self.winsorize_limits[0])
            upper_limit = data.quantile(self.winsorize_limits[1])
            data = data.clip(lower=lower_limit, upper=upper_limit, axis=1)
        else:
            data = raw_signals_df
        
        # Z-Score标准化
        return (data - data.mean()) / data.std()
