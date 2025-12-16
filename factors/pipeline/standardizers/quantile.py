"""分位数标准化器"""
import pandas as pd
from .base import BaseStandardizer
from core.registry import register_standardizer

@register_standardizer('Quantile', description='分位数标准化')
class CrossSectionalQuantileStandardizer(BaseStandardizer):
    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        return raw_signals_df.rank(pct=True) - 0.5
