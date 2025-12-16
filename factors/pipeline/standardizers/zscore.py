"""Z-Score标准化器"""
import pandas as pd
from .base import BaseStandardizer
from core.registry import register_standardizer

@register_standardizer('ZScore', description='Z-Score标准化')
class CrossSectionalZScoreStandardizer(BaseStandardizer):
    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        return (raw_signals_df - raw_signals_df.mean()) / raw_signals_df.std()
