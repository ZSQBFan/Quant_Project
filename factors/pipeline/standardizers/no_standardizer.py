"""无标准化器"""
import pandas as pd
from .base import BaseStandardizer
from core.registry import register_standardizer

@register_standardizer('None', description='不进行标准化')
class NoStandardizer(BaseStandardizer):
    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        return raw_signals_df
