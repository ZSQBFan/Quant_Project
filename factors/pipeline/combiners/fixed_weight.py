"""固定权重合成器"""
import pandas as pd
from .base import BaseFactorCombiner
from core.registry import register_combiner

@register_combiner('FixedWeight', description='固定权重合成策略')
class FixedWeightCombiner(BaseFactorCombiner):
    def __init__(self, factor_weights: dict, **kwargs):
        self.factor_weights = factor_weights
    
    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        weights_series = pd.Series(self.factor_weights)
        aligned_weights = weights_series.reindex(standardized_df.columns).fillna(0)
        return (standardized_df * aligned_weights).sum(axis=1)
