"""动态权重合成器"""
import pandas as pd
from .base import BaseFactorCombiner
from core.registry import register_combiner

@register_combiner('DynamicWeight', description='动态权重合成策略')
class DynamicWeightCombiner(BaseFactorCombiner):
    def __init__(self, factor_weights: dict, **kwargs):
        self.factor_weights = factor_weights
    
    def update_weights(self, new_weights: dict):
        self.factor_weights = new_weights
    
    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        weights_series = pd.Series(self.factor_weights)
        aligned_weights = weights_series.reindex(standardized_df.columns).fillna(0)
        return (standardized_df * aligned_weights).sum(axis=1)
