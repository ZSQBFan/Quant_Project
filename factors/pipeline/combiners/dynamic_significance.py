"""动态显著性加权合成器"""
import pandas as pd
import numpy as np
from .base import BaseFactorCombiner
from core.registry import register_combiner

@register_combiner('DynamicSignificance', description='动态显著性加权策略')
class DynamicSignificanceCombiner(BaseFactorCombiner):
    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        abs_significance = standardized_df.abs()
        total_significance = abs_significance.sum(axis=1).replace(0, np.finfo(float).eps)
        dynamic_weights = abs_significance.div(total_significance, axis=0)
        return (standardized_df * dynamic_weights).sum(axis=1)
