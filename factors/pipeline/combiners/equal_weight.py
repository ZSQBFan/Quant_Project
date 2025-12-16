"""等权重合成器"""
import pandas as pd
from .base import BaseFactorCombiner
from core.registry import register_combiner

@register_combiner('EqualWeight', description='等权重合成策略')
class EqualWeightCombiner(BaseFactorCombiner):
    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        return standardized_df.sum(axis=1)
