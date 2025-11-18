# strategies/combiners.py

import pandas as pd
import numpy as np
import logging
import joblib
from typing import Any
from core.abstractions import BaseFactorCombiner


class EqualWeightCombiner(BaseFactorCombiner):

    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        return standardized_df.sum(axis=1)


class FixedWeightCombiner(BaseFactorCombiner):

    def __init__(self, factor_weights: dict, **kwargs):
        self.factor_weights = factor_weights

    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        weights_series = pd.Series(self.factor_weights)
        aligned_weights = weights_series.reindex(
            standardized_df.columns).fillna(0)
        return (standardized_df * aligned_weights).sum(axis=1)


class DynamicSignificanceCombiner(BaseFactorCombiner):

    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        abs_significance = standardized_df.abs()
        total_significance = abs_significance.sum(axis=1).replace(
            0,
            np.finfo(float).eps)
        dynamic_weights = abs_significance.div(total_significance, axis=0)
        return (standardized_df * dynamic_weights).sum(axis=1)


class DynamicWeightCombiner(BaseFactorCombiner):

    def __init__(self, factor_weights: dict, **kwargs):
        self.factor_weights = factor_weights

    def update_weights(self, new_weights: dict):
        self.factor_weights = new_weights

    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        weights_series = pd.Series(self.factor_weights)
        aligned_weights = weights_series.reindex(
            standardized_df.columns).fillna(0)
        return (standardized_df * aligned_weights).sum(axis=1)


class AICombiner(BaseFactorCombiner):

    def __init__(self, initial_model_path: str = None, **kwargs):
        self.model = None
        if initial_model_path:
            try:
                self.model = joblib.load(initial_model_path)
            except Exception as e:
                logging.warning(f"  > ⚠️ 加载初始AI模型失败: {e}，将等待首次训练。")

    def update_model(self, new_model: Any):
        if new_model: self.model = new_model

    def update_weights(self, new_model: Any):
        self.update_model(new_model)

    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        if self.model is None: return pd.Series(0, index=standardized_df.index)
        try:
            features = self.model.feature_name_ if hasattr(
                self.model, 'feature_name_') else standardized_df.columns
            X_today = standardized_df[features]
            return pd.Series(self.model.predict(X_today), index=X_today.index)
        except Exception as e:
            logging.error(f"  > ❌ [AICombiner] 预测失败: {e}")
            return pd.Series(0, index=standardized_df.index)
