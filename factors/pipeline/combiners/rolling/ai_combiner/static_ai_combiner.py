"""静态 AI 模型合成器"""
import pandas as pd
import logging
import joblib
from typing import Any
from ...base import BaseFactorCombiner
from core.registry import register_combiner

@register_combiner('StaticAICombiner', description='静态 AI 模型合成策略')
class StaticAICombiner(BaseFactorCombiner):
    """
    静态 AI 模型合成器。
    
    加载预训练好的模型进行因子合成，不进行在线训练。
    """
    def __init__(self, initial_model_path: str = None, **kwargs):
        self.model = None
        if initial_model_path:
            try:
                self.model = joblib.load(initial_model_path)
            except Exception as e:
                logging.warning(f"⚠️ 加载AI模型失败: {e}")
    
    def update_model(self, new_model: Any):
        if new_model:
            self.model = new_model
    
    def update_weights(self, new_model: Any):
        self.update_model(new_model)
    
    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            return pd.Series(0, index=standardized_df.index)
        try:
            features = self.model.feature_name_ if hasattr(self.model, 'feature_name_') else standardized_df.columns
            X_today = standardized_df[features]
            return pd.Series(self.model.predict(X_today), index=X_today.index)
        except Exception as e:
            logging.error(f"❌ [StaticAICombiner] 预测失败: {e}")
            return pd.Series(0, index=standardized_df.index)
