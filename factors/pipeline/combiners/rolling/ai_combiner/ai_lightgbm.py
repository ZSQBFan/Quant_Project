"""
滚动 AI 训练器

使用机器学习模型（如 LightGBM）在滚动窗口内训练预测模型。
"""

import pandas as pd
import logging
from typing import Any

from core.abstractions import RollingCalculatorBase, AITrainerBase


class RollingAITrainer(RollingCalculatorBase):
    """
    滚动 AI 训练器。

    使用机器学习模型在滚动窗口内学习因子与远期收益的关系，
    直接输出预测值作为合成因子。
    """

    def __init__(self, training_calculator: AITrainerBase, **kwargs):
        """
        初始化滚动 AI 训练器。

        Args:
            training_calculator: AI 训练器实例（需实现 AITrainerBase 接口）
        """
        super().__init__(**kwargs)
        self.trainer = training_calculator

    def _calculate_payload_for_day(
            self, historical_data_window: pd.DataFrame) -> Any:
        """
        在滚动窗口内训练模型。

        Args:
            historical_data_window: 历史数据窗口

        Returns:
            训练好的模型
        """
        return self.trainer.train_model(historical_data_window,
                                        self.factor_names)

    def _combine_factors_for_day(self, payload: Any,
                                 daily_factors: pd.DataFrame) -> pd.Series:
        """
        使用训练好的模型进行预测。

        Args:
            payload: 训练好的模型
            daily_factors: 当日因子值 DataFrame

        Returns:
            模型预测值 Series
        """
        model = payload
        try:
            features = model.feature_name_ if hasattr(
                model, 'feature_name_') else daily_factors.columns
            X_today = daily_factors[features]
            return pd.Series(model.predict(X_today), index=X_today.index)
        except Exception as e:
            logging.error(f"  > ❌ [RollingAITrainer] 预测失败: {e}")
            return pd.Series(dtype=float)
