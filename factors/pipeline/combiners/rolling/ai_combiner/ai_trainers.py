# factors/pipeline/combiners/rolling/ai_combiner/ai_trainers.py
"""
AI 训练器实现

提供具体的 AI 模型训练器实现。
"""

import pandas as pd
import logging
from typing import Any, List

from factors.core.abstractions import AITrainerBase


class LightGBMTrainer(AITrainerBase):
    """使用 LightGBM 模型进行训练的具体实现。"""

    def train_model(self, historical_data_window: pd.DataFrame,
                    factor_names: List[str], current_date: pd.Timestamp) -> Any:
        """
        在历史数据窗口内训练 LightGBM 模型。

        Args:
            historical_data_window: 历史数据窗口，包含因子值和未来收益
            factor_names: 因子名称列表
            current_date: 当前计算日期

        Returns:
            训练好的 LightGBM 模型，失败时返回 None
        """
        try:
            import lightgbm as lgb
        except ImportError:
            logging.error(f"  > ❌ [{current_date.date()}] [LightGBM] lightgbm 未安装，请运行 pip install lightgbm")
            return None

        relevant_cols = factor_names + [self.return_col]
        training_data = historical_data_window[relevant_cols].dropna()
        min_samples = len(factor_names) + 50

        if len(training_data) < min_samples:
            logging.warning(
                f"  > ⚠️ [{current_date.date()}] [LightGBM] 训练数据不足 (需>{min_samples}，实际{len(training_data)})。"
            )
            return None

        X_train, y_train = training_data[factor_names], training_data[self.return_col]

        try:
            model = lgb.LGBMRegressor(**self.model_params)
            model.fit(X_train, y_train)
            logging.info(f"  > ✅ [{current_date.date()}] [LightGBM] 新模型训练完毕。")
            return model
        except Exception as e:
            logging.error(f"  > ❌ [{current_date.date()}] [LightGBM] 模型训练失败: {e}", exc_info=True)
            return None


__all__ = ['LightGBMTrainer']
