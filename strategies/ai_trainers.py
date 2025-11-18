# strategies/ai_trainers.py

import pandas as pd
import lightgbm as lgb
import logging
from core.abstractions import AITrainerBase


class LightGBMTrainer(AITrainerBase):
    """使用 LightGBM 模型进行训练的具体实现。"""

    def train_model(self, historical_data_window: pd.DataFrame,
                    factor_names: list):
        relevant_cols = factor_names + [self.return_col]
        training_data = historical_data_window[relevant_cols].dropna()
        min_samples = len(factor_names) + 50
        if len(training_data) < min_samples:
            logging.warning(
                f"  > ⚠️ [LightGBM] 训练数据不足 (需>{min_samples}，实际{len(training_data)})。"
            )
            return None
        X_train, y_train = training_data[factor_names], training_data[
            self.return_col]
        try:
            model = lgb.LGBMRegressor(**self.model_params)
            model.fit(X_train, y_train)
            logging.info("  > ✅ [LightGBM] 新模型训练完毕。")
            return model
        except Exception as e:
            logging.error(f"  > ❌ [LightGBM] 模型训练失败: {e}", exc_info=True)
            return None
