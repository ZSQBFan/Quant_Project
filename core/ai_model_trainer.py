# core/ai_model_trainer.py

import pandas as pd
import lightgbm as lgb
from abc import ABC, abstractmethod
from typing import Dict, Any
import logging


class AITrainerBase(ABC):
    """
    AI模型训练器的抽象基类。
    它的唯一职责是：接收一个历史数据窗口，返回一个训练好的模型。
    """

    def __init__(self, target_return_period: int, model_params: Dict[str,
                                                                     Any]):
        self.target_period = target_return_period
        self.return_col = f'forward_return_{self.target_period}d'
        self.model_params = model_params
        logging.info(f"--- {self.__class__.__name__} 已初始化 ---")
        logging.info(f"    > 目标收益 (Y): {self.return_col}")

    @abstractmethod
    def train_model(self, historical_data_window: pd.DataFrame,
                    factor_names: list):
        """
        在给定的历史数据窗口上，训练并返回一个新的模型对象。
        """
        pass


class LightGBMTrainer(AITrainerBase):
    """
    使用 LightGBM 模型进行训练的具体实现。
    """

    def train_model(self, historical_data_window: pd.DataFrame,
                    factor_names: list):
        logging.debug(
            f"  > ⚙️ [LightGBM] 正在使用 {len(historical_data_window)} 条记录训练新模型..."
        )

        relevant_cols = factor_names + [self.return_col]
        training_data = historical_data_window[relevant_cols].dropna()

        min_samples_needed = len(factor_names) + 50  # 确保有足够的样本进行训练
        if training_data.empty or len(training_data) < min_samples_needed:
            logging.warning(
                f"  > ⚠️ [LightGBM] 训练数据不足 (需 >{min_samples_needed}，实际 {len(training_data)})，跳过本次训练。"
            )
            return None

        X_train = training_data[factor_names]
        y_train = training_data[self.return_col]

        try:
            model = lgb.LGBMRegressor(**self.model_params)
            model.fit(X_train, y_train)
            logging.info(f"  > ✅ [LightGBM] 新模型训练完毕。")
            return model
        except Exception as e:
            logging.error(f"  > ❌ [LightGBM] 模型训练失败: {e}", exc_info=True)
            return None
