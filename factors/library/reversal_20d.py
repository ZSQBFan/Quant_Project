"""
20日反转因子 (Reversal 20D)

基于加权收益率的反转因子
"""

import pandas as pd
import numpy as np
from .base import BaseFactor
from core.registry import register_factor


@register_factor('Reversal20D', category='simple', description='20日反转因子')
class Reversal20DFactor(BaseFactor):
    """
    20日反转因子

    使用指数衰减权重计算过去N日的加权收益，然后取反。
    """

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        计算20日反转因子

        Args:
            df: 包含 'close' 列的 DataFrame

        Returns:
            反转因子值
        """
        if not self.validate_input(df, ['close']):
            return pd.Series([0] * len(df), index=df.index)

        period = self.params.get('period', 40)
        decay = self.params.get('decay', 20.0)

        daily_returns = df['close'].pct_change(fill_method=None)
        daily_returns.replace(0.0, np.nan, inplace=True)

        # 计算指数衰减权重
        weights = 2.0 ** -(np.arange(period, 0, -1) / decay)

        def weighted_average(window_returns):
            current_weights = weights[-len(window_returns):]
            valid_mask = ~np.isnan(window_returns)
            valid_returns = window_returns[valid_mask]
            valid_weights = current_weights[valid_mask]

            if len(valid_returns) == 0:
                return np.nan

            sum_of_weights = np.sum(valid_weights)
            if sum_of_weights == 0:
                return np.nan

            weighted_sum = np.dot(valid_returns, valid_weights)
            return weighted_sum / sum_of_weights

        weighted_returns = daily_returns.rolling(
            window=period,
            min_periods=int(period / 2)
        ).apply(weighted_average, raw=True)

        # 反转：取负值
        reversal_factor = -1.0 * weighted_returns

        return reversal_factor
