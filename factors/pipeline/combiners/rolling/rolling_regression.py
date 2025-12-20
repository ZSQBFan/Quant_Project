"""
滚动回归计算器

使用线性回归在滚动窗口内学习因子权重。
"""

import pandas as pd
import logging
from typing import Dict, List

from sklearn.linear_model import LinearRegression
from factors.core.abstractions import RollingCalculatorBase


class RollingRegressionCalculator(RollingCalculatorBase):
    """
    滚动回归计算器。

    使用线性回归在滚动窗口内学习因子与远期收益的关系，
    据此动态调整因子权重。
    """

    def __init__(self, target_return_period: int, **kwargs):
        """
        初始化滚动回归计算器。

        Args:
            target_return_period: 目标远期收益周期
        """
        super().__init__(**kwargs)
        self.return_col = f'forward_return_{target_return_period}d'

    def _calculate_payload_for_day(
            self, historical_data_window: pd.DataFrame, current_date: pd.Timestamp) -> Dict[str, float]:
        """
        使用线性回归计算单日的因子权重。

        Args:
            historical_data_window: 历史数据窗口
            current_date: 当前计算日期

        Returns:
            因子权重字典
        """
        data = historical_data_window[self.factor_names +
                                      [self.return_col]].dropna()
        if len(data) < len(self.factor_names) + 2:
            logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingRegressionCalculator] 数据点不足({len(data)})，无法进行回归")
            return {f: 0.0 for f in self.factor_names}

        X = data[self.factor_names]
        y = data[self.return_col]

        model = LinearRegression().fit(X, y)
        w = {n: c for n, c in zip(self.factor_names, model.coef_)}

        tot_s = sum(abs(v) for v in w.values())
        return {f: v / tot_s if tot_s else 0.0 for f, v in w.items()}

    def _combine_factors_for_day(self, payload: Dict[str, float],
                                 daily_factors: pd.DataFrame) -> pd.Series:
        """
        根据权重合成因子。

        Args:
            payload: 因子权重字典
            daily_factors: 当日因子值 DataFrame

        Returns:
            合成后的因子值 Series
        """
        weights = pd.Series(payload).reindex(daily_factors.columns,
                                             fill_value=0)
        return (daily_factors * weights).sum(axis=1)
