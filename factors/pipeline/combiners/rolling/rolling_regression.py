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
        # 添加数据质量检查
        if historical_data_window.empty:
            logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingRegressionCalculator] 历史数据窗口为空，返回None跳过计算")
            return None

        data = historical_data_window[self.factor_names +
                                      [self.return_col]].dropna()
        if len(data) < len(self.factor_names) + 2:
            logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingRegressionCalculator] 数据点不足({len(data)})，返回None跳过计算")
            return None

        X = data[self.factor_names]
        y = data[self.return_col]

        # 检查每个因子的数据质量
        valid_factors = []
        for fname in self.factor_names:
            factor_values = data[fname]
            return_values = data[self.return_col]
            
            # 检查因子值是否为常量
            if factor_values.nunique() <= 1:
                logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingRegressionCalculator] 因子 {fname} 值为常量，跳过该因子")
                continue
                
            # 检查收益率是否为常量
            if return_values.nunique() <= 1:
                logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingRegressionCalculator] 收益率 {self.return_col} 为常量，跳过因子 {fname}")
                continue
                
            valid_factors.append(fname)

        # 如果没有有效因子，返回None跳过计算
        if not valid_factors:
            logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingRegressionCalculator] 没有有效因子，返回None跳过计算")
            return None

        # 如果有效因子数量少于2个，无法进行有效的多变量回归
        if len(valid_factors) < 2:
            logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingRegressionCalculator] 有效因子数量不足({len(valid_factors)})，返回None跳过计算")
            return None

        # 使用有效因子进行回归
        X_valid = X[valid_factors]

        try:
            model = LinearRegression().fit(X_valid, y)
            w = {n: c for n, c in zip(valid_factors, model.coef_)}
            
            # 检查回归系数的合理性
            coef_values = list(w.values())
            max_coef = max(abs(c) for c in coef_values) if coef_values else 0
            
            # 如果系数过大，可能是数据异常，添加检查
            if max_coef > 100:  # 设置一个合理的阈值
                logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingRegressionCalculator] 回归系数过大(max={max_coef:.2f})，可能存在数据异常")
                # 对系数进行裁剪，避免极端值
                w = {k: v/max_coef * 10 for k, v in w.items()}
                logging.info(f"  > ℹ️ [{current_date.date()}] [RollingRegressionCalculator] 对系数进行裁剪处理")
            
        except Exception as e:
            logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingRegressionCalculator] 回归计算出错: {e}，返回None跳过计算")
            return None

        # 对无效因子权重设为0
        for fname in self.factor_names:
            if fname not in w:
                w[fname] = 0.0

        # 改进的权重归一化逻辑
        tot_s = sum(abs(v) for v in w.values())
        if tot_s == 0:
            logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingRegressionCalculator] 权重和为0，返回None跳过计算")
            return None
        
        if tot_s < 1e-10:  # 非常接近0的情况
            logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingRegressionCalculator] 权重和过小({tot_s:.2e})，返回None跳过计算")
            return None
        
        # 归一化权重
        normalized_weights = {f: v / tot_s for f, v in w.items()}
        
        # 最终检查：确保权重绝对值总和为1且没有NaN
        weight_abs_sum = sum(abs(v) for v in normalized_weights.values())
        if abs(weight_abs_sum - 1.0) > 1e-6 or any(pd.isna(v) for v in normalized_weights.values()):
            logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingRegressionCalculator] 权重归一化异常(abs_sum={weight_abs_sum:.6f})，返回None跳过计算")
            return None
        
        return normalized_weights

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
