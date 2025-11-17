# core/rolling_weight_calculator.py

import pandas as pd
import numpy as np
from . import analysis_metrics as metrics
from typing import Dict, List
from sklearn.linear_model import LinearRegression
import logging
from abc import ABC, abstractmethod  # 引入抽象基类工具
from tqdm import tqdm  # 引入tqdm来显示进度


class RollingCalculatorBase(ABC):
    """
    滚动计算器的抽象基类。

    它定义了“每日滚动合成”的通用算法框架（模板方法）：
    1. 逐日循环遍历所有交易日。
    2. 为每一天切分出对应的历史数据窗口。
    3. 调用一个由子类实现的【抽象方法】_calculate_new_weights来计算当日权重。
    4. 使用得到的权重对当日因子截面进行加权合成。
    5. 最终将所有每日结果拼接成完整的复合因子序列。
    """

    def __init__(self, factor_names: List[str], rolling_window_days: int):
        self.factor_names = factor_names
        self.rolling_window_days = rolling_window_days
        logging.info(f"--- RollingCalculatorBase 已初始化 ---")
        logging.info(f"    > 监控的因子: {self.factor_names}")
        logging.info(f"    > 滚动窗口: {self.rolling_window_days} 个交易日")

    @abstractmethod
    def _calculate_new_weights(
            self, historical_data_window: pd.DataFrame) -> Dict[str, float]:
        """
        【【【抽象方法】】】: 子类必须实现这个方法。

        在给定的历史数据窗口上，计算所有因子的新权重。
        这是模板方法中需要被填充的具体步骤。
        """
        pass

    def calculate_composite_factor(self,
                                   all_data_merged: pd.DataFrame) -> pd.Series:
        """
        【【【模板方法】】】: 这是外部调用的主入口。

        执行完整的每日滚动合成流程。
        """
        logging.info(f"⚙️ 开始每日滚动合成 (策略: {self.__class__.__name__})...")

        all_dates = all_data_merged.index.get_level_values(
            'date').unique().sort_values()

        # ==================== BUG 修复区域开始 ====================
        # 【【【修改】】】: 将 list 改为 dict，以保存日期信息
        composite_factor_parts = {}

        for i in tqdm(range(len(all_dates)),
                      desc=f"[{self.__class__.__name__}] 每日计算"):
            current_date = all_dates[i]

            # 1. 确定并切分历史数据窗口
            start_index = max(0, i - self.rolling_window_days)
            window_start_date = all_dates[start_index]
            historical_data_window = all_data_merged.loc[
                window_start_date:current_date]

            if historical_data_window.index.get_level_values(
                    'date').nunique() < 10:
                continue

            # 2. 调用子类的具体实现来计算权重
            current_weights = self._calculate_new_weights(
                historical_data_window)

            if not current_weights or sum(
                    abs(w) for w in current_weights.values()) == 0:
                logging.debug(f"  > {current_date}: 权重为空或全为0，跳过当天合成。")
                continue

            # 3. 获取当天因子截面并加权
            current_day_factors = all_data_merged.loc[current_date][
                self.factor_names]
            weights_series = pd.Series(current_weights).reindex(
                current_day_factors.columns, fill_value=0)
            daily_composite_factor = (current_day_factors *
                                      weights_series).sum(axis=1)

            # 【【【修改】】】: 将每日结果存入字典，以 current_date 为键
            composite_factor_parts[current_date] = daily_composite_factor

        if not composite_factor_parts:
            logging.error("❌ 滚动合成未能计算出任何结果。")
            return pd.Series(dtype=float, name="factor_value")

        # 【【【修改】】】: 使用字典进行 concat，Pandas 会自动将键作为外层索引
        final_composite_factor = pd.concat(composite_factor_parts)
        final_composite_factor.index.names = ['date', 'asset']  # 明确命名索引
        # ==================== BUG 修复区域结束 ====================

        logging.info("✅ 每日滚动合成完成。")
        return final_composite_factor


class RollingICIRCalculator(RollingCalculatorBase):
    """
    基于 IC/IR 的滚动权重计算器。
    它继承了基类的滚动框架，只需实现自己的权重计算逻辑。
    """

    def __init__(self, factor_weight_config: Dict[str, str],
                 forward_return_periods: List[int], factor_names: List[str],
                 rolling_window_days: int):
        super().__init__(factor_names=factor_names,
                         rolling_window_days=rolling_window_days)
        self.config = factor_weight_config
        self.periods = forward_return_periods
        logging.info("--- RollingICIRCalculator 已初始化 ---")
        logging.info(f"    > 权重配置: {self.config}")

    def _parse_metric_str(self, metric_str: str) -> (str, int):
        parts = metric_str.split('_')
        period_str = parts[-1]
        if period_str.endswith('d'):
            try:
                period = int(period_str[:-1])
                metric_key = '_'.join(parts[:-1])
                if metric_key == '': metric_key = 'ir'
                return metric_key, period
            except ValueError:
                pass
        default_period = self.periods[0]
        logging.warning(
            f"  > ⚠️  指标 '{metric_str}' 未指定周期或格式错误, 自动使用 {default_period}d")
        return metric_str.replace(f"_{default_period}d", ""), default_period

    def _calculate_new_weights(
            self, historical_data_window: pd.DataFrame) -> Dict[str, float]:
        logging.debug("  > ⚙️ [RollingICIR] 正在计算新权重...")
        raw_weights = {}
        for factor_name in self.factor_names:
            metric_str = self.config.get(factor_name, 'ir_5d')
            metric_key, period = self._parse_metric_str(metric_str)
            return_col = f'forward_return_{period}d'

            ic_calc_data = historical_data_window[[factor_name,
                                                   return_col]].copy()
            ic_calc_data.rename(columns={factor_name: 'factor_value'},
                                inplace=True)

            ic_series = metrics.calculate_rank_ic_series(ic_calc_data.dropna(),
                                                         period=period)
            stats = metrics.analyze_ic_statistics(ic_series)

            if metric_key in stats:
                raw_weights[factor_name] = stats[metric_key]
            else:
                logging.warning(
                    f"  > ⚠️ [RollingICIR] 指标 '{metric_key}' 无法从 {stats.keys()} 中找到。因子 {factor_name} 权重设为 0"
                )
                raw_weights[factor_name] = 0.0

        total_strength = sum(abs(w) for w in raw_weights.values())
        if total_strength == 0:
            logging.warning("  > ⚠️ [RollingICIR] 滚动窗口内所有因子的IC/IR指标(绝对值)均为0。")
            return {factor: 0.0 for factor in self.factor_names}

        normalized_weights = {
            factor: weight / total_strength
            for factor, weight in raw_weights.items()
        }
        logging.debug(f"  > ✅ [RollingICIR] 新权重计算完毕: {normalized_weights}")
        return normalized_weights


class RollingRegressionCalculator(RollingCalculatorBase):
    """
    基于多元线性回归的滚动权重计算器。
    """

    def __init__(self, target_return_period: int, factor_names: List[str],
                 rolling_window_days: int, **kwargs):
        super().__init__(factor_names=factor_names,
                         rolling_window_days=rolling_window_days)
        self.target_period = target_return_period
        self.return_col = f'forward_return_{self.target_period}d'
        logging.info("--- RollingRegressionCalculator 已初始化 ---")
        logging.info(f"    > 目标收益 (Y): {self.return_col}")

    def _calculate_new_weights(
            self, historical_data_window: pd.DataFrame) -> Dict[str, float]:
        logging.debug("  > ⚙️ [RollingRegression] 正在计算新权重...")

        relevant_cols = self.factor_names + [self.return_col]
        data = historical_data_window[relevant_cols].dropna()

        min_samples_needed = len(self.factor_names) + 2
        if data.empty or len(data) < min_samples_needed:
            logging.warning(
                f"  > ⚠️ [RollingRegression] 滚动窗口内数据不足 (需 {min_samples_needed}，实际 {len(data)})，无法执行回归。"
            )
            return {factor: 0.0 for factor in self.factor_names}

        X = data[self.factor_names]
        y = data[self.return_col]

        try:
            model = LinearRegression()
            model.fit(X, y)
        except Exception as e:
            logging.error(f"  > ❌ [RollingRegression] 线性回归拟合失败: {e}",
                          exc_info=True)
            return {factor: 0.0 for factor in self.factor_names}

        raw_weights = {
            name: coef
            for name, coef in zip(self.factor_names, model.coef_)
        }

        total_strength = sum(abs(w) for w in raw_weights.values())
        if total_strength == 0:
            logging.warning(
                "  > ⚠️ [RollingRegression] 滚动窗口内所有因子的回归系数(绝对值)均为0。")
            return {factor: 0.0 for factor in self.factor_names}

        normalized_weights = {
            factor: weight / total_strength
            for factor, weight in raw_weights.items()
        }
        logging.debug(f"    > [RollingRegression] 归一化权重: {normalized_weights}")
        return normalized_weights
