# core/rolling_weight_calculator.py

import pandas as pd
from typing import Any, Dict, List
from abc import ABC, abstractmethod
from tqdm import tqdm
import logging
from sklearn.linear_model import LinearRegression
from . import analysis_metrics as metrics
from .ai_model_trainer import AITrainerBase


class RollingCalculatorBase(ABC):
    """
    【【【最终版基类】】】
    滚动计算器的抽象基类。
    它定义了滚动回测的通用算法框架，并将具体实现委托给子类。
    """

    def __init__(self,
                 factor_names: List[str],
                 rolling_window_days: int,
                 rebalance_frequency: str = 'MS'):
        """
        初始化滚动计算器基类。
        
        Args:
            factor_names (List[str]): 参与计算的因子名称列表。
            rolling_window_days (int): 计算时回看历史数据的天数。
            rebalance_frequency (str): 重新计算“载荷”的频率，遵循Pandas频率字符串。
        """
        self.factor_names = factor_names
        self.rolling_window_days = rolling_window_days
        self.rebalance_frequency = rebalance_frequency
        self.current_payload = None
        logging.info(f"--- {self.__class__.__name__} 基类已初始化 ---")
        logging.info(f"    > 回看窗口: {self.rolling_window_days} 天")
        logging.info(f"    > 载荷更新频率: {self.rebalance_frequency}")

    @abstractmethod
    def _calculate_payload_for_day(
            self, historical_data_window: pd.DataFrame) -> Any:
        pass

    @abstractmethod
    def _combine_factors_for_day(self, payload: Any,
                                 daily_factors: pd.DataFrame) -> pd.Series:
        pass

    def calculate_composite_factor(self,
                                   all_data_merged: pd.DataFrame) -> pd.Series:
        logging.info(f"⚙️ 开始每日滚动合成 (策略: {self.__class__.__name__})...")
        all_dates = all_data_merged.index.get_level_values(
            'date').unique().sort_values()
        composite_factor_parts = {}

        # 【【【核心修改】】】: 根据配置的频率动态计算调仓日
        ideal_rebalance_dates = pd.date_range(start=all_dates.min(),
                                              end=all_dates.max(),
                                              freq=self.rebalance_frequency)
        # 将理想调仓日映射到实际存在的交易日上
        rebalance_dates_idx = all_dates.searchsorted(ideal_rebalance_dates,
                                                     side='right') - 1
        rebalance_dates = set(
            all_dates[rebalance_dates_idx[rebalance_dates_idx >= 0]].date)
        logging.info(f"  > ℹ️ 已生成 {len(rebalance_dates)} 个调仓日。")

        for current_date in tqdm(all_dates,
                                 desc=f"[{self.__class__.__name__}] 每日计算"):
            if current_date.date() in rebalance_dates:
                window_end_date = current_date
                window_start_date = window_end_date - pd.DateOffset(
                    days=self.rolling_window_days)
                historical_window_mask = (
                    (all_data_merged.index.get_level_values('date')
                     >= window_start_date) &
                    (all_data_merged.index.get_level_values('date')
                     < window_end_date))
                historical_data_window = all_data_merged.loc[
                    historical_window_mask]

                if not historical_data_window.empty:
                    new_payload = self._calculate_payload_for_day(
                        historical_data_window)
                    if new_payload is not None:
                        self.current_payload = new_payload
                        logging.debug(
                            f"  >  pivotal: {current_date.date()} 载荷已更新。")
                else:
                    logging.warning(
                        f"  > ⚠️ {current_date.date()}: 历史窗口数据为空，无法更新载荷。")

            if self.current_payload is None:
                continue

            current_day_factors = all_data_merged.loc[current_date][
                self.factor_names]
            daily_composite_factor = self._combine_factors_for_day(
                self.current_payload, current_day_factors)

            if daily_composite_factor is not None:
                composite_factor_parts[current_date] = daily_composite_factor

        if not composite_factor_parts:
            logging.error("❌ 滚动合成未能计算出任何结果。")
            return pd.Series(dtype=float, name="factor_value")

        final_composite_factor = pd.concat(composite_factor_parts)
        final_composite_factor.index.names = ['date', 'asset']
        logging.info("✅ 每日滚动合成完成。")
        return final_composite_factor


class RollingICIRCalculator(RollingCalculatorBase):

    def __init__(self, factor_weight_config: Dict[str, str],
                 forward_return_periods: List[int], **kwargs):
        super().__init__(**kwargs)
        self.config = factor_weight_config
        self.periods = forward_return_periods

    def _parse_metric_str(self, metric_str: str) -> (str, int):
        parts = metric_str.split('_')
        period_str = parts[-1]
        if period_str.endswith('d'):
            try:
                return '_'.join(parts[:-1]) or 'ir', int(period_str[:-1])
            except ValueError:
                pass
        return metric_str, self.periods[0]

    def _calculate_payload_for_day(
            self, historical_data_window: pd.DataFrame) -> Dict[str, float]:
        raw_weights = {}
        for factor_name in self.factor_names:
            metric_str = self.config.get(factor_name, 'ir_5d')
            metric_key, period = self._parse_metric_str(metric_str)
            return_col = f'forward_return_{period}d'
            ic_calc_data = historical_data_window[[
                factor_name, return_col
            ]].rename(columns={factor_name: 'factor_value'})
            ic_series = metrics.calculate_rank_ic_series(ic_calc_data.dropna(),
                                                         period=period)
            stats = metrics.analyze_ic_statistics(ic_series)
            raw_weights[factor_name] = stats.get(metric_key, 0.0)
        total_strength = sum(abs(w) for w in raw_weights.values())
        return {
            f: w / total_strength if total_strength else 0.0
            for f, w in raw_weights.items()
        }

    def _combine_factors_for_day(self, payload: Dict[str, float],
                                 daily_factors: pd.DataFrame) -> pd.Series:
        weights = pd.Series(payload).reindex(daily_factors.columns,
                                             fill_value=0)
        return (daily_factors * weights).sum(axis=1)


class RollingRegressionCalculator(RollingCalculatorBase):

    def __init__(self, target_return_period: int, **kwargs):
        super().__init__(**kwargs)
        self.return_col = f'forward_return_{target_return_period}d'

    def _calculate_payload_for_day(
            self, historical_data_window: pd.DataFrame) -> Dict[str, float]:
        data = historical_data_window[self.factor_names +
                                      [self.return_col]].dropna()
        if len(data) < len(self.factor_names) + 2:
            return {f: 0.0 for f in self.factor_names}
        X, y = data[self.factor_names], data[self.return_col]
        model = LinearRegression().fit(X, y)
        raw_weights = {
            name: coef
            for name, coef in zip(self.factor_names, model.coef_)
        }
        total_strength = sum(abs(w) for w in raw_weights.values())
        return {
            f: w / total_strength if total_strength else 0.0
            for f, w in raw_weights.items()
        }

    def _combine_factors_for_day(self, payload: Dict[str, float],
                                 daily_factors: pd.DataFrame) -> pd.Series:
        weights = pd.Series(payload).reindex(daily_factors.columns,
                                             fill_value=0)
        return (daily_factors * weights).sum(axis=1)


class RollingAITrainer(RollingCalculatorBase):

    def __init__(self, training_calculator: AITrainerBase, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(training_calculator, AITrainerBase):
            raise TypeError("training_calculator 必须是 AITrainerBase 的实例。")
        self.trainer = training_calculator

    def _calculate_payload_for_day(
            self, historical_data_window: pd.DataFrame) -> Any:
        return self.trainer.train_model(historical_data_window,
                                        self.factor_names)

    def _combine_factors_for_day(self, payload: Any,
                                 daily_factors: pd.DataFrame) -> pd.Series:
        model = payload
        try:
            features = model.feature_name_ if hasattr(
                model, 'feature_name_') else daily_factors.columns
            X_today = daily_factors[features]
            return pd.Series(model.predict(X_today), index=X_today.index)
        except Exception as e:
            logging.error(f"  > ❌ [RollingAITrainer] 模型预测失败: {e}")
            return None
