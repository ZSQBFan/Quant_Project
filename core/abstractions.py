# core/abstractions.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import pandas as pd
import logging
from tqdm import tqdm  # <-- 【【【新增导入】】】

# ==============================================================================
#                 --- 框架接口定义 (Framework Interfaces) ---
# ==============================================================================


class BaseStandardizer(ABC):
    """【抽象基类】: 因子标准化器"""

    @abstractmethod
    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        pass


class BaseFactorCombiner(ABC):
    """【抽象基类】: 因子合成器"""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        pass


class RollingCalculatorBase(ABC):
    """【抽象基类】: 滚动计算器"""

    def __init__(self,
                 factor_names: List[str],
                 rolling_window_days: int,
                 rebalance_frequency: str = 'MS',
                 **kwargs):
        self.factor_names = factor_names
        self.rolling_window_days = rolling_window_days
        self.rebalance_frequency = rebalance_frequency
        self.current_payload = None

    @abstractmethod
    def _calculate_payload_for_day(
            self, historical_data_window: pd.DataFrame) -> Any:
        pass

    @abstractmethod
    def _combine_factors_for_day(self, payload: Any,
                                 daily_factors: pd.DataFrame) -> pd.Series:
        pass

    # 【【【修正】】】: 恢复了被遗漏的核心“模板方法”。
    # 这个方法定义了所有滚动策略共享的、统一的执行流程。
    def calculate_composite_factor(self,
                                   all_data_merged: pd.DataFrame) -> pd.Series:
        """
        【【【模板方法】】】: 这是外部调用的主入口，执行完整的滚动合成流程。
        """
        logging.info(f"⚙️ 开始每日滚动合成 (策略: {self.__class__.__name__})...")
        all_dates = all_data_merged.index.get_level_values(
            'date').unique().sort_values()
        composite_factor_parts = {}

        ideal_rebalance_dates = pd.date_range(start=all_dates.min(),
                                              end=all_dates.max(),
                                              freq=self.rebalance_frequency)
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


class AITrainerBase(ABC):
    """【抽象基类】: AI模型训练器"""

    def __init__(self, target_return_period: int, model_params: Dict[str, Any],
                 **kwargs):
        self.target_return_period = target_return_period
        self.return_col = f'forward_return_{self.target_return_period}d'
        self.model_params = model_params

    @abstractmethod
    def train_model(self, historical_data_window: pd.DataFrame,
                    factor_names: List[str]) -> Any:
        pass
