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
                 weight_update_frequency: str = 'MS',
                 **kwargs):
        self.factor_names = factor_names
        self.rolling_window_days = rolling_window_days
        self.weight_update_frequency = weight_update_frequency
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

        ideal_weight_update_dates = pd.date_range(start=all_dates.min(),
                                              end=all_dates.max(),
                                              freq=self.weight_update_frequency)
        weight_update_dates_idx = all_dates.searchsorted(ideal_weight_update_dates,
                                                     side='right') - 1
        weight_update_dates = set(
            all_dates[weight_update_dates_idx[weight_update_dates_idx >= 0]].date)
        logging.info(f"  > ℹ️ 已生成 {len(weight_update_dates)} 个权重更新日。")

        for current_date in tqdm(all_dates,
                                 desc=f"[{self.__class__.__name__}] 每日计算"):
            if current_date.date() in weight_update_dates:
                window_end_date = current_date
                window_start_date = window_end_date - pd.DateOffset(
                    days=self.rolling_window_days)
                
                # 【修复】确保窗口包含两端，并且只在有足够历史数据时计算
                historical_window_mask = (
                    (all_data_merged.index.get_level_values('date')
                     >= window_start_date) &
                    (all_data_merged.index.get_level_values('date')
                     <= window_end_date))  # 改为 <= 包含当前日期
                
                historical_data_window = all_data_merged.loc[
                    historical_window_mask]
                
                # 【修复】检查窗口内是否有足够的数据点
                min_days_required = max(30, self.rolling_window_days // 2)  # 至少需要30天或窗口一半的数据
                window_days = len(historical_data_window.index.get_level_values('date').unique())
                
                if not historical_data_window.empty and window_days >= min_days_required:
                    logging.debug(f"🔍 [滚动窗口] {current_date.date()}: 使用 {window_days} 天历史数据")
                    new_payload = self._calculate_payload_for_day(
                        historical_data_window)
                    if new_payload is not None:
                        self.current_payload = new_payload
                        logging.debug(
                            f"  >  {current_date.date()} 权重更新完成。")
                else:
                    logging.debug(
                        f"  > {current_date.date()}: 历史窗口数据不足"
                        f"(只有 {window_days} 天，需要至少 {min_days_required} 天)，跳过更新。")
                    
                    # 【修复】如果没有载荷，使用一个简单的等权策略作为回退
                    if self.current_payload is None:
                        logging.debug(f"  > 使用等权回退策略作为初始载荷")
                        self.current_payload = {f: 1.0/len(self.factor_names) for f in self.factor_names}

            if self.current_payload is None:
                continue

            current_day_factors = all_data_merged.loc[current_date][
                self.factor_names]
            
            # 【调试】检查当天的因子值
            logging.debug(f"🔍 [滚动合成] 日期 {current_date.date()}")
            logging.debug(f"🔍 [滚动合成] 载荷: {self.current_payload}")
            logging.debug(f"🔍 [滚动合成] 当天因子值形状: {current_day_factors.shape}")
            logging.debug(f"🔍 [滚动合成] 当天因子值类型: {type(current_day_factors)}")
            
            # 【关键】检查数据是否为空
            if current_day_factors.empty:
                logging.error(f"❌ [滚动合成] 错误: 当天因子数据为空！")
                continue
                
            # 【关键】检查是否有数据
            if len(current_day_factors) == 0:
                logging.error(f"❌ [滚动合成] 错误: 当天因子数据长度为0！")
                continue
                
            logging.debug(f"🔍 [滚动合成] 当天因子值列: {list(current_day_factors.columns)}")
            logging.debug(f"🔍 [滚动合成] 当天因子值索引: {current_day_factors.index[:5] if len(current_day_factors) > 0 else '空'}")
            
            # 检查是否有NaN
            nan_counts = current_day_factors.isna().sum()
            logging.debug(f"🔍 [滚动合成] NaN数量: {nan_counts.to_dict()}")
            
            # 检查数据类型
            logging.debug(f"🔍 [滚动合成] 数据类型:\n{current_day_factors.dtypes}")
            
            # 如果describe()没有输出，手动计算一些统计量
            if not current_day_factors.empty:
                logging.debug(f"🔍 [滚动合成] 非空值数量: {current_day_factors.count().to_dict()}")
                logging.debug(f"🔍 [滚动合成] 均值:\n{current_day_factors.mean()}")
                logging.debug(f"🔍 [滚动合成] 标准差:\n{current_day_factors.std()}")
                logging.debug(f"🔍 [滚动合成] 最小值:\n{current_day_factors.min()}")
                logging.debug(f"🔍 [滚动合成] 最大值:\n{current_day_factors.max()}")
            
            # 【关键调试】检查因子值是否全为0或NaN
            if (current_day_factors == 0).all().all():
                logging.warning(f"⚠️ [滚动合成] 警告: 当天所有因子值都为0！")
                
            if current_day_factors.isna().all().all():
                logging.error(f"❌ [滚动合成] 致命错误: 当天所有因子值都是NaN！")
                
            # 打印一些样本看看
            sample_data = current_day_factors.head(10)
            logging.debug(f"🔍 [滚动合成] 样本数据:\n{sample_data}")
            
            daily_composite_factor = self._combine_factors_for_day(
                self.current_payload, current_day_factors)
                
            # 【调试】检查合成后的因子值
            logging.debug(f"🔍 [滚动合成] 合成因子值形状: {daily_composite_factor.shape}")
            logging.debug(f"🔍 [滚动合成] 合成因子统计:\n{daily_composite_factor.describe()}")
            
            # 【关键】检查合成后是否全为0
            if (daily_composite_factor == 0).all():
                logging.warning(f"⚠️ [滚动合成] 警告: 合成因子全为0！")
            
            if daily_composite_factor is not None:
                composite_factor_parts[current_date] = daily_composite_factor

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
