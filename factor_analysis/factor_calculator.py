# factor_calculator.py (已重构 - 多进程修正版 v2)

import pandas as pd
from data.data_manager import DataProviderManager
import warnings
from .factors import FACTOR_REGISTRY
import logging
from tqdm import tqdm
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# ==============================================================================
# 【【【 全局辅助函数 (用于多进程) 】】】
# ==============================================================================

_data_manager = None
_factor_func = None
_factor_name = None
_factor_params = None
_periods = None


def _init_worker(provider_configs: List, db_path: str, start_date: str,
                 end_date: str, factor_name: str, factor_params: Dict,
                 periods: List[int]):
    """
    【【【新增】】】
    每个子进程启动时会调用的初始化函数。
    
    【【重构日志】】:
    - 2025-11-10 (修正):
      - 在 DataProviderManager 初始化时传入 'auto_detect_universe=False'
        以阻止子进程重复打印“自动检测股票池”的日志。
    """
    global _data_manager, _factor_func, _factor_name, _factor_params, _periods

    try:
        # 1. 每个子进程创建自己的 DataProviderManager 实例
        _data_manager = DataProviderManager(
            provider_configs=provider_configs,
            symbols=[],
            start_date=start_date,
            end_date=end_date,
            db_path=db_path,
            auto_detect_universe=False  # <- 【【【核心修正】】】
        )

        # 2. 加载因子函数
        if factor_name not in FACTOR_REGISTRY:
            raise ValueError(f"子进程错误: 因子 '{factor_name}' 未注册。")
        _factor_func = FACTOR_REGISTRY[factor_name]

        # 3. 存储其他参数
        _factor_name = factor_name
        _factor_params = factor_params
        _periods = periods
        logging.info(
            f"  > ✅ Spawned-Worker-{os.getpid()} 已初始化 (因子: {factor_name})。")
    except Exception as e:
        logging.error(f"  > ❌ Spawned-Worker-{os.getpid()} 初始化失败: {e}",
                      exc_info=True)


def _process_single_symbol_mp(symbol: str) -> pd.DataFrame | None:
    """
    多进程版本的工作函数（在子进程中运行）。
    """
    try:
        if _data_manager is None:
            logging.error(
                f"  > ❌ Worker-{os.getpid()} 未能处理 {symbol}，因为 _data_manager 为 None。"
            )
            return None

        stock_df = _data_manager.get_dataframe(symbol)
        if stock_df is None or stock_df.empty or 'close' not in stock_df.columns:
            logging.debug(f"  > ℹ️ (跳过) 股票 {symbol} 数据为空或缺少 'close' 列。")
            return None

        stock_df['factor_value'] = _factor_func(stock_df, **_factor_params)

        for p in _periods:
            stock_df[f'forward_return_{p}d'] = (stock_df['close'].shift(-p) /
                                                stock_df['close']) - 1

        stock_df['asset'] = symbol
        required_cols = ['date', 'asset', 'factor_value'
                         ] + [f'forward_return_{p}d' for p in _periods]
        return stock_df.reset_index()[required_cols]
    except Exception as e:
        logging.error(f"  > ❌ 在处理 {symbol} (因子 {_factor_name}) 时出错: {e}",
                      exc_info=True)
        return None


def _process_single_symbol_for_returns_only_mp(
        symbol: str) -> pd.DataFrame | None:
    """
    多进程版本的工作函数（仅计算收益）。
    """
    try:
        if _data_manager is None:
            logging.error(
                f"  > ❌ Worker-{os.getpid()} 未能处理 {symbol} (仅收益)，因为 _data_manager 为 None。"
            )
            return None

        stock_df = _data_manager.get_dataframe(symbol)
        if stock_df is None or stock_df.empty or 'close' not in stock_df.columns:
            return None

        stock_df['factor_value'] = 0.0  # 占位符

        for p in _periods:
            stock_df[f'forward_return_{p}d'] = (stock_df['close'].shift(-p) /
                                                stock_df['close']) - 1

        stock_df['asset'] = symbol
        required_cols = ['date', 'asset', 'factor_value'
                         ] + [f'forward_return_{p}d' for p in _periods]
        return stock_df.reset_index()[required_cols]
    except Exception as e:
        logging.error(f"  > ❌ 在 (仅收益) 处理 {symbol} 时出错: {e}", exc_info=True)
        return None


# ==============================================================================
# 【【【 FactorCalculator 类本身 】】】
# ==============================================================================


class FactorCalculator:
    """
    【【重构日志】】:
    - 2025-11-10 (多进程修正版):
      - 修正 __init__ 签名：
        - 移除 'data_manager' 依赖。
        - 显式传入 'provider_configs' 和 'db_path'。
      - 确保 _init_worker 使用新参数。
    """

    def __init__(
            self,
            provider_configs: List[Dict],  # <- 替换 data_manager
            db_path: str,  # <- 新增
            universe: list,
            start_date: str,
            end_date: str,
            factor_name: str,
            factor_params: dict,
            forward_return_periods: list,
            num_threads: int = 16):

        self.provider_configs = provider_configs
        self.db_path = db_path

        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.factor_name = factor_name
        self.factor_params = factor_params

        if factor_name not in FACTOR_REGISTRY:
            logging.critical(
                f"⛔ 因子 '{factor_name}' 未在 FACTOR_REGISTRY (factors.py) 中注册。")
            raise ValueError(f"因子 '{factor_name}' 未在 FACTOR_REGISTRY 中注册。")

        self.periods = forward_return_periods
        self.num_processes = num_threads

        logging.info(
            f"  > FactorCalculator for '{self.factor_name}' 已初始化 (使用 {self.num_processes} 个进程)。"
        )

    def calculate_factor_and_returns(self,
                                     run_factor_calc: bool = True
                                     ) -> pd.DataFrame:
        """
        使用多进程 ProcessPoolExecutor。
        """
        if run_factor_calc:
            logging.info(
                f"⚙️ 正在(多进程)计算 (Type 1) 因子 '{self.factor_name}' 及未来收益...")
            target_func = _process_single_symbol_mp
            cols_to_check = ['factor_value'] + [
                f'forward_return_{p}d' for p in self.periods
            ]
        else:
            logging.info(
                f"⚙️ 正在(多进程) (仅) 计算未来收益 (因子: {self.factor_name} 被跳过)...")
            target_func = _process_single_symbol_for_returns_only_mp
            cols_to_check = [f'forward_return_{p}d' for p in self.periods]

        all_factor_data = []

        initializer_args = (self.provider_configs, self.db_path,
                            self.start_date, self.end_date, self.factor_name,
                            self.factor_params, self.periods)

        with ProcessPoolExecutor(max_workers=self.num_processes,
                                 initializer=_init_worker,
                                 initargs=initializer_args) as executor:

            futures = {
                executor.submit(target_func, symbol): symbol
                for symbol in self.universe
            }

            pbar = tqdm(
                as_completed(futures),
                total=len(self.universe),
                desc=f"因子计算 ({self.factor_name})",
                ncols=100,
                file=sys.stdout  # (保持与 logger_config.py 一致)
            )

            for future in pbar:
                try:
                    result_df = future.result()
                    if result_df is not None and not result_df.empty:
                        all_factor_data.append(result_df)
                except Exception as e:
                    symbol = futures[future]
                    logging.error(f"❌ 任务 {symbol} 在进程池中执行失败: {e}",
                                  exc_info=True)

        logging.info(f"✅ {self.factor_name}: 所有股票数据处理完毕，正在合并...")

        if not all_factor_data:
            logging.warning(
                f"⚠️ 警告 ({self.factor_name}): 未找到任何有效数据进行处理，将返回一个空的DataFrame。")
            return pd.DataFrame()

        final_df = pd.concat(all_factor_data, ignore_index=True)
        logging.info(f"  > 合并后共 {len(final_df)} 条总记录。")

        final_df.dropna(subset=cols_to_check, inplace=True)
        final_df.set_index('date', inplace=True)

        logging.info(f"  > 清理(dropna)后剩余 {len(final_df)} 条有效记录。")
        return final_df
