"""
因子计算器模块

提供并行因子计算功能，支持新的注册表系统。
"""

import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import Optional, List, Callable

from data import DataProviderManager
from core.registry import get_factor


class FactorCalculator:
    """
    因子计算器，支持并行计算多个股票的因子值。

    使用新的注册表系统获取因子计算函数。
    """

    def __init__(self,
                 provider_configs,
                 db_path,
                 universe: List[str],
                 start_date: str,
                 end_date: str,
                 factor_name: str,
                 factor_params: dict,
                 num_threads: int = 8,
                 required_columns: Optional[List[str]] = None):
        """
        初始化因子计算器。

        Args:
            provider_configs: 数据提供者配置
            db_path: 数据库路径
            universe: 股票池列表
            start_date: 开始日期
            end_date: 结束日期
            factor_name: 因子名称
            factor_params: 因子参数字典
            num_threads: 并行线程数
            required_columns: 所需数据列
        """
        self.provider_configs = provider_configs
        self.db_path = db_path
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.factor_name = factor_name
        self.factor_params = factor_params
        self.num_threads = num_threads
        self.required_columns = required_columns

        self.data_manager = DataProviderManager(
            provider_configs=self.provider_configs,
            db_path=self.db_path,
            symbols=self.universe,
            start_date=self.start_date,
            end_date=self.end_date)

        # 获取因子计算函数
        self._factor_func = self._get_factor_function()

    def _get_factor_function(self) -> Optional[Callable]:
        """
        获取因子计算函数。

        优先使用新的注册表系统，如果找不到则尝试从旧的因子库加载。
        """
        # 尝试从新注册表获取
        try:
            factor_cls = get_factor(self.factor_name)
            if factor_cls is not None:
                logging.info(f"[FactorCalculator] 从注册表获取因子: {self.factor_name}")
                return factor_cls
        except Exception as e:
            logging.debug(f"[FactorCalculator] 从注册表获取因子失败: {e}")

        # 尝试从旧的因子库加载（向后兼容）
        try:
            # 动态导入旧的因子模块
            import importlib
            old_factors = importlib.import_module('old_code.factor_analysis.factors')
            calc_func = getattr(
                old_factors, f"calculate_{self.factor_name.lower()}_factor",
                None)
            if calc_func is not None:
                logging.info(f"[FactorCalculator] 从旧因子库获取因子: {self.factor_name}")
                return calc_func
        except Exception as e:
            logging.debug(f"[FactorCalculator] 从旧因子库获取因子失败: {e}")

        logging.error(f"❌ 因子 '{self.factor_name}' 的计算函数未找到。")
        return None

    def _process_single_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        处理单个股票的因子计算。

        Args:
            symbol: 股票代码

        Returns:
            包含因子值的 DataFrame，失败返回 None
        """
        try:
            df = self.data_manager.get_dataframe(symbol,
                                                 columns=self.required_columns)
            if df is None or df.empty:
                return None

            if self._factor_func is None:
                logging.error(
                    f"❌ 因子 '{self.factor_name}' 的计算函数未找到。")
                return None

            # 检查是否是类（新注册表系统）还是函数（旧系统）
            if isinstance(self._factor_func, type):
                # 新系统：实例化因子类并调用 calculate 方法
                factor_instance = self._factor_func(params=self.factor_params)
                factor_series = factor_instance.calculate(df)
            else:
                # 旧系统：直接调用函数
                factor_series = self._factor_func(df, **self.factor_params)

            # 创建结果 DataFrame，保留 factor_series 的 index（即 date）作为列
            result_df = pd.DataFrame({
                'asset': symbol,
                'factor_value': factor_series
            })
            result_df.index.name = 'date'
            result_df = result_df.reset_index()
            result_df = result_df.dropna(subset=['factor_value'])

            return result_df
        except Exception as e:
            logging.error(f"❌ 处理股票 {symbol} 时发生错误: {e}", exc_info=True)
            return None

    def calculate_factor(self) -> pd.DataFrame:
        """
        并行计算所有股票的因子值。

        Returns:
            包含所有股票因子值的 DataFrame
        """
        all_factor_data = []
        with ProcessPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(self._process_single_symbol, symbol): symbol
                for symbol in self.universe
            }

            pbar = tqdm(futures.keys(),
                        total=len(self.universe),
                        desc=f"[因子计算] {self.factor_name}")
            for future in pbar:
                try:
                    result_df = future.result()
                    if result_df is not None and not result_df.empty:
                        all_factor_data.append(result_df)
                except Exception as e:
                    symbol = futures[future]
                    logging.error(f"❌ 任务 {symbol} 在进程池中执行失败: {e}",
                                  exc_info=True)

        if not all_factor_data:
            return pd.DataFrame()

        final_df = pd.concat(all_factor_data, ignore_index=True)
        final_df.set_index('date', inplace=True)
        return final_df
