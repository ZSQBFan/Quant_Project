# factor_analysis/factor_calculator.py

import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from data.data_manager import DataProviderManager
from . import factors as factor_library


class FactorCalculator:

    def __init__(self,
                 provider_configs,
                 db_path,
                 universe,
                 start_date,
                 end_date,
                 factor_name,
                 factor_params,
                 num_threads=8,
                 required_columns=None):  # 【【【新增参数】】】
        self.provider_configs = provider_configs
        self.db_path = db_path
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.factor_name = factor_name
        self.factor_params = factor_params
        self.num_threads = num_threads
        self.required_columns = required_columns  # 【【【保存参数】】】

        self.data_manager = DataProviderManager(
            provider_configs=self.provider_configs,
            db_path=self.db_path,
            symbols=self.universe,
            start_date=self.start_date,
            end_date=self.end_date)

    def _process_single_symbol(self, symbol: str) -> pd.DataFrame:
        try:
            # 【【【修改】】】: 传递 required_columns
            df = self.data_manager.get_dataframe(symbol,
                                                 columns=self.required_columns)
            if df is None or df.empty:
                return None

            calc_func = getattr(
                factor_library, f"calculate_{self.factor_name.lower()}_factor",
                None)
            if not calc_func:
                logging.error(
                    f"❌ 因子 '{self.factor_name}' 的计算函数未在 factors.py 中找到。")
                return None

            factor_series = calc_func(df, **self.factor_params)

            result_df = pd.DataFrame({
                'asset': symbol,
                'factor_value': factor_series
            }).dropna()

            return result_df
        except Exception as e:
            logging.error(f"❌ 处理股票 {symbol} 时发生错误: {e}", exc_info=True)
            return None

    def calculate_factor(self) -> pd.DataFrame:
        # (保持不变)
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
