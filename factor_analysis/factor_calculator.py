# factor_calculator.py (修改后的版本)

import pandas as pd
from data.data_manager import DataProviderManager
import warnings
from .factors import FACTOR_REGISTRY  # <--- 核心改变：从新文件导入

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class FactorCalculator:
    """
    为单因子分析准备核心数据 (已重构)。
    
    职责:
    1. 从 FACTOR_REGISTRY 获取指定的因子计算函数。
    2. 对股票池中的每只股票，调用该函数计算其每日的因子值。
    3. 计算未来N日的收益率 (forward returns)。
    4. 将所有数据合并成一个大的DataFrame。
    """

    def __init__(
            self,
            data_manager: DataProviderManager,
            universe: list,
            start_date: str,
            end_date: str,
            # 将 signal_class 改为 factor_name (字符串)
            factor_name: str,
            # 将 signal_params 改为 factor_params
            factor_params: dict,
            forward_return_periods: list):
        self.data_manager = data_manager
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date

        # --- 核心改变 ---
        self.factor_name = factor_name
        self.factor_params = factor_params
        if factor_name not in FACTOR_REGISTRY:
            raise ValueError(f"因子 '{factor_name}' 未在 FACTOR_REGISTRY 中注册。")
        self.factor_func = FACTOR_REGISTRY[factor_name]
        # --- 结束 ---

        self.periods = forward_return_periods

    # `_calculate_signal_in_pandas` 方法已被完全移除

    def calculate_factor_and_returns(self) -> pd.DataFrame:
        print("开始计算因子值和未来收益率...")
        all_factor_data = []

        for i, symbol in enumerate(self.universe):
            print(f"  处理中: {symbol} ({i+1}/{len(self.universe)})...", end='\r')

            stock_df = self.data_manager.get_dataframe(symbol)
            if stock_df is None or stock_df.empty or 'close' not in stock_df.columns:
                continue

            # --- 核心改变：动态调用因子函数 ---
            # **是 **kwargs 将参数字典解包传入函数
            stock_df['factor_value'] = self.factor_func(
                stock_df, **self.factor_params)
            # --- 结束 ---

            for p in self.periods:
                stock_df[f'forward_return_{p}d'] = (
                    stock_df['close'].shift(-p) / stock_df['close']) - 1

            stock_df['asset'] = symbol
            required_cols = ['date', 'asset', 'factor_value'] + [
                f'forward_return_{p}d' for p in self.periods
            ]
            all_factor_data.append(stock_df.reset_index()[required_cols])

        print("\n所有股票数据处理完毕，正在合并...")

        # 在合并前，检查列表是否为空
        if not all_factor_data:
            print("⚠️ 警告: 未找到任何有效数据进行处理，将返回一个空的DataFrame。")
            return pd.DataFrame()  # 如果没有数据，直接返回空DF，防止concat报错
        # --- 修复结束 ---

        # 4. 合并所有股票数据为一个大的DataFrame
        final_df = pd.concat(all_factor_data, ignore_index=True)

        # 5. 清理数据并返回
        final_df.dropna(inplace=True)
        final_df.set_index('date', inplace=True)

        return final_df
