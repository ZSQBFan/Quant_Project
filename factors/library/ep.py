"""市盈率倒数因子 (E/P)"""
import pandas as pd
import numpy as np
import logging
from .complex_factor_base import ComplexFactorBase
from core.registry import register_factor


@register_factor('EP', category='complex', description='市盈率倒数 (E/P)')
class EPFactor(ComplexFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "EP"
        required_cols = ['close', 'share_capital', 'net_profit_parent']
        if not all(col in all_data_df.columns for col in required_cols):
            logging.error(f"❌ [{factor_name}] 缺少必需列")
            return None

        logging.info(f"    > ⚙️ 计算: {factor_name}...")

        # 计算市值
        market_cap = all_data_df['close'] * all_data_df['share_capital']

        # 计算 E/P (净利润 / 市值)
        ep = all_data_df['net_profit_parent'] / market_cap.replace(0, np.nan)
        ep.name = factor_name

        return ep.sort_index()
