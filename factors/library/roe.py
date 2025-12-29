"""净资产收益率因子 (ROE)"""
import pandas as pd
import numpy as np
import logging
from .complex_factor_base import ComplexFactorBase
from core.registry import register_factor


@register_factor('ROE', category='complex', description='净资产收益率')
class ROEFactor(ComplexFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "ROE"
        required_cols = ['net_profit_parent', 'total_equity_parent']
        if not all(col in all_data_df.columns for col in required_cols):
            logging.error(f"❌ [{factor_name}] 缺少必需列")
            return None

        logging.info(f"    > ⚙️ 计算: {factor_name}...")

        net_profit = all_data_df['net_profit_parent']
        equity = all_data_df['total_equity_parent']

        # 计算 ROE
        roe = net_profit / equity.replace(0, np.nan)
        roe = roe.replace([np.inf, -np.inf], np.nan)
        roe.name = factor_name

        return roe.sort_index()
