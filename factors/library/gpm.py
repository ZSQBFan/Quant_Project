"""毛利率因子 (GPM)"""
import pandas as pd
import numpy as np
import logging
from .complex_factor_base import ComplexFactorBase
from core.registry import register_factor


@register_factor('GPM', category='complex', description='毛利率')
class GPMFactor(ComplexFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "GPM"
        required_cols = ['total_revenue', 'cost_of_goods_sold']
        if not all(col in all_data_df.columns for col in required_cols):
            logging.error(f"❌ [{factor_name}] 缺少必需列")
            return None

        logging.info(f"    > ⚙️ 计算: {factor_name}...")

        # 计算毛利润
        gross_profit = all_data_df['total_revenue'] - all_data_df['cost_of_goods_sold']

        # 计算毛利率
        gpm = gross_profit / all_data_df['total_revenue'].replace(0, np.nan)
        gpm.name = factor_name

        return gpm.sort_index()
