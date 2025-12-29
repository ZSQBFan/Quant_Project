"""流动比率因子"""
import pandas as pd
import numpy as np
import logging
from .complex_factor_base import ComplexFactorBase
from core.registry import register_factor


@register_factor('CurrentRatio', category='complex', description='流动比率')
class CurrentRatioFactor(ComplexFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "CurrentRatio"
        required_cols = ['current_assets', 'current_liabilities']
        if not all(col in all_data_df.columns for col in required_cols):
            logging.error(f"❌ [{factor_name}] 缺少必需列")
            return None

        logging.info(f"    > ⚙️ 计算: {factor_name}...")

        ca = all_data_df['current_assets']
        cl = all_data_df['current_liabilities']

        # 计算流动比率
        cr = ca / cl.replace(0, np.nan)
        cr = cr.replace([np.inf, -np.inf], np.nan)
        cr.name = factor_name

        return cr.sort_index()
