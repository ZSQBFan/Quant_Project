"""总资产周转率因子"""
import pandas as pd
import numpy as np
import logging
from .complex_factor_base import ComplexFactorBase
from core.registry import register_factor


@register_factor('AssetTurnover', category='complex', description='总资产周转率')
class AssetTurnoverFactor(ComplexFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "AssetTurnover"
        required_cols = ['total_revenue', 'total_assets']
        if not all(col in all_data_df.columns for col in required_cols):
            logging.error(f"❌ [{factor_name}] 缺少必需列")
            return None

        logging.info(f"    > ⚙️ 计算: {factor_name}...")

        revenue = all_data_df['total_revenue']
        assets = all_data_df['total_assets']

        # 计算总资产周转率
        ato = revenue / assets.replace(0, np.nan)
        ato = ato.replace([np.inf, -np.inf], np.nan)
        ato.name = factor_name

        return ato.sort_index()
