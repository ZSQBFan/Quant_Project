"""行业中性总资产周转率因子"""
import pandas as pd
import numpy as np
import logging
from .base import IndustryNeutralFactorBase
from core.registry import register_factor

@register_factor('IndNeu_AssetTurnover', category='complex', description='行业中性总资产周转率')
class IndNeuAssetTurnoverFactor(IndustryNeutralFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "IndNeu_AssetTurnover"
        required_cols = ['total_revenue', 'total_assets']
        if not all(col in all_data_df.columns for col in required_cols):
            logging.error(f"❌ [{factor_name}] 缺少必需列")
            return None

        logging.info(f"    > ⚙️ 计算: {factor_name}...")

        revenue = all_data_df['total_revenue']
        assets = all_data_df['total_assets']

        # 计算总资产周转率
        raw_ato = revenue / assets.replace(0, np.nan)
        raw_ato = raw_ato.replace([np.inf, -np.inf], np.nan)

        temp_df = all_data_df.copy()
        temp_df['base_ato'] = raw_ato

        return self.neutralize_by_industry(temp_df, 'base_ato', factor_name)
