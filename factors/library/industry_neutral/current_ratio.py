"""行业中性流动比率因子"""
import pandas as pd
import numpy as np
import logging
from .base import IndustryNeutralFactorBase
from core.registry import register_factor

@register_factor('IndNeu_CurrentRatio', category='complex', description='行业中性流动比率')
class IndNeuCurrentRatioFactor(IndustryNeutralFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "IndNeu_CurrentRatio"
        required_cols = ['current_assets', 'current_liabilities']
        if not all(col in all_data_df.columns for col in required_cols):
            logging.error(f"❌ [{factor_name}] 缺少必需列")
            return None

        logging.info(f"    > ⚙️ 计算: {factor_name}...")

        ca = all_data_df['current_assets']
        cl = all_data_df['current_liabilities']

        # 计算流动比率
        raw_cr = ca / cl.replace(0, np.nan)
        raw_cr = raw_cr.replace([np.inf, -np.inf], np.nan)

        temp_df = all_data_df.copy()
        temp_df['base_cr'] = raw_cr

        return self.neutralize_by_industry(temp_df, 'base_cr', factor_name)
