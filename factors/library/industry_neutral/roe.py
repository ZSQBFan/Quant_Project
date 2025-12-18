"""行业中性净资产收益率因子"""
import pandas as pd
import numpy as np
import logging
from .base import IndustryNeutralFactorBase
from core.registry import register_factor

@register_factor('IndNeu_ROE', category='complex', description='行业中性净资产收益率')
class IndNeuROEFactor(IndustryNeutralFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "IndNeu_ROE"
        required_cols = ['net_profit_parent', 'total_equity_parent']
        if not all(col in all_data_df.columns for col in required_cols):
            logging.error(f"❌ [{factor_name}] 缺少必需列")
            return None

        logging.info(f"    > ⚙️ 计算: {factor_name}...")

        net_profit = all_data_df['net_profit_parent']
        equity = all_data_df['total_equity_parent']

        # 计算 ROE
        raw_roe = net_profit / equity.replace(0, np.nan)
        raw_roe = raw_roe.replace([np.inf, -np.inf], np.nan)

        temp_df = all_data_df.copy()
        temp_df['base_roe'] = raw_roe

        return self.neutralize_by_industry(temp_df, 'base_roe', factor_name)
