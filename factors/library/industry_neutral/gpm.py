"""行业中性毛利率因子"""
import pandas as pd
import numpy as np
import logging
from .base import IndustryNeutralFactorBase
from core.registry import register_factor

@register_factor('IndNeu_GPM', category='complex', description='行业中性毛利率')
class IndNeuGPMFactor(IndustryNeutralFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "IndNeu_GPM"
        required_cols = ['total_revenue', 'cost_of_goods_sold']
        if not all(col in all_data_df.columns for col in required_cols):
            logging.error(f"❌ [{factor_name}] 缺少必需列")
            return None
        
        logging.info(f"    > ⚙️ 计算: {factor_name}...")
        
        gross_profit = all_data_df['total_revenue'] - all_data_df['cost_of_goods_sold']
        gpm = gross_profit / all_data_df['total_revenue'].replace(0, np.nan)
        
        temp_df = all_data_df.copy()
        temp_df['base_gpm'] = gpm
        
        return self.neutralize_by_industry(temp_df, 'base_gpm', factor_name)
