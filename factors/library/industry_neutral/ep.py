"""行业中性市盈率倒数因子"""
import pandas as pd
import numpy as np
import logging
from .base import IndustryNeutralFactorBase
from core.registry import register_factor

@register_factor('IndNeu_EP', category='complex', description='行业中性市盈率倒数 (E/P)')
class IndNeuEPFactor(IndustryNeutralFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "IndNeu_EP"
        required_cols = ['close', 'share_capital', 'net_profit_parent']
        if not all(col in all_data_df.columns for col in required_cols):
            logging.error(f"❌ [{factor_name}] 缺少必需列")
            return None
        
        logging.info(f"    > ⚙️ 计算: {factor_name}...")
        
        market_cap = all_data_df['close'] * all_data_df['share_capital']
        ep = all_data_df['net_profit_parent'] / market_cap.replace(0, np.nan)
        
        temp_df = all_data_df.copy()
        temp_df['base_ep'] = ep
        
        return self.neutralize_by_industry(temp_df, 'base_ep', factor_name)
