"""行业中性经营现金流市价率因子"""
import pandas as pd
import numpy as np
import logging
from .base import IndustryNeutralFactorBase
from core.registry import register_factor

@register_factor('IndNeu_CFOP', category='complex', description='行业中性经营现金流市价率')
class IndNeuCFOPFactor(IndustryNeutralFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "IndNeu_CFOP"
        required_cols = ['net_cash_flow_ops', 'close', 'share_capital']
        if not all(col in all_data_df.columns for col in required_cols):
            logging.error(f"❌ [{factor_name}] 缺少必需列")
            return None
        
        logging.info(f"    > ⚙️ 计算: {factor_name}...")
        
        market_cap = all_data_df['close'] * all_data_df['share_capital']
        cfop = all_data_df['net_cash_flow_ops'] / market_cap.replace(0, np.nan)
        
        temp_df = all_data_df.copy()
        temp_df['base_cfop'] = cfop
        
        return self.neutralize_by_industry(temp_df, 'base_cfop', factor_name)
