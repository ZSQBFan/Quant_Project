"""行业中性市净率倒数因子"""
import pandas as pd
import numpy as np
import logging
from .base import IndustryNeutralFactorBase
from core.registry import register_factor

@register_factor('IndNeu_BP', category='complex', description='行业中性市净率倒数 (B/P)')
class IndNeuBPFactor(IndustryNeutralFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "IndNeu_BP"
        required_cols = ['close', 'share_capital', 'total_equity_parent']
        if not all(col in all_data_df.columns for col in required_cols):
            logging.error(f"❌ [{factor_name}] 缺少必需列")
            return None
        
        logging.info(f"    > ⚙️ 计算: {factor_name}...")
        
        market_cap = all_data_df['close'] * all_data_df['share_capital']
        bp = all_data_df['total_equity_parent'] / market_cap.replace(0, np.nan)
        
        temp_df = all_data_df.copy()
        temp_df['base_bp'] = bp
        
        return self.neutralize_by_industry(temp_df, 'base_bp', factor_name)
