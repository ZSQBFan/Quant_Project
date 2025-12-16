"""行业中性动量因子"""
import pandas as pd
import logging
from .base import IndustryNeutralFactorBase
from core.registry import register_factor

@register_factor('IndNeu_Momentum', category='complex', description='行业中性动量因子')
class IndNeuMomentumFactor(IndustryNeutralFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "IndNeu_Momentum"
        if 'industry' not in all_data_df.columns or 'close' not in all_data_df.columns:
            logging.error(f"❌ [{factor_name}] 缺少必需列")
            return None
        
        logging.info(f"    > ⚙️ 计算: {factor_name}...")
        base_momentum = all_data_df.groupby(level='asset')['close'].pct_change(periods=20, fill_method=None) * 100
        
        temp_df = all_data_df.copy()
        temp_df['base_momentum'] = base_momentum
        
        return self.neutralize_by_industry(temp_df, 'base_momentum', factor_name)
