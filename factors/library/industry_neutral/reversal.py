"""行业中性反转因子"""
import pandas as pd
import numpy as np
import logging
from .base import IndustryNeutralFactorBase
from core.registry import register_factor

@register_factor('IndNeu_Reversal20D', category='complex', description='行业中性反转因子')
class IndNeuReversalFactor(IndustryNeutralFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "IndNeu_Reversal20D"
        if 'close' not in all_data_df.columns:
            logging.error(f"❌ [{factor_name}] 缺少 close 列")
            return None
        
        logging.info(f"    > ⚙️ 计算: {factor_name}...")
        
        daily_returns = all_data_df.groupby(level='asset')['close'].pct_change(fill_method=None)
        daily_returns = daily_returns.replace(0.0, np.nan)
        
        period = 40
        decay = 20.0
        weights = 2.0 ** -(np.arange(period, 0, -1) / decay)
        
        def weighted_avg(window):
            w = weights[-len(window):]
            mask = ~np.isnan(window)
            if not mask.any():
                return np.nan
            return np.dot(window[mask], w[mask]) / w[mask].sum()
        
        weighted_returns = daily_returns.groupby(level='asset').rolling(period, min_periods=int(period/2)).apply(weighted_avg, raw=True)
        weighted_returns = weighted_returns.droplevel(0)
        
        base_reversal = -1.0 * weighted_returns
        
        temp_df = all_data_df.copy()
        temp_df['base_reversal'] = base_reversal
        
        return self.neutralize_by_industry(temp_df, 'base_reversal', factor_name)
