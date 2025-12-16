"""行业中性成交量变异系数因子"""
import pandas as pd
import numpy as np
import logging
from .base import IndustryNeutralFactorBase
from core.registry import register_factor

@register_factor('IndNeu_VolumeCV', category='complex', description='行业中性成交量变异系数')
class IndNeuVolumeCVFactor(IndustryNeutralFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "IndNeu_VolumeCV"
        if 'volume' not in all_data_df.columns:
            logging.error(f"❌ [{factor_name}] 缺少 volume 列")
            return None
        
        logging.info(f"    > ⚙️ 计算: {factor_name}...")
        
        def cv(x):
            mean_val = x.mean()
            if mean_val == 0 or np.isnan(mean_val):
                return np.nan
            return x.std() / mean_val
        
        volume_cv = all_data_df.groupby(level='asset')['volume'].rolling(20).apply(cv, raw=False)
        volume_cv = volume_cv.droplevel(0)
        
        temp_df = all_data_df.copy()
        temp_df['base_volume_cv'] = volume_cv
        
        return self.neutralize_by_industry(temp_df, 'base_volume_cv', factor_name)
