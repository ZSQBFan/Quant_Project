"""行业中性营收增长率因子"""
import pandas as pd
import logging
from .base import IndustryNeutralFactorBase
from core.registry import register_factor

@register_factor('IndNeu_SalesGrowth', category='complex', description='行业中性营收同比增长率')
class IndNeuSalesGrowthFactor(IndustryNeutralFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "IndNeu_SalesGrowth"
        if 'total_revenue' not in all_data_df.columns:
            logging.error(f"❌ [{factor_name}] 缺少 total_revenue 列")
            return None
        
        logging.info(f"    > ⚙️ 计算: {factor_name}...")
        
        sales_growth = all_data_df.groupby(level='asset')['total_revenue'].pct_change(periods=252, fill_method=None) * 100
        
        temp_df = all_data_df.copy()
        temp_df['base_sales_growth'] = sales_growth
        
        return self.neutralize_by_industry(temp_df, 'base_sales_growth', factor_name)
