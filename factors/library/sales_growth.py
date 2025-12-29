"""营收增长率因子"""
import pandas as pd
import logging
from .complex_factor_base import ComplexFactorBase
from core.registry import register_factor


@register_factor('SalesGrowth', category='complex', description='营收同比增长率')
class SalesGrowthFactor(ComplexFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "SalesGrowth"
        if 'total_revenue' not in all_data_df.columns:
            logging.error(f"❌ [{factor_name}] 缺少 total_revenue 列")
            return None

        logging.info(f"    > ⚙️ 计算: {factor_name}...")

        # 计算营收同比增长率 (252个交易日约为1年)
        sales_growth = all_data_df.groupby(level='asset')['total_revenue'].pct_change(
            periods=252, fill_method=None
        ) * 100
        sales_growth.name = factor_name

        return sales_growth.sort_index()
