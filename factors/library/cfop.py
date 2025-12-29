"""经营现金流市价率因子 (CFOP)"""
import pandas as pd
import numpy as np
import logging
from .complex_factor_base import ComplexFactorBase
from core.registry import register_factor


@register_factor('CFOP', category='complex', description='经营现金流市价率')
class CFOPFactor(ComplexFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "CFOP"
        required_cols = ['net_cash_flow_ops', 'close', 'share_capital']
        if not all(col in all_data_df.columns for col in required_cols):
            logging.error(f"❌ [{factor_name}] 缺少必需列")
            return None

        logging.info(f"    > ⚙️ 计算: {factor_name}...")

        # 计算市值
        market_cap = all_data_df['close'] * all_data_df['share_capital']

        # 计算经营现金流市价率
        cfop = all_data_df['net_cash_flow_ops'] / market_cap.replace(0, np.nan)
        cfop.name = factor_name

        return cfop.sort_index()
