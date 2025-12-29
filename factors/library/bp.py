"""市净率倒数因子 (B/P)"""
import pandas as pd
import numpy as np
import logging
from .complex_factor_base import ComplexFactorBase
from core.registry import register_factor


@register_factor('BP', category='complex', description='市净率倒数 (B/P)')
class BPFactor(ComplexFactorBase):
    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        factor_name = "BP"
        required_cols = ['close', 'share_capital', 'total_equity_parent']
        if not all(col in all_data_df.columns for col in required_cols):
            logging.error(f"❌ [{factor_name}] 缺少必需列")
            return None

        logging.info(f"    > ⚙️ 计算: {factor_name}...")

        # 计算市值
        market_cap = all_data_df['close'] * all_data_df['share_capital']

        # 计算 B/P (净资产 / 市值)
        bp = all_data_df['total_equity_parent'] / market_cap.replace(0, np.nan)
        bp.name = factor_name

        return bp.sort_index()
