"""
动量因子 (Momentum Factor)

计算价格动量 - N日价格变化率
"""

import pandas as pd
from .base import BaseFactor
from core.registry import register_factor


@register_factor('Momentum', category='simple', description='动量因子 - N日价格变化率')
class MomentumFactor(BaseFactor):
    """
    动量因子

    计算公式：(close[t] - close[t-period]) / close[t-period] * 100
    """

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        计算动量因子

        Args:
            df: 包含 'close' 列的 DataFrame

        Returns:
            动量因子值
        """
        if not self.validate_input(df, ['close']):
            return pd.Series([0] * len(df), index=df.index)

        period = self.params.get('period', 20)

        # 计算百分比变化率
        momentum = df['close'].pct_change(periods=period, fill_method=None) * 100

        return momentum
