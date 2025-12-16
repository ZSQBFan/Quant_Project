"""
均线交叉因子 (Moving Average Cross)

短期均线与长期均线的相对位置 - 趋势跟踪因子
"""

import pandas as pd
import numpy as np
from .base import BaseFactor
from core.registry import register_factor


@register_factor('MovingAverageCross', category='simple', description='均线交叉因子')
class MovingAverageCrossFactor(BaseFactor):
    """
    均线交叉因子

    计算公式：因子值 = (短期MA - 长期MA) / 长期MA * 100
    """

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        计算均线交叉因子

        Args:
            df: 包含 'close' 列的 DataFrame

        Returns:
            均线交叉因子值
        """
        if not self.validate_input(df, ['close']):
            return pd.Series([0] * len(df), index=df.index)

        short_ma = self.params.get('short_ma', 14)
        long_ma = self.params.get('long_ma', 30)

        short_ma_series = df['close'].rolling(window=short_ma).mean()
        long_ma_series = df['close'].rolling(window=long_ma).mean()

        safe_long_ma = long_ma_series.replace(0, np.nan)

        return (short_ma_series - long_ma_series) / safe_long_ma * 100
