"""
成交量突增因子 (Volume Spike)

成交量相对于历史平均的偏离程度
"""

import pandas as pd
import numpy as np
from .base import BaseFactor
from core.registry import register_factor


@register_factor('VolumeSpike', category='simple', description='成交量突增因子')
class VolumeSpikeFactor(BaseFactor):
    """
    成交量突增因子

    计算公式：因子值 = (当前成交量 - 平均成交量) / 平均成交量
    """

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        计算成交量突增因子

        Args:
            df: 包含 'volume' 列的 DataFrame

        Returns:
            成交量突增因子值
        """
        if not self.validate_input(df, ['volume']):
            return pd.Series([0] * len(df), index=df.index)

        period = self.params.get('period', 20)

        ma_vol = df['volume'].rolling(window=period).mean()
        safe_ma_vol = ma_vol.replace(0, np.nan)

        return (df['volume'] - ma_vol) / safe_ma_vol
