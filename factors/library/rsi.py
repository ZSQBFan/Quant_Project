"""
RSI因子 (Relative Strength Index)

相对强弱指数 - 均值回归因子
"""

import pandas as pd
import numpy as np
import logging
from .base import BaseFactor
from core.registry import register_factor

try:
    import pandas_ta as ta
except ImportError:
    logging.critical("⛔ 致命错误: `pandas-ta` 库未安装。请运行 'pip install pandas-ta' 进行安装。")
    ta = None


@register_factor('RSI', category='simple', description='相对强弱指数')
class RSIFactor(BaseFactor):
    """
    RSI因子

    计算公式：因子值 = 50 - RSI
    当 RSI 越低（超卖），因子值越高，符合"低买高卖"。
    """

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        计算RSI因子

        Args:
            df: 包含 'close' 列的 DataFrame

        Returns:
            RSI因子值
        """
        if ta is None:
            self.logger.error("❌ 无法计算 RSI，因为 'pandas-ta' 未加载。")
            raise ImportError("pandas-ta 未安装")

        if not self.validate_input(df, ['close']):
            return pd.Series([0] * len(df), index=df.index)

        rsi_period = self.params.get('rsi_period', 14)

        rsi_series = ta.rsi(df['close'], length=rsi_period)

        if rsi_series is None:
            self.logger.warning(f"⚠️ [RSI] (Period={rsi_period}) 计算返回 None (可能数据不足)。")
            return pd.Series([0] * len(df), index=df.index)

        return 50 - rsi_series
