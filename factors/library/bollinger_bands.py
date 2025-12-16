"""
布林带因子 (Bollinger Bands)

价格相对于布林带的位置 - 均值回归因子
"""

import pandas as pd
import numpy as np
import logging
from .base import BaseFactor
from core.registry import register_factor

try:
    import pandas_ta as ta
except ImportError:
    logging.critical("⛔ 致命错误: `pandas-ta` 库未安装。")
    ta = None


@register_factor('BollingerBands', category='simple', description='布林带指标')
class BollingerBandsFactor(BaseFactor):
    """
    布林带因子

    计算公式：因子值 = -(close - mid_band) / band_width
    价格越接近下轨，因子值越大（做多信号）
    """

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        计算布林带因子

        Args:
            df: 包含 'close' 列的 DataFrame

        Returns:
            布林带因子值
        """
        if ta is None:
            self.logger.error("❌ 无法计算 BollingerBands，因为 'pandas-ta' 未加载。")
            raise ImportError("pandas-ta 未安装")

        if not self.validate_input(df, ['close']):
            return pd.Series(0, index=df.index)

        period = self.params.get('period', 20)
        devfactor = self.params.get('devfactor', 2.0)

        bbands = ta.bbands(df['close'], length=period, std=devfactor)

        if bbands is None or bbands.empty:
            self.logger.warning(f"⚠️ [BollingerBands] (Period={period}) 计算返回 None")
            return pd.Series(0, index=df.index)

        try:
            mid_band_col = [col for col in bbands.columns if 'BBM' in col][0]
            top_band_col = [col for col in bbands.columns if 'BBU' in col][0]
        except IndexError:
            self.logger.error(
                f"❌ [BollingerBands] 无法在 {bbands.columns.tolist()} 中找到 'BBM' 或 'BBU'。"
            )
            return pd.Series(0, index=df.index)

        mid_band = bbands[mid_band_col]
        top_band = bbands[top_band_col]

        band_width = top_band - mid_band
        safe_band_width = band_width.replace(0, np.nan)

        return -(df['close'] - mid_band) / safe_band_width
