"""
ADX/DMI因子

平均趋向指数 - 趋势强度和方向
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


@register_factor('ADXDMI', category='simple', description='ADX/DMI趋势因子')
class ADXDMIFactor(BaseFactor):
    """
    ADX/DMI因子

    计算公式：
    - direction_strength = (DI+ - DI-) / (DI+ + DI-)
    - trend_strength_weight = ADX / 100
    - 因子值 = direction_strength * trend_strength_weight (当ADX > threshold)
    """

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        计算ADX/DMI因子

        Args:
            df: 包含 'high', 'low', 'close' 列的 DataFrame

        Returns:
            ADX/DMI因子值
        """
        if ta is None:
            self.logger.error("❌ 无法计算 ADXDMI，因为 'pandas-ta' 未加载。")
            raise ImportError("pandas-ta 未安装")

        if not self.validate_input(df, ['high', 'low', 'close']):
            return pd.Series(0, index=df.index)

        period = self.params.get('period', 14)
        trend_threshold = self.params.get('trend_threshold', 20)

        dmi = ta.adx(df['high'], df['low'], df['close'], length=period)

        if dmi is None or dmi.empty:
            self.logger.warning(f"⚠️ [ADXDMI] (Period={period}) 计算返回 None")
            return pd.Series(0, index=df.index)

        try:
            adx_col = [col for col in dmi.columns if 'ADX' in col][0]
            plus_di_col = [col for col in dmi.columns if 'DMP' in col][0]
            minus_di_col = [col for col in dmi.columns if 'DMN' in col][0]
        except IndexError:
            self.logger.error(
                f"❌ [ADXDMI] 无法在 {dmi.columns.tolist()} 中找到 ADX/DMP/DMN。"
            )
            return pd.Series(0, index=df.index)

        adx = dmi[adx_col]
        plus_di = dmi[plus_di_col]
        minus_di = dmi[minus_di_col]

        direction_strength = (plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        trend_strength_weight = adx / 100.0

        factor_series = pd.Series(
            np.where(
                adx > trend_threshold,
                direction_strength * trend_strength_weight,
                0.0
            ),
            index=df.index
        )

        return factor_series
