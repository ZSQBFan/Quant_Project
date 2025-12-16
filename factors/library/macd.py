"""
MACD因子

移动平均收敛发散指标 - 趋势跟踪因子
"""

import pandas as pd
import logging
from .base import BaseFactor
from core.registry import register_factor

try:
    import pandas_ta as ta
except ImportError:
    logging.critical("⛔ 致命错误: `pandas-ta` 库未安装。")
    ta = None


@register_factor('MACD', category='simple', description='MACD指标')
class MACDFactor(BaseFactor):
    """
    MACD因子

    计算公式：因子值 = MACD线 / close * 100
    """

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        计算MACD因子

        Args:
            df: 包含 'close' 列的 DataFrame

        Returns:
            MACD因子值
        """
        if ta is None:
            self.logger.error("❌ 无法计算 MACD，因为 'pandas-ta' 未加载。")
            raise ImportError("pandas-ta 未安装")

        if not self.validate_input(df, ['close']):
            return pd.Series([0] * len(df), index=df.index)

        fast_ema = self.params.get('fast_ema', 12)
        slow_ema = self.params.get('slow_ema', 26)
        signal_ema = self.params.get('signal_ema', 9)

        macd = ta.macd(
            df['close'],
            fast=fast_ema,
            slow=slow_ema,
            signal=signal_ema
        )

        if macd is None or macd.empty:
            self.logger.warning(
                f"⚠️ [MACD] (Config={fast_ema}_{slow_ema}_{signal_ema}) 计算返回 None"
            )
            return pd.Series([0] * len(df), index=df.index)

        macd_line_col = f'MACD_{fast_ema}_{slow_ema}_{signal_ema}'

        if macd_line_col not in macd.columns:
            self.logger.error(
                f"❌ [MACD] 无法在 {macd.columns.tolist()} 中找到 {macd_line_col}。"
            )
            return pd.Series([0] * len(df), index=df.index)

        macd_line = macd[macd_line_col]
        safe_close = df['close'].replace(0, float('nan'))

        return macd_line / safe_close * 100
