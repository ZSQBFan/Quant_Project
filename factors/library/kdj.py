"""
KDJ因子

随机指标 - 均值回归因子
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


@register_factor('KDJ', category='simple', description='KDJ指标')
class KDJFactor(BaseFactor):
    """
    KDJ因子

    计算公式：因子值 = K值 - D值
    金叉时因子值上穿零轴。
    """

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        计算KDJ因子

        Args:
            df: 包含 'high', 'low', 'close' 列的 DataFrame

        Returns:
            KDJ因子值
        """
        if ta is None:
            self.logger.error("❌ 无法计算 KDJ，因为 'pandas-ta' 未加载。")
            raise ImportError("pandas-ta 未安装")

        if not self.validate_input(df, ['high', 'low', 'close']):
            return pd.Series([0] * len(df), index=df.index)

        period = self.params.get('period', 9)
        period_dfast = self.params.get('period_dfast', 3)
        period_dslow = self.params.get('period_dslow', 3)

        kdj = ta.stoch(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            k=period,
            d=period_dfast,
            smooth_k=period_dslow
        )

        if kdj is None or kdj.empty:
            self.logger.warning(
                f"⚠️ [KDJ] (Config={period}_{period_dfast}_{period_dslow}) 计算返回 None"
            )
            return pd.Series([0] * len(df), index=df.index)

        k_col = f'STOCHk_{period}_{period_dfast}_{period_dslow}'
        d_col = f'STOCHd_{period}_{period_dfast}_{period_dslow}'

        if k_col not in kdj.columns or d_col not in kdj.columns:
            self.logger.error(
                f"❌ [KDJ] 无法在 {kdj.columns.tolist()} 中找到 {k_col} 或 {d_col}。"
            )
            return pd.Series([0] * len(df), index=df.index)

        return kdj[k_col] - kdj[d_col]
