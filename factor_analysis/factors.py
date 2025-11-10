# factors.py (已重构)

import pandas as pd
import numpy as np
import logging  # <- 【【【新增】】】

# 因子计算严重依赖 pandas_ta 库
try:
    import pandas_ta as ta
except ImportError:
    # 【【【修改】】】
    logging.critical(
        "⛔ 致命错误: `pandas-ta` 库未安装。请运行 'pip install pandas-ta' 进行安装。")
    logging.critical("   > 因子计算将无法进行。")
    ta = None
"""
本文件定义了所有用于因子分析的计算逻辑。

【【重构日志】】:
- 2025-11-09:
  - 引入 'logging' 模块，替换所有 'print' 和 'ImportError' 语句。
  - 增强了因子函数 (如 BBands, ADX) 内部的日志记录。
"""

# --- 均值回归因子 ---


def calculate_rsi_factor(df: pd.DataFrame, rsi_period: int = 14) -> pd.Series:
    """
    计算 RSI 因子。
    因子值 = 50 - RSI。当 RSI 越低（超卖），因子值越高，符合“低买高卖”。
    """
    if ta is None:
        logging.error("❌ 无法计算 RSI，因为 'pandas-ta' 未加载。")
        raise ImportError("pandas-ta 未安装")
    rsi_series = ta.rsi(df['close'], length=rsi_period)
    if rsi_series is None:
        logging.warning(
            f"  > ⚠️ [RSI] (Period={rsi_period}) 计算返回 None (可能数据不足)。")
        return pd.Series([0] * len(df), index=df.index)
    return 50 - rsi_series


def calculate_kdj_factor(df: pd.DataFrame,
                         period: int = 9,
                         period_dfast: int = 3,
                         period_dslow: int = 3) -> pd.Series:
    """
    计算 KDJ 因子。
    因子值 = K值 - D值。金叉时因子值上穿零轴。
    """
    if ta is None:
        logging.error("❌ 无法计算 KDJ，因为 'pandas-ta' 未加载。")
        raise ImportError("pandas-ta 未安装")
    kdj = ta.stoch(high=df['high'],
                   low=df['low'],
                   close=df['close'],
                   k=period,
                   d=period_dfast,
                   smooth_k=period_dslow)

    if kdj is None or kdj.empty:
        logging.warning(
            f"  > ⚠️ [KDJ] (Config={period}_{period_dfast}_{period_dslow}) 计算返回 None (可能数据不足)。"
        )
        return pd.Series([0] * len(df), index=df.index)

    k_col = f'STOCHk_{period}_{period_dfast}_{period_dslow}'
    d_col = f'STOCHd_{period}_{period_dfast}_{period_dslow}'

    if k_col not in kdj.columns or d_col not in kdj.columns:
        logging.error(
            f"❌ [KDJ] 无法在 {kdj.columns.tolist()} 中找到 {k_col} 或 {d_col}。")
        return pd.Series([0] * len(df), index=df.index)

    return kdj[k_col] - kdj[d_col]


def calculate_bollinger_bands_factor(df: pd.DataFrame,
                                     period: int = 20,
                                     devfactor: float = 2.0) -> pd.Series:
    """
    计算布林带因子。
    """
    if ta is None:
        logging.error("❌ 无法计算 BollingerBands，因为 'pandas-ta' 未加载。")
        raise ImportError("pandas-ta 未安装")
    bbands = ta.bbands(df['close'], length=period, std=devfactor)

    if bbands is None or bbands.empty:
        logging.warning(
            f"  > ⚠️ [BollingerBands] (Period={period}) 计算返回 None (可能数据不足)。")
        return pd.Series(0, index=df.index)

    try:
        mid_band_col = [col for col in bbands.columns if 'BBM' in col][0]
        top_band_col = [col for col in bbands.columns if 'BBU' in col][0]
    except IndexError:
        # 【【【修改】】】
        logging.error(
            f"❌ [BollingerBands] 无法在 {bbands.columns.tolist()} 中找到 'BBM' 或 'BBU'。"
        )
        return pd.Series(0, index=df.index)

    mid_band = bbands[mid_band_col]
    top_band = bbands[top_band_col]

    band_width = top_band - mid_band
    safe_band_width = band_width.replace(0, np.nan)

    return -(df['close'] - mid_band) / safe_band_width


# --- 趋势跟踪因子 ---


def calculate_ma_cross_factor(df: pd.DataFrame,
                              short_ma: int = 14,
                              long_ma: int = 30) -> pd.Series:
    """
    计算均线交叉因子。
    """
    short_ma_series = df['close'].rolling(window=short_ma).mean()
    long_ma_series = df['close'].rolling(window=long_ma).mean()
    safe_long_ma = long_ma_series.replace(0, np.nan)
    return (short_ma_series - long_ma_series) / safe_long_ma * 100


def calculate_macd_factor(df: pd.DataFrame,
                          fast_ema: int = 12,
                          slow_ema: int = 26,
                          signal_ema: int = 9) -> pd.Series:
    """
    计算 MACD 因子。
    """
    if ta is None:
        logging.error("❌ 无法计算 MACD，因为 'pandas-ta' 未加载。")
        raise ImportError("pandas-ta 未安装")
    macd = ta.macd(df['close'],
                   fast=fast_ema,
                   slow=slow_ema,
                   signal=signal_ema)

    if macd is None or macd.empty:
        logging.warning(
            f"  > ⚠️ [MACD] (Config={fast_ema}_{slow_ema}_{signal_ema}) 计算返回 None (可能数据不足)。"
        )
        return pd.Series([0] * len(df), index=df.index)

    macd_line_col = f'MACD_{fast_ema}_{slow_ema}_{signal_ema}'
    if macd_line_col not in macd.columns:
        logging.error(
            f"❌ [MACD] 无法在 {macd.columns.tolist()} 中找到 {macd_line_col}。")
        return pd.Series([0] * len(df), index=df.index)

    macd_line = macd[macd_line_col]
    safe_close = df['close'].replace(0, np.nan)
    return macd_line / safe_close * 100


def calculate_adx_dmi_factor(df: pd.DataFrame,
                             period: int = 14,
                             trend_threshold: int = 20) -> pd.Series:
    """
    计算 ADX/DMI 趋势因子。
    """
    if ta is None:
        logging.error("❌ 无法计算 ADXDMI，因为 'pandas-ta' 未加载。")
        raise ImportError("pandas-ta 未安装")
    dmi = ta.adx(df['high'], df['low'], df['close'], length=period)

    if dmi is None or dmi.empty:
        logging.warning(
            f"  > ⚠️ [ADXDMI] (Period={period}) 计算返回 None (可能数据不足)。")
        return pd.Series(0, index=df.index)

    try:
        adx_col = [col for col in dmi.columns if 'ADX' in col][0]
        plus_di_col = [col for col in dmi.columns if 'DMP' in col][0]
        minus_di_col = [col for col in dmi.columns if 'DMN' in col][0]
    except IndexError:
        # 【【【修改】】】
        logging.error(
            f"❌ [ADXDMI] 无法在 {dmi.columns.tolist()} 中找到 ADX/DMP/DMN。")
        return pd.Series(0, index=df.index)

    adx = dmi[adx_col]
    plus_di = dmi[plus_di_col]
    minus_di = dmi[minus_di_col]

    direction_strength = (plus_di - minus_di) / (plus_di + minus_di).replace(
        0, np.nan)
    trend_strength_weight = adx / 100.0

    factor_series = pd.Series(np.where(
        adx > trend_threshold, direction_strength * trend_strength_weight,
        0.0),
                              index=df.index)

    return factor_series


def calculate_momentum_factor(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    计算动量因子 (Rate of Change)。
    """
    return df['close'].pct_change(periods=period) * 100


# --- 其他因子 ---


def calculate_volume_spike_factor(df: pd.DataFrame,
                                  period: int = 20) -> pd.Series:
    """
    计算成交量突增因子。
    """
    ma_vol = df['volume'].rolling(window=period).mean()
    safe_ma_vol = ma_vol.replace(0, np.nan)
    return (df['volume'] - ma_vol) / safe_ma_vol


def calculate_reversal_20d_factor(df: pd.DataFrame,
                                  period: int = 40,
                                  decay: float = 20.0) -> pd.Series:
    """
    计算20日反转因子 (已修正)。
    """
    daily_returns = df['close'].pct_change()
    daily_returns.replace(0.0, np.nan, inplace=True)

    weights = 2.0**-(np.arange(period, 0, -1) / decay)

    def weighted_average(window_returns):
        current_weights = weights[-len(window_returns):]
        valid_mask = ~np.isnan(window_returns)
        valid_returns = window_returns[valid_mask]
        valid_weights = current_weights[valid_mask]

        if len(valid_returns) == 0:
            return np.nan

        sum_of_weights = np.sum(valid_weights)
        if sum_of_weights == 0:
            return np.nan

        weighted_sum = np.dot(valid_returns, valid_weights)
        return weighted_sum / sum_of_weights

    weighted_returns = daily_returns.rolling(
        window=period, min_periods=int(period / 2)).apply(weighted_average,
                                                          raw=True)

    reversal_factor = -1.0 * weighted_returns
    return reversal_factor


# ==============================================================================
# 因子注册表 (Factor Registry)
# ==============================================================================
FACTOR_REGISTRY = {
    'RSI': calculate_rsi_factor,
    'KDJ': calculate_kdj_factor,
    'BollingerBands': calculate_bollinger_bands_factor,
    'MovingAverageCross': calculate_ma_cross_factor,
    'MACD': calculate_macd_factor,
    'ADXDMI': calculate_adx_dmi_factor,
    'Momentum': calculate_momentum_factor,
    'VolumeSpike': calculate_volume_spike_factor,
    'Reversal20D': calculate_reversal_20d_factor,
}
