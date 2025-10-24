# factors.py

import pandas as pd
import numpy as np

# 因子计算严重依赖 pandas_ta 库，它提供了大量技术指标的向量化实现。
# 在运行前，请确保已安装: pip install pandas-ta
try:
    import pandas_ta as ta
except ImportError:
    print("错误: `pandas-ta` 库未安装。请运行 'pip install pandas-ta' 进行安装。")
    ta = None
"""
本文件定义了所有用于因子分析的计算逻辑。

核心设计思想：
1.  与 Backtrader 完全解耦：所有函数都基于 Pandas DataFrame 进行操作。
2.  向量化计算：利用 Pandas 和 a 的能力，对整个数据序列进行一次性计算，性能远高于逐个元素的循环。
3.  纯函数：每个函数接收一个DataFrame和一些参数，返回一个包含因子值的 Pandas Series，没有副作用。
4.  因子注册表 (FACTOR_REGISTRY)：提供一个集中的地方来注册和管理所有因子函数，便于扩展和调用。
"""

# --- 均值回归因子 ---


def calculate_rsi_factor(df: pd.DataFrame, rsi_period: int = 14) -> pd.Series:
    """
    计算 RSI 因子。
    因子值 = 50 - RSI。当 RSI 越低（超卖），因子值越高，符合“低买高卖”。
    """
    if ta is None: raise ImportError("pandas-ta 未安装")
    rsi_series = ta.rsi(df['close'], length=rsi_period)
    # Handle cases where RSI calculation fails (returns None)
    if rsi_series is None:
        # Return series of zeros with same index as input
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
    if ta is None: raise ImportError("pandas-ta 未安装")
    kdj = ta.stoch(high=df['high'],
                   low=df['low'],
                   close=df['close'],
                   k=period,
                   d=period_dfast,
                   smooth_k=period_dslow)
    # pandas-ta的列名是 STOCHk_9_3_3 和 STOCHd_9_3_3
    k_col = f'STOCHk_{period}_{period_dfast}_{period_dslow}'
    d_col = f'STOCHd_{period}_{period_dfast}_{period_dslow}'
    return kdj[k_col] - kdj[d_col]


def calculate_bollinger_bands_factor(df: pd.DataFrame,
                                     period: int = 20,
                                     devfactor: float = 2.0) -> pd.Series:
    """
    计算布林带因子（已更新为动态列名查找，更稳健）。
    因子值表示价格偏离中轨的归一化程度，符号取反以符合“低买高卖”。
    因子值为正表示价格在中轨之下，值越大表明离下轨越近（买入信号）。
    """
    if ta is None: raise ImportError("pandas-ta 未安装")
    bbands = ta.bbands(df['close'], length=period, std=devfactor)

    # 健壮性检查：如果数据不足无法计算，则返回一个填充为0的Series
    if bbands is None or bbands.empty:
        return pd.Series(0, index=df.index)

    # 【【核心修复】】不再硬编码列名，而是通过关键字动态查找
    # 这样做可以适应不同 pandas-ta 版本生成的不同列名格式
    try:
        # 查找包含 'BBM' (布林中轨) 的列名
        mid_band_col = [col for col in bbands.columns if 'BBM' in col][0]
        # 查找包含 'BBU' (布林上轨) 的列名
        top_band_col = [col for col in bbands.columns if 'BBU' in col][0]
    except IndexError:
        # 如果没有找到对应的列，则打印错误并返回，防止程序崩溃
        print(f"错误：无法在 {bbands.columns.tolist()} 中找到包含 'BBM' 或 'BBU' 的列")
        return pd.Series(0, index=df.index)

    mid_band = bbands[mid_band_col]
    top_band = bbands[top_band_col]

    band_width = top_band - mid_band
    # 防止除零错误
    safe_band_width = band_width.replace(0, np.nan)

    return -(df['close'] - mid_band) / safe_band_width


# --- 趋势跟踪因子 ---


def calculate_ma_cross_factor(df: pd.DataFrame,
                              short_ma: int = 14,
                              long_ma: int = 30) -> pd.Series:
    """
    计算均线交叉因子。
    因子值 = (短期均线 - 长期均线) / 长期均线 * 100。
    金叉时，因子值为正。
    """
    short_ma_series = df['close'].rolling(window=short_ma).mean()
    long_ma_series = df['close'].rolling(window=long_ma).mean()
    # 防止除零错误
    safe_long_ma = long_ma_series.replace(0, np.nan)
    return (short_ma_series - long_ma_series) / safe_long_ma * 100


def calculate_macd_factor(df: pd.DataFrame,
                          fast_ema: int = 12,
                          slow_ema: int = 26,
                          signal_ema: int = 9) -> pd.Series:
    """
    计算 MACD 因子。
    因子值 = MACD线的值 / 收盘价 * 100 (价格归一化)。
    """
    if ta is None: raise ImportError("pandas-ta 未安装")
    macd = ta.macd(df['close'],
                   fast=fast_ema,
                   slow=slow_ema,
                   signal=signal_ema)
    # Handle cases where MACD calculation fails (returns None)
    if macd is None:
        # Return series of zeros with same index as input
        return pd.Series([0] * len(df), index=df.index)

    # pandas-ta 的列名是 MACD_12_26_9
    macd_line = macd[f'MACD_{fast_ema}_{slow_ema}_{signal_ema}']
    safe_close = df['close'].replace(0, np.nan)
    return macd_line / safe_close * 100


def calculate_adx_dmi_factor(df: pd.DataFrame,
                             period: int = 14,
                             trend_threshold: int = 20) -> pd.Series:
    """
    计算 ADX/DMI 趋势因子。
    结合了趋势强度(ADX)和方向(DMI)，当ADX低于阈值时认为无趋势（因子值为0）。
    """
    if ta is None: raise ImportError("pandas-ta 未安装")
    dmi = ta.adx(df['high'], df['low'], df['close'], length=period)

    # 【【核心修复】】 增加健壮性检查
    # 如果数据不足无法计算指标，ta.adx 会返回 None，我们需要处理这种情况
    if dmi is None or dmi.empty:
        return pd.Series(0, index=df.index)  # 返回一个全为0的Series，避免程序崩溃

    # --- 如果检查通过，才继续执行后续代码 ---

    # 注意：与布林带问题类似，我们最好也使用动态查找以防万一
    try:
        adx_col = [col for col in dmi.columns if 'ADX' in col][0]
        plus_di_col = [col for col in dmi.columns if 'DMP' in col][0]
        minus_di_col = [col for col in dmi.columns if 'DMN' in col][0]
    except IndexError:
        print(f"错误：无法在 {dmi.columns.tolist()} 中找到 ADX/DMP/DMN 相关的列")
        return pd.Series(0, index=df.index)

    adx = dmi[adx_col]
    plus_di = dmi[plus_di_col]
    minus_di = dmi[minus_di_col]

    direction_strength = (plus_di - minus_di) / (plus_di + minus_di).replace(
        0, np.nan)
    trend_strength_weight = adx / 100.0

    # 核心逻辑：只有当ADX大于阈值时，信号才有效
    factor_series = pd.Series(np.where(
        adx > trend_threshold, direction_strength * trend_strength_weight,
        0.0),
                              index=df.index)

    return factor_series


def calculate_momentum_factor(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    计算动量因子 (Rate of Change)。
    因子值 = 价格在过去N期内的变化百分比。
    """
    return df['close'].pct_change(periods=period) * 100


# --- 其他因子 ---


def calculate_volume_spike_factor(df: pd.DataFrame,
                                  period: int = 20) -> pd.Series:
    """
    计算成交量突增因子。
    因子值 = (当前成交量 - 均线成交量) / 均线成交量。
    """
    ma_vol = df['volume'].rolling(window=period).mean()
    safe_ma_vol = ma_vol.replace(0, np.nan)
    return (df['volume'] - ma_vol) / safe_ma_vol


def calculate_reversal_20d_factor(df: pd.DataFrame,
                                  period: int = 40,
                                  decay: float = 20.0) -> pd.Series:
    """
    计算20日反转因子 (移植自你之前的 revs 逻辑)。
    
    该因子计算了过去 `period` (40) 个交易日的时间加权平均收益，并取其相反数。
    权重随时间呈指数衰减，最近的交易日权重最高。

    - 因子值为正: 表示股票近期表现较差 (下跌)，具备潜在的反弹可能 (买入信号)。
    - 因子值为负: 表示股票近期表现较好 (上涨)，具备潜在的回调可能 (卖出信号)。
    """
    # 1. 计算日收益率
    daily_returns = df['close'].pct_change()

    # 2. 定义时间衰减权重。
    # np.arange(period, 0, -1) 生成 [40, 39, ..., 1]
    # 权重数组中，最后一个元素对应最近的收益率，权重最大。
    weights = 2.0**-(np.arange(period, 0, -1) / decay)

    # 3. 定义用于滚动的加权函数
    def weighted_average(window):
        # ==================================================================
        # --- 核心修正点 ---
        # 传入的 window 长度在序列开始时可能小于 period。
        # 因此，我们必须只使用 weights 数组的最后 len(window) 个元素
        # 来匹配当前窗口的大小。
        current_weights = weights[-len(window):]
        # ==================================================================

        # 使用 np.nansum 可以安全地处理窗口内的 NaN 值
        return np.nansum(window * current_weights)

    # 4. 使用 rolling().apply() 计算加权的滚动平均值
    # raw=True 使得传递给 apply 函数的是 NumPy 数组，以提高性能
    weighted_returns = daily_returns.rolling(
        window=period,
        min_periods=int(period / 2)  # 至少需要一半的数据才开始计算
    ).apply(weighted_average, raw=True)

    # 5. 乘以 -1，得到最终的反转因子
    reversal_factor = -1.0 * weighted_returns

    return reversal_factor


# ==============================================================================
# 因子注册表 (Factor Registry)
# ==============================================================================
# 将所有因子计算函数注册到这里。
# key: 因子名称 (str)
# value: 对应的计算函数 (function)
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
