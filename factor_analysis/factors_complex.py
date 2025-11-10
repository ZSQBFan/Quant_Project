# factor_analysis/factors_complex.py (已更新)
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
"""
【【【复合因子（Type 2）计算中心 (重构版)】】】

【【重构日志】】:
- 2025-11-10 (新增因子):
  - 增加了 'calculate_industry_neutral_reversal20d'。
  - 增加了 'calculate_industry_neutral_adxdmi'。
  - 增加了必要的辅助函数 (_weighted_average_helper, _calculate_base_adxdmi)
    以保持与 factors.py 的解耦。
"""


# ------------------------------------------------------------------------------
# 警告：为了实现【完全独立】，本文件会【重新实现】
# 一些 `factors.py` 中的基础计算 (例如 动量)。
# ------------------------------------------------------------------------------
def _calculate_base_momentum(stock_df, period=20):
    """一个内部的、简化的动量计算"""
    return stock_df['close'].pct_change(periods=period) * 100


# ------------------------------------------------------------------------------
# 【【【新增】】】: Reversal20D 的辅助函数
# (逻辑复制自 factors.py)
# ------------------------------------------------------------------------------
def _weighted_average_helper(window_returns, weights):
    """
    一个可以被 .apply(raw=True) 调用的加权平均辅助函数。
    注意：weights 必须通过闭包 (closure) 或 functools.partial 传入。
    """
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


# ------------------------------------------------------------------------------
# 【【【新增】】】: ADXDMI 的辅助函数
# (逻辑复制自 factors.py)
# ------------------------------------------------------------------------------
def _calculate_base_adxdmi(df, period=14, trend_threshold=20):
    """
    在单个 DataFrame 上计算基础 ADXDMI 因子值。
    """
    if ta is None:
        logging.error("❌ 无法计算 ADXDMI，因为 'pandas-ta' 未加载。")
        raise ImportError("pandas-ta 未安装")

    # 确保索引是 DatetimeIndex (当从 groupby().apply() 调用时)
    if not isinstance(df.index, pd.DatetimeIndex):
        logging.error("❌ [ADXDMI 辅助函数] 接收到的 DataFrame 索引不是 DatetimeIndex。")
        df = df.set_index('date').sort_index()

    dmi = ta.adx(df['high'], df['low'], df['close'], length=period)

    if dmi is None or dmi.empty:
        logging.warning(
            f"  > ⚠️ [ADXDMI 辅助函数] (Period={period}) 计算返回 None (可能数据不足)。")
        return pd.Series(0, index=df.index)

    try:
        adx_col = [col for col in dmi.columns if 'ADX' in col][0]
        plus_di_col = [col for col in dmi.columns if 'DMP' in col][0]
        minus_di_col = [col for col in dmi.columns if 'DMN' in col][0]
    except IndexError:
        logging.error(
            f"❌ [ADXDMI 辅助函数] 无法在 {dmi.columns.tolist()} 中找到 ADX/DMP/DMN。")
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


# ==============================================================================
#                      --- 复合因子实现区 ---
# ==============================================================================


def calculate_industry_neutral_momentum(
        all_data_df: pd.DataFrame) -> pd.Series:
    """
    计算【行业中性动量因子】- 简化版本
    """
    if 'industry' not in all_data_df.columns or 'close' not in all_data_df.columns:
        logging.error("❌ [IndNeu_Momentum] 无法计算：缺少 'industry' 或 'close' 列。")
        return None

    logging.info("    > ⚙️ 正在计算 (Type 2): IndNeu_Momentum...")

    # 1. 计算基础动量
    logging.info("      > (1/2) 正在计算基础动量...")
    all_data_df = all_data_df.copy()
    all_data_df['temp_momentum'] = all_data_df.groupby(
        level='asset')['close'].pct_change(20) * 100

    # 2. 行业中性化
    logging.info("      > (2/2) 正在执行行业中性化...")
    valid_data = all_data_df[['temp_momentum', 'industry']].dropna()
    if valid_data.empty:
        logging.warning("      > ⚠️ [IndNeu_Momentum] 没有有效的（动量+行业）数据进行中性化。")
        return None

    industry_means = valid_data.groupby(['date',
                                         'industry'])['temp_momentum'].mean()

    valid_data = valid_data.reset_index()
    industry_means = industry_means.reset_index(name='industry_mean')
    merged_data = pd.merge(valid_data,
                           industry_means,
                           on=['date', 'industry'],
                           how='left')
    merged_data['IndNeu_Momentum'] = merged_data[
        'temp_momentum'] - merged_data['industry_mean']

    result_series = merged_data.set_index(['date', 'asset'])['IndNeu_Momentum']
    result_series.name = "IndNeu_Momentum"
    return result_series.sort_index()


# ------------------------------------------------------------------------------
# 【【【【【【 新因子 1: 行业中性 20 日反转 】】】】】】
# ------------------------------------------------------------------------------
def calculate_industry_neutral_reversal20d(
        all_data_df: pd.DataFrame) -> pd.Series:
    """
    计算【行业中性 20 日反转因子】
    使用 'Reversal20D' 的参数 (period=40, decay=20)
    """
    if 'industry' not in all_data_df.columns or 'close' not in all_data_df.columns:
        logging.error("❌ [IndNeu_Reversal20D] 无法计算：缺少 'industry' 或 'close' 列。")
        return None

    logging.info("    > ⚙️ 正在计算 (Type 2): IndNeu_Reversal20D...")

    # --- 1. 计算基础 20 日反转因子 (逻辑同 factors.py) ---
    logging.info("      > (1/2) 正在计算基础 Reversal20D (p=40, d=20)...")
    period = 40
    decay = 20

    daily_returns = all_data_df.groupby(level='asset')['close'].pct_change()
    daily_returns.replace(0.0, np.nan, inplace=True)

    weights = 2.0**-(np.arange(period, 0, -1) / decay)

    # (使用 functools.partial 是一种更清晰的传递 weights 的方式，
    #  但为了保持与 factor.py 辅助函数一致，我们使用 lambda 闭包)
    weighted_avg_func = lambda x: _weighted_average_helper(x, weights=weights)

    # (这是一个计算密集型步骤)
    weighted_returns = daily_returns.groupby(level='asset').rolling(
        window=period,
        min_periods=int(period / 2)).apply(weighted_avg_func,
                                           raw=True)  # raw=True 提速

    # 结果是 MultiIndex (asset, date)，将其转回 (date, asset)
    base_reversal = -1.0 * weighted_returns.reset_index(
        level=0, drop=True).sort_index()

    all_data_df = all_data_df.copy()
    all_data_df['temp_reversal'] = base_reversal

    # --- 2. 行业中性化 (逻辑同 momentum) ---
    logging.info("      > (2/2) 正在执行行业中性化...")
    valid_data = all_data_df[['temp_reversal', 'industry']].dropna()
    if valid_data.empty:
        logging.warning("      > ⚠️ [IndNeu_Reversal20D] 没有有效的（反转+行业）数据进行中性化。")
        return None

    industry_means = valid_data.groupby(['date',
                                         'industry'])['temp_reversal'].mean()

    valid_data = valid_data.reset_index()
    industry_means = industry_means.reset_index(name='industry_mean')
    merged_data = pd.merge(valid_data,
                           industry_means,
                           on=['date', 'industry'],
                           how='left')

    merged_data['IndNeu_Reversal20D'] = merged_data[
        'temp_reversal'] - merged_data['industry_mean']

    result_series = merged_data.set_index(['date',
                                           'asset'])['IndNeu_Reversal20D']
    result_series.name = "IndNeu_Reversal20D"
    return result_series.sort_index()


# ------------------------------------------------------------------------------
# 【【【【【【 新因子 2: 行业中性 ADX/DMI 】】】】】】
# ------------------------------------------------------------------------------
def calculate_industry_neutral_adxdmi(all_data_df: pd.DataFrame) -> pd.Series:
    """
    计算【行业中性 ADX/DMI 因子】
    使用 (period=14, trend_threshold=20)
    """
    cols_needed = ['industry', 'high', 'low', 'close']
    if not all(col in all_data_df.columns for col in cols_needed):
        logging.error(f"❌ [IndNeu_ADXDMI] 无法计算：缺少必要列 {cols_needed}。")
        return None

    logging.info("    > ⚙️ 正在计算 (Type 2): IndNeu_ADXDMI...")

    # --- 1. 计算基础 ADXDMI 因子 (逻辑同 factors.py) ---
    logging.info("      > (1/2) 正在计算基础 ADXDMI (p=14, t=20)...")
    period = 14
    trend_threshold = 20

    # (这是一个计算密集型步骤：在 5000+ 股票上 apply)
    base_adx = all_data_df.groupby(
        level='asset').apply(lambda df: _calculate_base_adxdmi(
            df.droplevel('asset'),  # (droplevel 'asset' 剩下 'date' 索引)
            period=period,
            trend_threshold=trend_threshold))
    # (groupby().apply() 可能会返回一个带有多余索引的 Series，
    #  我们需要将其清理为 (date, asset) MultiIndex)
    if isinstance(base_adx.index, pd.MultiIndex):
        # (如果 apply 返回 (asset, date) 索引, 交换它)
        if 'asset' in base_adx.index.names:
            base_adx = base_adx.swaplevel().sort_index()
        else:
            # (处理其他可能的 apply 结果)
            base_adx = base_adx.reset_index(level=0, drop=True).sort_index()

    all_data_df = all_data_df.copy()
    all_data_df['temp_adx'] = base_adx

    # --- 2. 行业中性化 (逻辑同 momentum) ---
    logging.info("      > (2/2) 正在执行行业中性化...")
    valid_data = all_data_df[['temp_adx', 'industry']].dropna()
    if valid_data.empty:
        logging.warning("      > ⚠️ [IndNeu_ADXDMI] 没有有效的（ADX+行业）数据进行中性化。")
        return None

    industry_means = valid_data.groupby(['date',
                                         'industry'])['temp_adx'].mean()

    valid_data = valid_data.reset_index()
    industry_means = industry_means.reset_index(name='industry_mean')
    merged_data = pd.merge(valid_data,
                           industry_means,
                           on=['date', 'industry'],
                           how='left')

    merged_data['IndNeu_ADXDMI'] = merged_data['temp_adx'] - merged_data[
        'industry_mean']

    result_series = merged_data.set_index(['date', 'asset'])['IndNeu_ADXDMI']
    result_series.name = "IndNeu_ADXDMI"
    return result_series.sort_index()


# ==============================================================================
#                 --- 【【【复合因子注册表】】】 ---
# ==============================================================================

COMPLEX_FACTOR_REGISTRY = {
    # 键 (str) : 值 (计算函数)
    "IndNeu_Momentum": calculate_industry_neutral_momentum,

    # 【【【新增】】】
    "IndNeu_Reversal20D": calculate_industry_neutral_reversal20d,
    "IndNeu_ADXDMI": calculate_industry_neutral_adxdmi,
}
