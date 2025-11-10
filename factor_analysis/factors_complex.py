# factor_analysis/factors_complex.py (已按功能块重构)
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import functools  # (用于 Reversal20D 辅助函数)
"""
【【【复合因子（Type 2）计算中心 (重构版)】】】

【【重构日志】】:
- 2025-11-10 (新增因子):
  - 增加了 'calculate_industry_neutral_volume_cv' (成交量变异系数)
  - 增加了 'calculate_industry_neutral_reversal20d'
  - 增加了 'calculate_industry_neutral_adxdmi'
- 2025-11-10 (结构重构):
  - 将辅助函数 (helper) 与其对应的计算函数 (calculator) 放在一起。
  - 将通用的辅助函数 (_neutralize_by_industry) 移至顶部。
"""

# ------------------------------------------------------------------------------
# 警告：为了实现【完全独立】，本文件会【重新实现】
# 一些 `factors.py` 中的基础计算。
# ------------------------------------------------------------------------------

# ==============================================================================
#                      --- 通用辅助函数 (Common Helpers) ---
# ==============================================================================


def _neutralize_by_industry(base_factor_series: pd.Series,
                            industry_col: pd.Series,
                            factor_name: str) -> pd.Series:
    """
    一个通用的行业中性化辅助函数，用于简化代码。
    被所有 IndNeu_* 因子调用。
    """
    logging.info(f"      > (2/2) 正在为 {factor_name} 执行行业中性化...")

    # 1. 将基础因子和行业数据合并
    all_data_df = pd.concat([base_factor_series, industry_col], axis=1)
    all_data_df.columns = ['temp_factor', 'industry']  # 重命名以便通用

    # 2. 中性化
    valid_data = all_data_df.dropna()
    if valid_data.empty:
        logging.warning(f"      > ⚠️ [{factor_name}] 没有有效的（因子+行业）数据进行中性化。")
        return None

    industry_means = valid_data.groupby(['date',
                                         'industry'])['temp_factor'].mean()

    valid_data = valid_data.reset_index()
    industry_means = industry_means.reset_index(name='industry_mean')

    merged_data = pd.merge(valid_data,
                           industry_means,
                           on=['date', 'industry'],
                           how='left')

    # 3. 计算中性化因子 (因子值 - 行业均值)
    merged_data[factor_name] = merged_data['temp_factor'] - merged_data[
        'industry_mean']

    result_series = merged_data.set_index(['date', 'asset'])[factor_name]
    result_series.name = factor_name
    return result_series.sort_index()


# ==============================================================================
#                  --- 1. 行业中性动量 (IndNeu_Momentum) ---
# ==============================================================================


def _calculate_base_momentum(stock_df, period=20):
    """
    [IndNeu_Momentum 辅助函数]
    一个内部的、简化的动量计算。
    """
    return stock_df['close'].pct_change(periods=period) * 100


def calculate_industry_neutral_momentum(
        all_data_df: pd.DataFrame) -> pd.Series:
    """
    计算【行业中性动量因子】
    """
    factor_name = "IndNeu_Momentum"
    if 'industry' not in all_data_df.columns or 'close' not in all_data_df.columns:
        logging.error(f"❌ [{factor_name}] 无法计算：缺少 'industry' 或 'close' 列。")
        return None

    logging.info(f"    > ⚙️ 正在计算 (Type 2): {factor_name}...")

    # 1. 计算基础动量
    logging.info(f"      > (1/2) 正在计算基础 {factor_name} (20d pct_change)...")
    base_momentum = all_data_df.groupby(
        level='asset')['close'].pct_change(20) * 100

    # 2. 行业中性化 (使用通用辅助函数)
    return _neutralize_by_industry(base_momentum, all_data_df['industry'],
                                   factor_name)


# ==============================================================================
#                 --- 2. 行业中性反转 (IndNeu_Reversal20D) ---
# ==============================================================================


def _weighted_average_helper(window_returns, weights):
    """
    [IndNeu_Reversal20D 辅助函数]
    (复制自 factors.py)
    一个可以被 .apply(raw=True) 调用的加权平均辅助函数。
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


def calculate_industry_neutral_reversal20d(
        all_data_df: pd.DataFrame) -> pd.Series:
    """
    计算【行业中性 20 日反转因子】
    """
    factor_name = "IndNeu_Reversal20D"
    if 'industry' not in all_data_df.columns or 'close' not in all_data_df.columns:
        logging.error(f"❌ [{factor_name}] 无法计算：缺少 'industry' 或 'close' 列。")
        return None

    logging.info(f"    > ⚙️ 正在计算 (Type 2): {factor_name}...")

    # --- 1. 计算基础 20 日反转因子 (p=40, d=20) ---
    logging.info(f"      > (1/2) 正在计算基础 {factor_name} (p=40, d=20)...")
    period = 40
    decay = 20

    daily_returns = all_data_df.groupby(level='asset')['close'].pct_change()
    daily_returns.replace(0.0, np.nan, inplace=True)

    weights = 2.0**-(np.arange(period, 0, -1) / decay)

    # (使用 functools.partial 绑定 weights 参数，比 lambda 更清晰)
    weighted_avg_func = functools.partial(_weighted_average_helper,
                                          weights=weights)

    weighted_returns = daily_returns.groupby(level='asset').rolling(
        window=period, min_periods=int(period / 2)).apply(weighted_avg_func,
                                                          raw=True)

    base_reversal = -1.0 * weighted_returns.reset_index(
        level=0, drop=True).sort_index()

    # --- 2. 行业中性化 (使用通用辅助函数) ---
    return _neutralize_by_industry(base_reversal, all_data_df['industry'],
                                   factor_name)


# ==============================================================================
#                   --- 3. 行业中性 ADXDMI (IndNeu_ADXDMI) ---
# ==============================================================================


def _calculate_base_adxdmi(df, period=14, trend_threshold=20):
    """
    [IndNeu_ADXDMI 辅助函数]
    (复制自 factors.py)
    在单个 DataFrame 上计算基础 ADXDMI 因子值。
    """
    if ta is None:
        logging.error("❌ 无法计算 ADXDMI，因为 'pandas-ta' 未加载。")
        raise ImportError("pandas-ta 未安装")

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            # (当从 groupby().apply() 调用时, date 可能是一个列)
            if 'date' in df.columns:
                df = df.set_index('date').sort_index()
            else:
                raise ValueError("DataFrame 既没有 DatetimeIndex 也没有 'date' 列")
        except Exception as e:
            logging.error(
                f"❌ [ADXDMI 辅助函数] 接收到的 DataFrame 索引不是 DatetimeIndex: {e}")
            return pd.Series(0, index=df.index)

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


def calculate_industry_neutral_adxdmi(all_data_df: pd.DataFrame) -> pd.Series:
    """
    计算【行业中性 ADX/DMI 因子】
    """
    factor_name = "IndNeu_ADXDMI"
    cols_needed = ['industry', 'high', 'low', 'close']
    if not all(col in all_data_df.columns for col in cols_needed):
        logging.error(f"❌ [{factor_name}] 无法计算：缺少必要列 {cols_needed}。")
        return None

    logging.info(f"    > ⚙️ 正在计算 (Type 2): {factor_name}...")

    # --- 1. 计算基础 ADXDMI 因子 (p=14, t=20) ---
    logging.info(f"      > (1/2) 正在计算基础 {factor_name} (p=14, t=20)...")
    period = 14
    trend_threshold = 20

    base_adx = all_data_df.groupby(
        level='asset'
    ).apply(lambda df: _calculate_base_adxdmi(
        df.droplevel('asset'), period=period, trend_threshold=trend_threshold))
    if isinstance(base_adx.index, pd.MultiIndex):
        if 'asset' in base_adx.index.names:
            base_adx = base_adx.swaplevel().sort_index()
        else:
            base_adx = base_adx.reset_index(level=0, drop=True).sort_index()

    # --- 2. 行业中性化 (使用通用辅助函数) ---
    return _neutralize_by_industry(base_adx, all_data_df['industry'],
                                   factor_name)


# ==============================================================================
#                --- 4. 行业中性成交量CV (IndNeu_VolumeCV) ---
# ==============================================================================


def _calculate_base_volume_cv(df_group, period=20):
    """
    [IndNeu_VolumeCV 辅助函数]
    在单个股票分组上计算成交量变异系数 (CV = std / mean)。
    """
    rolling_std = df_group['volume'].rolling(window=period).std()
    rolling_mean = df_group['volume'].rolling(window=period).mean()

    # 防止除零
    safe_rolling_mean = rolling_mean.replace(0, np.nan)

    return rolling_std / safe_rolling_mean


def calculate_industry_neutral_volume_cv(
        all_data_df: pd.DataFrame) -> pd.Series:
    """
    计算【行业中性 成交量变异系数 (CV) 因子】
    使用 20 日滚动窗口 (CV = std / mean)
    """
    factor_name = "IndNeu_VolumeCV"
    if 'industry' not in all_data_df.columns or 'volume' not in all_data_df.columns:
        logging.error(f"❌ [{factor_name}] 无法计算：缺少 'industry' 或 'volume' 列。")
        return None

    logging.info(f"    > ⚙️ 正在计算 (Type 2): {factor_name}...")

    # --- 1. 计算基础成交量 CV (p=20) ---
    logging.info(f"      > (1/2) 正在计算基础 {factor_name} (p=20)...")
    period = 20

    # (这是一个计算密集型步骤)
    base_volume_cv = all_data_df.groupby(
        level='asset').apply(lambda df: _calculate_base_volume_cv(
            df.droplevel('asset'), period=period))

    # (清理 apply 带来的多余索引)
    if isinstance(base_volume_cv.index, pd.MultiIndex):
        if 'asset' in base_volume_cv.index.names:
            base_volume_cv = base_volume_cv.swaplevel().sort_index()
        else:
            base_volume_cv = base_volume_cv.reset_index(
                level=0, drop=True).sort_index()

    # --- 2. 行业中性化 (使用通用辅助函数) ---
    return _neutralize_by_industry(base_volume_cv, all_data_df['industry'],
                                   factor_name)


# ==============================================================================
#                 --- 【【【复合因子注册表】】】 ---
# ==============================================================================

COMPLEX_FACTOR_REGISTRY = {
    # 键 (str) : 值 (计算函数)
    "IndNeu_Momentum": calculate_industry_neutral_momentum,
    "IndNeu_Reversal20D": calculate_industry_neutral_reversal20d,
    "IndNeu_ADXDMI": calculate_industry_neutral_adxdmi,

    # 【【【新增】】】
    "IndNeu_VolumeCV": calculate_industry_neutral_volume_cv,
}
