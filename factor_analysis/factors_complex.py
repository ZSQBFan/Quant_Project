# factor_analysis/factors_complex.py (已按功能块重构)
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import functools  # (用于 Reversal20D 辅助函数)
"""

【【【复合因子（Type 2）计算中心】】】

复合因子使用全量宽表数据进行向量化计算，
并且通常包含行业中性化步骤。
每个因子的计算函数接受一个 DataFrame 参数:

- all_data_df: pd.DataFrame
    索引为 (date, asset) 的 MultiIndex DataFrame，
    包含计算该因子所需的所有列数据。

返回值:
- pd.Series
    索引为 (date, asset) 的 MultiIndex Series，
    包含计算得到的因子值。

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


def calculate_industry_neutral_adxdmi(all_data_df: pd.DataFrame) -> pd.Series:
    """
    计算【行业中性 ADX/DMI 因子】
    
    【【重构】】: 不再使用 'groupby().apply()'，
    改为在 (date, asset) 索引上直接计算，
    并使用 'groupby(level='asset')' 来处理 'pandas-ta' 的 MultiIndex 兼容性。
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

    if ta is None:
        logging.error("❌ 无法计算 ADXDMI，因为 'pandas-ta' 未加载。")
        raise ImportError("pandas-ta 未安装")

    # (确保数据按 asset, date 排序，这对于 groupby().ta.adx 至关重要)
    # (我们的 all_data_df 已经是 (date, asset) 排序，
    #  所以 swaplevel() 之后就是 (asset, date) 排序)
    data_sorted_by_asset = all_data_df.swaplevel().sort_index()

    # (使用 'pandas-ta' 的 MultiIndex 功能)
    # 'groupby(level='asset')' 告诉 ta 在每个 'asset' 组内独立计算
    try:
        dmi = ta.adx(
            data_sorted_by_asset['high'],
            data_sorted_by_asset['low'],
            data_sorted_by_asset['close'],
            length=period,
            groupby=data_sorted_by_asset.index.get_level_values('asset'))
    except Exception as e:
        logging.error(f"  > ❌ [IndNeu_ADXDMI] 'pandas-ta' 库在计算 ADX 时发生错误: {e}",
                      exc_info=True)
        return None

    if dmi is None or dmi.empty:
        logging.warning(
            f"  > ⚠️ [IndNeu_ADXDMI] 'pandas-ta' 库返回了 None/Empty (可能数据不足)。")
        return None

    # (将索引转回 (date, asset))
    dmi = dmi.swaplevel().sort_index()

    # (提取列，逻辑同 factors.py)
    try:
        adx_col = [col for col in dmi.columns if 'ADX' in col][0]
        plus_di_col = [col for col in dmi.columns if 'DMP' in col][0]
        minus_di_col = [col for col in dmi.columns if 'DMN' in col][0]
    except IndexError:
        logging.error(
            f"❌ [IndNeu_ADXDMI] 无法在 {dmi.columns.tolist()} 中找到 ADX/DMP/DMN。")
        return None

    adx = dmi[adx_col]
    plus_di = dmi[plus_di_col]
    minus_di = dmi[minus_di_col]

    direction_strength = (plus_di - minus_di) / (plus_di + minus_di).replace(
        0, np.nan)
    trend_strength_weight = adx / 100.0

    # (我们必须使用 .values 来避免索引错位)
    base_adx = pd.Series(np.where(
        adx.values > trend_threshold,
        direction_strength.values * trend_strength_weight.values, 0.0),
                         index=all_data_df.index)  # (确保索引与 all_data_df 一致)

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
#                 --- 基本面复合因子实现 (Fundamental Factors) ---
# ==============================================================================


def calculate_industry_neutral_ep(all_data_df: pd.DataFrame) -> pd.Series:
    """
    计算行业中性的 EP (Earnings / Price) 因子。
    """
    factor_name = "IndNeu_EP"
    logging.info(f"    > ⚙️ 正在计算 (Type 2): {factor_name}...")

    # 1. 准备数据
    # 注意：all_data_df 索引是 (date, asset)
    close = all_data_df['close']
    share_capital = all_data_df['share_capital']
    net_profit = all_data_df['net_profit_parent']

    # 2. 计算总市值 (Market Cap)
    market_cap = close * share_capital

    # 3. 计算原始因子 (E/P)
    # 处理分母为0或NaN的情况
    raw_ep = net_profit / market_cap.replace(0, np.nan)

    # 4. 极值处理 (简单的去极值，防止数据错误导致的 Inf)
    # 这里简单处理：将无限值设为NaN
    raw_ep = raw_ep.replace([np.inf, -np.inf], np.nan)

    # 5. 行业中性化
    return _neutralize_by_industry(raw_ep, all_data_df['industry'],
                                   factor_name)


def calculate_industry_neutral_bp(all_data_df: pd.DataFrame) -> pd.Series:
    """
    计算行业中性的 BP (Book / Price) 因子。
    """
    factor_name = "IndNeu_BP"
    logging.info(f"    > ⚙️ 正在计算 (Type 2): {factor_name}...")

    close = all_data_df['close']
    share_capital = all_data_df['share_capital']
    equity = all_data_df['total_equity_parent']

    market_cap = close * share_capital

    # 计算 B/P
    raw_bp = equity / market_cap.replace(0, np.nan)
    raw_bp = raw_bp.replace([np.inf, -np.inf], np.nan)

    return _neutralize_by_industry(raw_bp, all_data_df['industry'],
                                   factor_name)


def calculate_industry_neutral_roe(all_data_df: pd.DataFrame) -> pd.Series:
    """
    计算行业中性的 ROE (Return on Equity) 因子。
    """
    factor_name = "IndNeu_ROE"
    logging.info(f"    > ⚙️ 正在计算 (Type 2): {factor_name}...")

    net_profit = all_data_df['net_profit_parent']
    equity = all_data_df['total_equity_parent']

    # 计算 ROE
    raw_roe = net_profit / equity.replace(0, np.nan)
    raw_roe = raw_roe.replace([np.inf, -np.inf], np.nan)

    return _neutralize_by_industry(raw_roe, all_data_df['industry'],
                                   factor_name)


def calculate_industry_neutral_sales_growth(
        all_data_df: pd.DataFrame) -> pd.Series:
    """
    计算行业中性的营收增长率 (Sales Growth YoY)。
    逻辑: (当前总营收 - 去年同期总营收) / 去年同期总营收
    注意: 使用 252 个交易日作为一年的近似滞后期。
    """
    factor_name = "IndNeu_SalesGrowth"
    # 确保列名与数据库/数据源一致，这里假设为 'total_revenue'
    cols_needed = ['industry', 'total_revenue']

    if not all(col in all_data_df.columns for col in cols_needed):
        logging.error(f"❌ [{factor_name}] 无法计算：缺少必要列 {cols_needed}。")
        return None

    logging.info(f"    > ⚙️ 正在计算 (Type 2): {factor_name}...")

    # 1. 计算基础增长率 (YoY)
    # 使用 groupby(level='asset') 确保只在同一只股票的时间序列内计算
    logging.info(f"      > (1/2) 正在计算基础 {factor_name} (YoY, lag=252)...")

    # pct_change(252) 计算 (Today - Today-252) / Today-252
    # 添加 fill_method=None 以避免 FutureWarning
    raw_growth = all_data_df.groupby(
        level='asset')['total_revenue'].pct_change(periods=252, fill_method=None)

    # 2. 极值处理
    # 营收增长率可能因基数极小出现极端值，将其视为无效
    raw_growth = raw_growth.replace([np.inf, -np.inf], np.nan)

    # 3. 行业中性化
    return _neutralize_by_industry(raw_growth, all_data_df['industry'],
                                   factor_name)


def calculate_industry_neutral_cfop(all_data_df: pd.DataFrame) -> pd.Series:
    """
    计算行业中性的经营现金流市价率 (CFO / Market Cap)。
    """
    factor_name = "IndNeu_CFOP"
    logging.info(f"    > ⚙️ 正在计算 (Type 2): {factor_name}...")

    cfo = all_data_df['net_cash_flow_ops']
    market_cap = all_data_df['close'] * all_data_df['share_capital']

    # 计算 CFO/P
    raw_cfop = cfo / market_cap.replace(0, np.nan)
    raw_cfop = raw_cfop.replace([np.inf, -np.inf], np.nan)

    return _neutralize_by_industry(raw_cfop, all_data_df['industry'],
                                   factor_name)


def calculate_industry_neutral_gpm(all_data_df: pd.DataFrame) -> pd.Series:
    """
    计算行业中性的毛利率 (Gross Profit Margin)。
    公式: (Revenue - COGS) / Revenue
    """
    factor_name = "IndNeu_GPM"
    logging.info(f"    > ⚙️ 正在计算 (Type 2): {factor_name}...")

    revenue = all_data_df['total_revenue']
    cogs = all_data_df['cost_of_goods_sold']

    gross_profit = revenue - cogs
    raw_gpm = gross_profit / revenue.replace(0, np.nan)

    # 毛利率通常在 0~1 之间，但也可能为负，极值处理
    raw_gpm = raw_gpm.replace([np.inf, -np.inf], np.nan)

    # 简单的异常值过滤 (例如毛利率 > 100% 或 < -100% 通常是数据错误或极端情况)
    # 这里暂时不做强行截断，交给标准化处理

    return _neutralize_by_industry(raw_gpm, all_data_df['industry'],
                                   factor_name)


def calculate_industry_neutral_asset_turnover(
        all_data_df: pd.DataFrame) -> pd.Series:
    """
    计算行业中性的总资产周转率 (Asset Turnover)。
    公式: Revenue / Total Assets
    """
    factor_name = "IndNeu_AssetTurnover"
    logging.info(f"    > ⚙️ 正在计算 (Type 2): {factor_name}...")

    revenue = all_data_df['total_revenue']
    assets = all_data_df['total_assets']

    raw_ato = revenue / assets.replace(0, np.nan)
    raw_ato = raw_ato.replace([np.inf, -np.inf], np.nan)

    return _neutralize_by_industry(raw_ato, all_data_df['industry'],
                                   factor_name)


def calculate_industry_neutral_current_ratio(
        all_data_df: pd.DataFrame) -> pd.Series:
    """
    计算行业中性的流动比率 (Current Ratio)。
    公式: Current Assets / Current Liabilities
    """
    factor_name = "IndNeu_CurrentRatio"
    logging.info(f"    > ⚙️ 正在计算 (Type 2): {factor_name}...")

    ca = all_data_df['current_assets']
    cl = all_data_df['current_liabilities']

    raw_cr = ca / cl.replace(0, np.nan)
    raw_cr = raw_cr.replace([np.inf, -np.inf], np.nan)

    return _neutralize_by_industry(raw_cr, all_data_df['industry'],
                                   factor_name)


# ==============================================================================
#                 --- 【【【复合因子注册表】】】 ---
# ==============================================================================

COMPLEX_FACTOR_REGISTRY = {
    # 键 (str) : 值 (计算函数)
    "IndNeu_Momentum": calculate_industry_neutral_momentum,
    "IndNeu_Reversal20D": calculate_industry_neutral_reversal20d,
    "IndNeu_ADXDMI": calculate_industry_neutral_adxdmi,
    "IndNeu_VolumeCV": calculate_industry_neutral_volume_cv,
    "IndNeu_EP": calculate_industry_neutral_ep,
    "IndNeu_BP": calculate_industry_neutral_bp,
    "IndNeu_ROE": calculate_industry_neutral_roe,
    "IndNeu_SalesGrowth": calculate_industry_neutral_sales_growth,
    "IndNeu_CFOP": calculate_industry_neutral_cfop,
    "IndNeu_GPM": calculate_industry_neutral_gpm,
    "IndNeu_AssetTurnover": calculate_industry_neutral_asset_turnover,
    "IndNeu_CurrentRatio": calculate_industry_neutral_current_ratio,
}
