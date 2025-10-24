# analysis_metrics.py

import pandas as pd
from scipy.stats import spearmanr
from typing import Dict


# 这是一个内部帮助函数，用于替换之前在 apply 中使用的 lambda 函数
# 这样做可以让类型检查器更容易理解代码
def _calculate_spearman_for_group(group: pd.DataFrame,
                                  return_col: str) -> float:
    """计算单个分组的斯皮尔曼秩相关系数。"""
    # 如果分组内数据少于2个，无法计算相关性，返回NaN
    if len(group['factor_value']) < 2 or len(group[return_col]) < 2:
        return float('nan')

    # spearmanr 返回 (correlation, p-value)
    corr, _ = spearmanr(group['factor_value'], group[return_col])
    return float(corr) if pd.notna(corr) else float('nan')


def calculate_rank_ic_series(factor_data: pd.DataFrame,
                             period: int) -> pd.Series:
    """
    计算每日的 Rank IC (Spearman 秩相关系数) 序列。
    """
    return_col = f'forward_return_{period}d'

    ic_by_date: pd.Series = factor_data.groupby(level='date').apply(
        _calculate_spearman_for_group, return_col=return_col)

    ic_by_date = ic_by_date.dropna()
    ic_by_date.name = f'rank_ic_{period}d'
    return ic_by_date


def analyze_ic_statistics(ic_series: pd.Series) -> Dict[str, float]:
    """
    根据IC时间序列计算常用的统计指标。
    """
    if ic_series.empty:
        return {}

    mean = ic_series.mean()
    std = ic_series.std()

    stats: Dict[str, float] = {
        'IC Mean (均值)':
        mean,
        'IC Std. (标准差)':
        std,
        'Information Ratio (IR)':
        mean / std if std > 0 else 0.0,
        'IC > 0 Prob. (IC>0概率)':
        (ic_series > 0).sum() / len(ic_series) if len(ic_series) > 0 else 0.0,
        'IC abs > 0.02 Prob. (IC绝对值>0.02概率)': (ic_series.abs() > 0.02).sum() /
        len(ic_series) if len(ic_series) > 0 else 0.0,
    }
    return stats


def calculate_quantile_returns(factor_data: pd.DataFrame,
                               period: int,
                               quantiles: int = 5) -> pd.Series:
    """
    计算因子分层收益。（已增强对无法分层情况的处理）
    """
    return_col = f'forward_return_{period}d'

    def get_quantile_return(df: pd.DataFrame) -> pd.Series:
        df['quantile'] = pd.qcut(df['factor_value'],
                                 quantiles,
                                 labels=False,
                                 duplicates='drop')
        return df.groupby('quantile')[return_col].mean()

    # 对每个截面日应用分层函数
    grouped_returns = factor_data.groupby(
        level='date').apply(get_quantile_return)

    # --- 核心修复逻辑 ---
    # 检查 grouped_returns 的类型。正常情况下，它是一个DataFrame (index=date, columns=quantiles)
    # 如果无法分层，它可能会退化成一个 Series (index=date, value=mean_return)
    if isinstance(grouped_returns, pd.DataFrame):
        # 正常情况：数据可以被有效分层
        quantile_returns = grouped_returns.mean(axis=0)
        quantile_returns.index = [
            f'Q{int(i)+1}' for i in quantile_returns.index
        ]
    else:
        # 边界情况：数据无法有效分层，grouped_returns 是一个 Series
        print(
            f"  ⚠️ 警告 (周期 {period}d): 因子值无法有效分层为 {quantiles} 组 (可能因子值差异过小)。仅返回整体平均收益。"
        )
        mean_ret = grouped_returns.mean()
        # 构建一个最小化的 Series 以免下游代码崩溃
        quantile_returns = pd.Series({'Q1': mean_ret})

    quantile_returns.name = f'mean_return_{period}d'

    # 计算多空收益 (最高层 - 最低层)
    # 只有在能分出多于一层时，计算多空才有意义
    if len(quantile_returns) > 1:
        ls_return = quantile_returns.iloc[-1] - quantile_returns.iloc[0]
        long_short_series = pd.Series({'Long-Short': ls_return})
        quantile_returns = pd.concat([quantile_returns, long_short_series])

    return quantile_returns


def calculate_factor_portfolio_returns(factor_data: pd.DataFrame,
                                       period: int,
                                       quantiles: int = 5) -> pd.Series:
    """
    计算基于因子排序构建的多空组合的每日收益序列。
    """
    return_col = f'forward_return_{period}d'

    def get_ls_return(df: pd.DataFrame) -> float:
        # 检查是否有足够的唯一值来分层
        if df['factor_value'].nunique() < quantiles:
            return 0.0  # 如果不能有效分层，当日多空收益为0

        df['quantile'] = pd.qcut(df['factor_value'],
                                 quantiles,
                                 labels=False,
                                 duplicates='drop')

        if (quantiles -
                1) in df['quantile'].values and 0 in df['quantile'].values:
            long_ret = df[df['quantile'] == quantiles - 1][return_col].mean()
            short_ret = df[df['quantile'] == 0][return_col].mean()
            return long_ret - short_ret
        return 0.0

    daily_ls_returns: pd.Series = factor_data.groupby(
        level='date').apply(get_ls_return)

    resampled_returns = daily_ls_returns.rolling(window=period).mean() / period

    return resampled_returns.fillna(0)
