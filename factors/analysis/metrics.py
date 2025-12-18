"""
因子分析指标计算模块

提供 IC 计算、分层收益计算等核心分析指标。
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict
import logging


_spearman_warning_shown = False  # 全局标志，只显示一次警告

def _calculate_spearman_for_group(group: pd.DataFrame,
                                  return_col: str) -> float:
    """计算单个分组的斯皮尔曼秩相关系数。"""
    global _spearman_warning_shown

    # 检查列是否存在
    if 'factor_value' not in group.columns:
        if not _spearman_warning_shown:
            logging.warning(f"  > ⚠️ [_calculate_spearman_for_group] 组中没有 'factor_value' 列，列: {group.columns.tolist()}")
            _spearman_warning_shown = True
        return float('nan')
    if return_col not in group.columns:
        if not _spearman_warning_shown:
            logging.warning(f"  > ⚠️ [_calculate_spearman_for_group] 组中没有 '{return_col}' 列，列: {group.columns.tolist()}")
            _spearman_warning_shown = True
        return float('nan')

    if len(group['factor_value']) < 2 or len(group[return_col]) < 2:
        if not _spearman_warning_shown:
            logging.warning(f"  > ⚠️ [_calculate_spearman_for_group] 数据点不足: factor_value={len(group['factor_value'])}, {return_col}={len(group[return_col])}")
            _spearman_warning_shown = True
        return float('nan')

    # 检查输入是否为常量，避免ConstantInputWarning
    if group['factor_value'].nunique() <= 1 or group[return_col].nunique() <= 1:
        if not _spearman_warning_shown:
            logging.warning(f"  > ⚠️ [_calculate_spearman_for_group] 数据为常量: factor_value唯一值={group['factor_value'].nunique()}, {return_col}唯一值={group[return_col].nunique()}")
            _spearman_warning_shown = True
        return float('nan')

    corr, _ = spearmanr(group['factor_value'], group[return_col])
    return float(corr) if pd.notna(corr) else float('nan')


def calculate_rank_ic_series(factor_data: pd.DataFrame,
                             period: int) -> pd.Series:
    """
    计算每日的 Rank IC (Spearman 秩相关系数) 序列。

    Args:
        factor_data: 包含 'factor_value' 和 'forward_return_{period}d' 列的 DataFrame，
                     以 'date' 为索引层级
        period: 远期收益计算周期

    Returns:
        每日 IC 值的 Series
    """
    return_col = f'forward_return_{period}d'

    # 检查收益率列是否存在
    if return_col not in factor_data.columns:
        logging.warning(f"  > ⚠️ [calculate_rank_ic_series] 列 '{return_col}' 不存在于数据中。可用列: {factor_data.columns.tolist()}")
        return pd.Series(dtype=float, name=f'rank_ic_{period}d')

    # 检查 factor_value 列是否存在
    if 'factor_value' not in factor_data.columns:
        logging.warning(f"  > ⚠️ [calculate_rank_ic_series] 列 'factor_value' 不存在于数据中。可用列: {factor_data.columns.tolist()}")
        return pd.Series(dtype=float, name=f'rank_ic_{period}d')

    if 'date' not in factor_data.index.names:
        logging.warning("  > ⚠️ [calculate_rank_ic_series] 期望 'date' 在索引中。")
        if 'date' in factor_data.columns:
            factor_data = factor_data.set_index('date', append=True)
        else:
            pass  # 假设索引第0层是日期

    # 检查每日数据点数量
    group_sizes = factor_data.groupby(level='date').size()
    num_groups_with_1_or_less = (group_sizes <= 1).sum()
    if num_groups_with_1_or_less > 0:
        logging.warning(f"  > ⚠️ [calculate_rank_ic_series] 有 {num_groups_with_1_or_less}/{len(group_sizes)} 个日期只有 <=1 条记录，无法计算 IC")

    # 重置警告标志
    global _spearman_warning_shown
    _spearman_warning_shown = False

    ic_by_date: pd.Series = factor_data.groupby(level='date').apply(
        _calculate_spearman_for_group, return_col=return_col)

    # 报告 NaN 数量
    nan_count = ic_by_date.isna().sum()
    total_count = len(ic_by_date)
    if nan_count == total_count and total_count > 0:
        logging.warning(f"  > ⚠️ [calculate_rank_ic_series] 所有 {total_count} 个日期的 IC 都是 NaN！")

    ic_by_date = ic_by_date.dropna()
    ic_by_date.name = f'rank_ic_{period}d'
    return ic_by_date


def analyze_ic_statistics(ic_series: pd.Series) -> Dict[str, float]:
    """
    根据IC时间序列计算常用的统计指标。

    Args:
        ic_series: IC 时间序列

    Returns:
        包含 IC 均值、标准差、IR 等统计指标的字典
    """
    if ic_series.empty:
        logging.warning("  > ⚠️ [analyze_ic_statistics] 传入的 IC 序列为空。")
        return {}

    mean = ic_series.mean()
    std = ic_series.std()

    safe_std = std if std > 0 else 1e-6
    safe_len = len(ic_series) if len(ic_series) > 0 else 1

    stats: Dict[str, float] = {
        'ic_mean': mean,
        'ic_std': std,
        'ir': mean / safe_std,
        'ic_gt_0_prob': (ic_series > 0).sum() / safe_len,
        'ic_abs_gt_0.02_prob': (ic_series.abs() > 0.02).sum() / safe_len,
        'ic_t_stat': (mean / safe_std) * np.sqrt(safe_len)
    }
    return stats


def calculate_quantile_returns(factor_data: pd.DataFrame,
                               period: int,
                               quantiles: int = 5) -> pd.Series:
    """
    计算因子分层收益。

    Args:
        factor_data: 因子数据
        period: 远期收益计算周期
        quantiles: 分层数量

    Returns:
        各分层的平均收益 Series
    """
    return_col = f'forward_return_{period}d'

    def get_quantile_return(df: pd.DataFrame) -> pd.Series:
        try:
            df['quantile'] = pd.qcut(df['factor_value'],
                                     quantiles,
                                     labels=False,
                                     duplicates='drop')
        except ValueError as e:
            # 当数据点太少时，qcut 可能会失败
            logging.debug(f"  > 🐞 [qcut 失败] (周期 {period}d): {e}。可能当日数据点过少。")
            df['quantile'] = 0  # 将所有归为一层

        return df.groupby('quantile')[return_col].mean()

    # 对每个截面日应用分层函数
    grouped_returns = factor_data.groupby(
        level='date').apply(get_quantile_return)

    if isinstance(grouped_returns, pd.DataFrame):
        quantile_returns = grouped_returns.mean(axis=0)
        quantile_returns.index = [
            f'Q{int(i)+1}' for i in quantile_returns.index
        ]
    else:
        logging.warning(
            f"  > ⚠️  (周期 {period}d): 因子值无法有效分层为 {quantiles} 组 (可能因子值差异过小)。"
            f" 仅返回整体平均收益。")
        mean_ret = grouped_returns.mean()
        quantile_returns = pd.Series({'Q1': mean_ret})

    if len(quantile_returns) > 1:
        # (确保 Q1 和 QN 存在)
        if f'Q{quantiles}' in quantile_returns.index and 'Q1' in quantile_returns.index:
            ls_return = quantile_returns[f'Q{quantiles}'] - quantile_returns[
                'Q1']
        else:
            # 如果分层不完整，使用最后一层和第一层
            ls_return = quantile_returns.iloc[-1] - quantile_returns.iloc[0]

        long_short_series = pd.Series({'Long-Short': ls_return})
        quantile_returns = pd.concat([quantile_returns, long_short_series])
    elif len(quantile_returns) > 0:
        # 至少有一个分位数，但不足以计算多空
        long_short_series = pd.Series({'Long-Short': 0.0})
        quantile_returns = pd.concat([quantile_returns, long_short_series])

    quantile_returns.name = f'mean_return_{period}d'
    return quantile_returns


def calculate_factor_portfolio_returns(factor_data: pd.DataFrame,
                                       period: int,
                                       quantiles: int = 5) -> pd.Series:
    """
    计算基于因子排序构建的多空组合的每日收益序列。

    Args:
        factor_data: 因子数据
        period: 远期收益计算周期
        quantiles: 分层数量

    Returns:
        每日多空组合收益 Series
    """
    return_col = f'forward_return_{period}d'

    def get_ls_return(df: pd.DataFrame) -> float:
        try:
            if df['factor_value'].nunique() < quantiles:
                return 0.0

            df['quantile'] = pd.qcut(df['factor_value'],
                                     quantiles,
                                     labels=False,
                                     duplicates='drop')

            if (quantiles -
                    1) in df['quantile'].values and 0 in df['quantile'].values:
                long_ret = df[df['quantile'] == quantiles -
                              1][return_col].mean()
                short_ret = df[df['quantile'] == 0][return_col].mean()
                return long_ret - short_ret
            return 0.0
        except Exception as e:
            logging.warning(f"  > ⚠️ [get_ls_return] 计算多空收益时出错: {e}")
            return 0.0

    daily_ls_returns: pd.Series = factor_data.groupby(
        level='date').apply(get_ls_return)

    # 找到 1d 的收益列
    return_1d_col = 'forward_return_1d'
    if 'forward_return_1d' not in factor_data.columns and 1 in factor_data.columns:
        return_1d_col = f'forward_return_1d'
    elif 'forward_return_1d' not in factor_data.columns:
        logging.warning(
            "  > ⚠️ [L/S Portfolio] 无法计算净值曲线，因为 'forward_return_1d' 未提供。")
        return pd.Series(
            0.0, index=factor_data.index.get_level_values('date').unique())

    def get_1d_ls_return(df: pd.DataFrame) -> float:
        if df['factor_value'].nunique() < quantiles:
            return 0.0

        df['quantile'] = pd.qcut(df['factor_value'],
                                 quantiles,
                                 labels=False,
                                 duplicates='drop')

        if (quantiles -
                1) in df['quantile'].values and 0 in df['quantile'].values:
            long_ret = df[df['quantile'] == quantiles -
                          1][return_1d_col].mean()
            short_ret = df[df['quantile'] == 0][return_1d_col].mean()
            return (long_ret - short_ret) / 2  # 因子值为中性，多空各一半仓位
        return 0.0

    # `daily_ls_returns_1d` 是 *第二天* 的收益（因为 1d return 是 T+1 的收益）
    daily_ls_returns_1d = factor_data.groupby(
        level='date').apply(get_1d_ls_return)

    # 将 T 日的因子信号，shift(1) 到 T+1 日才能产生收益
    return daily_ls_returns_1d.shift(1).fillna(0)
