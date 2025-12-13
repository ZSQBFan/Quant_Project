# core/analysis_metrics.py (已重构)

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict
import logging  # <- 【【【新增】】】


# 这是一个内部帮助函数
def _calculate_spearman_for_group(group: pd.DataFrame,
                                  return_col: str) -> float:
    """计算单个分组的斯皮尔曼秩相关系数。"""
    if len(group['factor_value']) < 2 or len(group[return_col]) < 2:
        return float('nan')
    
    # 【修复】检查数据是否有效（非空且非常量）
    factor_values = group['factor_value'].dropna()
    return_values = group[return_col].dropna()
    
    if len(factor_values) < 2 or len(return_values) < 2:
        return float('nan')
    
    # 检查是否为常量（所有值相同）
    if factor_values.nunique() <= 1 or return_values.nunique() <= 1:
        # logging.debug(f"  > 跳过常量数据: factor_nunique={factor_values.nunique()}, return_nunique={return_values.nunique()}")
        return float('nan')

    corr, _ = spearmanr(factor_values, return_values)
    return float(corr) if pd.notna(corr) else float('nan')


def calculate_rank_ic_series(factor_data: pd.DataFrame,
                             period: int) -> pd.Series:
    """
    计算每日的 Rank IC (Spearman 秩相关系数) 序列。
    """
    return_col = f'forward_return_{period}d'
    
    # 【调试】检查输入数据结构
    logging.debug(f"🔍 [IC计算] 输入数据形状: {factor_data.shape}")
    logging.debug(f"🔍 [IC计算] 输入数据索引: {factor_data.index.names}")
    logging.debug(f"🔍 [IC计算] 输入数据列: {list(factor_data.columns)}")
    logging.debug(f"🔍 [IC计算] 目标收益率列: {return_col}")
    logging.debug(f"🔍 [IC计算] 因子值列: 'factor_value'")
    
    # 检查数据是否存在
    if factor_data.empty:
        logging.warning("  > ⚠️ [IC计算] 输入数据为空！")
        return pd.Series(dtype=float, name=f'rank_ic_{period}d')
    
    # 检查必需列是否存在
    required_cols = ['factor_value', return_col]
    missing_cols = [col for col in required_cols if col not in factor_data.columns]
    if missing_cols:
        logging.error(f"  > ❌ [IC计算] 缺少必需列: {missing_cols}")
        logging.error(f"  > ❌ [IC计算] 可用列: {list(factor_data.columns)}")
        return pd.Series(dtype=float, name=f'rank_ic_{period}d')

    if 'date' not in factor_data.index.names:
        logging.warning("  > ⚠️ [IC计算] 期望 'date' 在索引中。")
        if 'date' in factor_data.columns:
            factor_data = factor_data.set_index('date', append=True)
            logging.info(f"  > ✅ [IC计算] 已将 'date' 设置为索引")
        else:
            logging.error("  > ❌ [IC计算] 无法找到 'date' 列或索引")
            return pd.Series(dtype=float, name=f'rank_ic_{period}d')

    # 检查数据样例
    logging.debug(f"🔍 [IC计算] 数据样例:\n{factor_data.head()}")
    logging.debug(f"🔍 [IC计算] 数据统计 - 因子值: 均值={factor_data['factor_value'].mean():.4f}, 非空数={factor_data['factor_value'].count()}")
    logging.debug(f"🔍 [IC计算] 数据统计 - 收益率: 均值={factor_data[return_col].mean():.4f}, 非空数={factor_data[return_col].count()}")

    # 【调试】逐个日期检查
    daily_groups = factor_data.groupby(level='date')
    logging.debug(f"🔍 [IC计算] 总共 {len(daily_groups)} 个交易日期")
    
    valid_days = 0
    invalid_days = 0
    
    def debug_calculate_spearman(group, return_col):
        nonlocal valid_days, invalid_days
        date = group.name if hasattr(group, 'name') else 'unknown'
        
        # 基本检查
        if len(group) < 2:
            logging.debug(f"  > [日期 {date}] 跳过: 数据点不足 ({len(group)} < 2)")
            invalid_days += 1
            return float('nan')
        
        # 检查非空数据
        factor_values = group['factor_value'].dropna()
        return_values = group[return_col].dropna()
        
        if len(factor_values) < 2 or len(return_values) < 2:
            logging.debug(f"  > [日期 {date}] 跳过: 非空数据不足 (因子:{len(factor_values)}, 收益:{len(return_values)})")
            invalid_days += 1
            return float('nan')
        
        # 检查唯一值
        if factor_values.nunique() <= 1:
            logging.debug(f"  > [日期 {date}] 跳过: 因子值全相同 ({factor_values.iloc[0] if len(factor_values) > 0 else 'N/A'})")
            invalid_days += 1
            return float('nan')
        
        if return_values.nunique() <= 1:
            logging.debug(f"  > [日期 {date}] 跳过: 收益率全相同 ({return_values.iloc[0] if len(return_values) > 0 else 'N/A'})")
            invalid_days += 1
            return float('nan')
        
        # 计算相关系数
        try:
            corr, _ = spearmanr(factor_values, return_values)
            valid_days += 1
            if valid_days <= 5:  # 只记录前5个valid的例子
                logging.debug(f"  > [日期 {date}] ✅ 有效: IC={corr:.4f}, 样本数={len(factor_values)}")
            return float(corr) if pd.notna(corr) else float('nan')
        except Exception as e:
            logging.debug(f"  > [日期 {date}] 计算出错: {e}")
            invalid_days += 1
            return float('nan')
    
    ic_by_date = daily_groups.apply(debug_calculate_spearman, return_col=return_col)
    
    logging.debug(f"🔍 [IC计算] 有效日期数: {valid_days}, 无效日期数: {invalid_days}")

    ic_by_date = ic_by_date.dropna()
    logging.debug(f"🔍 [IC计算] 最终IC序列长度: {len(ic_by_date)}")
    
    if len(ic_by_date) == 0:
        logging.warning("  > ⚠️ [IC计算] 所有日期的IC计算都失败了！")
        # 打印第一个日期的详细数据用于调试
        first_date = factor_data.index.get_level_values('date')[0]
        first_day_data = factor_data.loc[first_date]
        logging.debug(f"  > ❌ [第一个日期 {first_date}] 数据详情:")
        logging.debug(f"     因子值统计: {first_day_data['factor_value'].describe()}")
        logging.debug(f"     收益率统计: {first_day_data[return_col].describe()}")
        logging.debug(f"     因子值唯一值数: {first_day_data['factor_value'].nunique()}")
        logging.debug(f"     收益率唯一值数: {first_day_data[return_col].nunique()}")
    
    ic_by_date.name = f'rank_ic_{period}d'
    return ic_by_date


def analyze_ic_statistics(ic_series: pd.Series) -> Dict[str, float]:
    """
    根据IC时间序列计算常用的统计指标。
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
    计算因子分层收益。（已增强对无法分层情况的处理）
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
        # 【【【修改】】】
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

    # 【【【重要修正】】】
    # 原始的多空收益计算 (rolling(period).mean() / period) 是有问题的。
    # 正确的、无重叠（non-overlapping）的组合收益应该如下：

    # 1. 假设我们在 `date` 这一天，根据因子值构建了多空组合。
    # 2. 我们持有了 `period` 天。
    # 3. 收益 `daily_ls_returns` 是这 `period` 天的总收益（或平均收益，取决于 `return_col`）。

    # `factor_data` 包含 `forward_return_{p}d`
    # `get_ls_return` 计算的是 `(Q_N_ret - Q_1_ret)`，其中 `ret` 是 `p` 日收益
    # 所以 `daily_ls_returns` 已经是 P 日的组合收益。

    # 假设 daily_ls_returns 是每日权重更新、持有 p 天的收益。
    # 要将其转换为 *日收益率*，我们只需将其错开 p 天即可。
    # (更简单的重叠方法是：`daily_ls_returns.shift(1).rolling(window=period).mean() / period`)

    # 为保持简单，我们假设 `daily_ls_returns` 是 p-day 收益的日均值
    # (这与 Alphalens 的标准做法一致)

    # 我们假设 `get_ls_return` 得到的已经是 P 日的平均 *日* 收益（如果 `return_col` 是 `1d` 的话）
    # 但 `return_col` 是 `forward_return_{p}d`
    # 正确的方式应该是使用 1d 收益来计算组合，但这里为了保持与 `period` 一致：

    # 我们假设 daily_ls_returns 是在当天建仓、持有P天的 *总收益*
    # 要将其转换为 *日均收益*，我们除以 P

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
