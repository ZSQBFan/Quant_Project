# factor_analysis/factors_complex.py (已重构)
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging  # <- 【【【新增】】】
"""
【【【复合因子（Type 2）计算中心 (重构版)】】】

【【重构日志】】:
- 2025-11-09:
  - 引入 'logging' 模块，替换所有 'print' 语句。
"""


# ------------------------------------------------------------------------------
# 警告：为了实现【完全独立】，本文件会【重新实现】
# 一些 `factors.py` 中的基础计算 (例如 动量)。
# ------------------------------------------------------------------------------
def _calculate_base_momentum(stock_df, period=20):
    """一个内部的、简化的动量计算"""
    return stock_df['close'].pct_change(periods=period) * 100


# ==============================================================================
#                      --- 复合因子实现区 ---
# ==============================================================================


def calculate_industry_neutral_momentum(
        all_data_df: pd.DataFrame) -> pd.Series:
    """
    计算【行业中性动量因子】- 简化版本
    """
    if 'industry' not in all_data_df.columns or 'close' not in all_data_df.columns:
        # 【【【修改】】】
        logging.error("❌ [IndNeu_Momentum] 无法计算：缺少 'industry' 或 'close' 列。")
        return None

    # 【【【修改】】】
    logging.info("    > ⚙️ 正在计算 (Type 2): IndNeu_Momentum...")

    # 1. 计算基础动量
    # 【【【修改】】】
    logging.info("      > (1/2) 正在计算基础动量...")

    all_data_df = all_data_df.copy()
    all_data_df['temp_momentum'] = all_data_df.groupby(
        level='asset')['close'].pct_change(20) * 100

    # 2. 行业中性化
    # 【【【修改】】】
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


# 【【【示例：如何添加 E/P 因子】】】
def calculate_ep_factor(all_data_df: pd.DataFrame) -> pd.Series:
    """
    一个简单的 E/P 因子。
    (假设 E/P 越高越好)
    """
    if 'ep_ratio' not in all_data_df.columns:
        # 【【【修改】】】
        logging.error(
            "  > ❌ [EP_Factor] 无法计算: 'ep_ratio' 列在 all_data_df 中不存在。")
        logging.error(
            "    > 提示：您是否在 main_analyzer.py 中启用了 LOAD_FUNDAMENTAL_DATA？")
        return None

    # 【【【修改】】】
    logging.info("    > ⚙️ 正在计算 (Type 2): EP_Factor...")

    # 因子值就是 E/P 值本身 (假设已在 main_analyzer.py 中 ffill)
    result_series = all_data_df['ep_ratio']
    result_series.name = "EP_Factor"
    return result_series.sort_index()


# ==============================================================================
#                 --- 【【【复合因子注册表】】】 ---
# ==============================================================================

COMPLEX_FACTOR_REGISTRY = {
    # 键 (str) : 值 (计算函数)
    "IndNeu_Momentum": calculate_industry_neutral_momentum,
    # "EP_Factor": calculate_ep_factor, # (取消注释以启用)
}
