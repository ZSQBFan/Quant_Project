"""
因子分析模块

提供因子计算、IC分析、分层收益计算和报告生成功能。
"""

from .metrics import (
    calculate_rank_ic_series,
    analyze_ic_statistics,
    calculate_quantile_returns,
    calculate_factor_portfolio_returns
)
from .calculator import FactorCalculator
from .report import FactorReport

__all__ = [
    # 分析指标
    'calculate_rank_ic_series',
    'analyze_ic_statistics',
    'calculate_quantile_returns',
    'calculate_factor_portfolio_returns',

    # 因子计算器
    'FactorCalculator',

    # 报告生成
    'FactorReport',
]
