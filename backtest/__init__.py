"""
回测领域

提供 Backtrader 回测框架的封装和扩展。

模块结构：
- core/              核心策略组件
- pipeline/          回测流水线（选股、权重、资金）
- triggers/          触发器（调仓、止损）
- data/              数据导出
- reports/           报告生成
"""

from . import reports

__all__ = [
    'reports',
]
