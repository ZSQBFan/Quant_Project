# bt/triggers/__init__.py

"""
Backtrader 触发器系统

设计原则：
- 触发器只负责感知和提交意图
- 不检查停牌状态（由策略基类统一处理）
"""

from .base import TriggerBase
from .stop_loss import StopLossTrigger
from .rebalance_day import RebalanceDayTrigger

__all__ = ['TriggerBase', 'StopLossTrigger', 'RebalanceDayTrigger']
