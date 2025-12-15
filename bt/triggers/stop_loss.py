# bt/triggers/stop_loss.py

"""
止损触发器 - MVP 最简实现

设计：
- 遍历所有持仓，计算盈亏比例
- 如果亏损超过阈值，提交平仓意图
- 不检查停牌（由策略基类统一处理）
"""

from .base import TriggerBase
from bt.core.constants import ActionPriority, ActionType


class StopLossTrigger(TriggerBase):
    """
    固定百分比止损触发器
    
    参数:
        strategy: 策略实例
        loss_threshold: 止损阈值，默认 -0.10（即 -10%），必须是负数
    """
    
    def __init__(self, strategy, loss_threshold: float = -0.10):
        super().__init__(strategy)
        self.loss_threshold = loss_threshold
    
    def check_and_execute(self):
        """检查所有持仓，触发止损（不检查停牌）"""
        for data in self._iter_held_stocks():
            pos = self.s.getposition(data)
            
            # 防止除零
            if pos.price <= 0:
                continue
            
            # 计算盈亏比例
            current_price = data.close[0]
            pct_change = (current_price - pos.price) / pos.price
            
            # 触发止损 - 直接提交，不管停牌
            if pct_change < self.loss_threshold:
                self.s.submit_action(
                    data=data,
                    action=ActionType.CLOSE,
                    reason=f'止损触发: {pct_change:.2%} < {self.loss_threshold:.2%}',
                    priority=ActionPriority.STOP_LOSS
                )