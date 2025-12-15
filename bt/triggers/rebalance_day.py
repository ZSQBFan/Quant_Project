# bt/triggers/rebalance_day.py

"""
调仓日触发器

职责：
1. 检查当前日期是否为调仓日
2. 如果是，调用 Pipeline 获取目标仓位
3. 提交买卖指令（不检查停牌，由策略基类统一处理）
"""

import pandas as pd
from .base import TriggerBase
from bt.core.constants import ActionPriority, ActionType


class RebalanceDayTrigger(TriggerBase):
    """
    调仓日触发器
    
    参数:
        strategy: 策略实例
        trading_days_list: 调仓日期列表（字符串格式，如 ['2024-01-02', '2024-01-15']）
    """
    
    def __init__(self, strategy, trading_days_list: list):
        super().__init__(strategy)
        self.rebalance_dates = set(pd.to_datetime(trading_days_list).date)
    
    def check_and_execute(self):
        """检查是否为调仓日，执行调仓逻辑"""
        current_date = self.s.datetime.date(0)
        
        if current_date not in self.rebalance_dates:
            return
        
        self.s.log("=" * 40)
        self.s.log(f"📅 触发调仓日: {current_date}")
        
        # 调用选股器
        all_datas = list(self.s.datas)
        selected_stocks = self.s.selector.select(all_datas)
        self.s.log(f"  选中 {len(selected_stocks)} 只股票")
        
        # 调用权重分配器
        weights = self.s.allocator.allocate(selected_stocks)
        
        # 获取可用资金
        total_value = self.s.broker.getvalue()
        target_cash = self.s.capital_manager.get_allocation(total_value)
        self.s.log(f"  目标投资金额: {target_cash:,.2f}")
        
        # 生成卖出指令：清理不在名单中的持仓（不检查停牌）
        selected_set = set(selected_stocks)
        for data in self._iter_held_stocks():
            if data not in selected_set:
                self.s.submit_action(
                    data=data,
                    action=ActionType.CLOSE,
                    reason='调仓清理: 不在新持仓名单',
                    priority=ActionPriority.REBALANCE
                )
        
        # 生成买入指令（不检查停牌）
        for data, weight in weights.items():
            target_value = target_cash * weight
            price = data.close[0]
            
            if price > 0:
                size = int(target_value / price / 100) * 100
                if size > 0:
                    self.s.submit_action(
                        data=data,
                        action=ActionType.BUY,
                        size=size,
                        reason=f'调仓买入 weight={weight:.2%}',
                        priority=ActionPriority.REBALANCE
                    )
        
        self.s.log("=" * 40)