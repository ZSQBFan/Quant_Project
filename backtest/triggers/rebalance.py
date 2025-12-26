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
from backtest.core.constants import ActionPriority, ActionType


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

        self.s.log("=" * 60)
        self.s.log(f"📅 触发器激活: RebalanceDayTrigger ({current_date})")
        self.s.log("=" * 60)

        # 调用选股器
        all_datas = list(self.s.datas)
        selected_stocks = self.s.selector.select(all_datas)
        self.s.log(f"  ✅ 选股器选中 {len(selected_stocks)} 只股票")

        # 如果没有选中任何股票，则跳过本次调仓
        if not selected_stocks:
            self.s.log("  ⚠️ 未选中任何股票，跳过本次调仓")
            self.s.log("=" * 60)
            return

        # 显示选中的股票列表（前10只）
        import logging
        stock_names = [d._name for d in selected_stocks[:10]]
        if len(selected_stocks) > 10:
            stock_names.append(f"...及其他{len(selected_stocks) - 10}只")
        logging.debug(f"     选中股票: {', '.join(stock_names)}")

        # 调用权重分配器
        weights = self.s.allocator.allocate(selected_stocks)
        self.s.log(f"  ✅ 权重分配器完成分配 (等权重: {1/len(selected_stocks):.2%})")

        # 获取可用资金
        total_value = self.s.broker.getvalue()
        target_cash = self.s.capital_manager.get_allocation(total_value)
        self.s.log(f"  💰 当前总市值: {total_value:,.2f} 元")
        self.s.log(f"  💰 目标投资金额: {target_cash:,.2f} 元 (95%)")

        # 生成卖出指令：清理不在名单中的持仓（不检查停牌）
        selected_set = set(selected_stocks)
        close_count = 0
        for data in self._iter_held_stocks():
            if data not in selected_set:
                self.s.submit_action(
                    data=data,
                    action=ActionType.CLOSE,
                    reason='调仓清理: 不在新持仓名单',
                    priority=ActionPriority.REBALANCE
                )
                close_count += 1

        if close_count > 0:
            self.s.log(f"  🔴 提交 {close_count} 个平仓指令")

        # 生成买入指令（不检查停牌）
        buy_count = 0
        total_buy_value = 0
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
                    buy_count += 1
                    total_buy_value += size * price

        if buy_count > 0:
            self.s.log(f"  🟢 提交 {buy_count} 个买入指令 (计划投入 {total_buy_value:,.2f} 元)")

        self.s.log("=" * 60)