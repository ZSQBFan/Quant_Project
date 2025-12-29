# bt/triggers/rebalance_day.py

"""
调仓日触发器

职责：
1. 检查当前日期是否为调仓日
2. 如果是，调用 Pipeline 获取目标仓位
3. 提交买卖指令（不检查停牌，由策略基类统一处理）
4. 顺延机制：当选股器返回空结果时，最多顺延3个交易日重新尝试
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
    
    顺延机制:
        - 当选股器返回空列表时，自动顺延到下一个交易日
        - 最多顺延3次（包括原始调仓日）
        - 顺延期间记录详细日志
    """
    
    # 最大顺延次数（包括原始调仓日，最多尝试4次：原日+顺延3日）
    MAX_DEFERRALS = 3
    
    def __init__(self, strategy, trading_days_list: list):
        super().__init__(strategy)
        self.rebalance_dates = set(pd.to_datetime(trading_days_list).date)
    
    def _get_deferral_count(self) -> int:
        """
        获取当前调仓周期的顺延次数
        
        顺延状态存储在策略实例上，格式：
        _rebalance_deferral_info: {
            'original_date': date,  # 原始调仓日
            'count': int           # 当前顺延次数
        }
        """
        deferral_info = getattr(self.s, '_rebalance_deferral_info', None)
        current_date = self.s.datetime.date(0)
        
        # 检查是否是同一个调仓周期
        if deferral_info and deferral_info.get('original_date') == current_date:
            return deferral_info.get('count', 0)
        return 0
    
    def _set_deferral_info(self, original_date, count: int):
        """设置顺延信息到策略实例"""
        self.s._rebalance_deferral_info = {
            'original_date': original_date,
            'count': count
        }
    
    def _clear_deferral_info(self):
        """清除顺延信息"""
        if hasattr(self.s, '_rebalance_deferral_info'):
            delattr(self.s, '_rebalance_deferral_info')
    
    def _has_pending_deferral(self) -> bool:
        """
        检查是否有待处理的顺延调仓

        只要有顺延信息且顺延次数未用完，就返回 True
        """
        deferral_info = getattr(self.s, '_rebalance_deferral_info', None)
        if deferral_info is None:
            return False

        count = deferral_info.get('count', 0)

        # 只要顺延次数未用完，就返回 True
        return count < self.MAX_DEFERRALS
    
    def check_and_execute(self):
        """
        检查是否为调仓日，执行调仓逻辑

        顺延逻辑：
        1. 如果当前日期是调仓日，检查选股器返回值
        2. 如果选中0只股票且还能顺延，记录顺延并返回（等待下一交易日）
        3. 如果已顺延3次仍无结果，执行空调仓（清理所有持仓或跳过）
        """
        current_date = self.s.datetime.date(0)

        # 检查是否有待处理的顺延
        deferral_info = getattr(self.s, '_rebalance_deferral_info', None)
        has_pending_deferral = (deferral_info is not None and
                               deferral_info.get('count', 0) < self.MAX_DEFERRALS)

        # 判断是否是新的调仓日
        is_scheduled_rebalance_day = current_date in self.rebalance_dates

        # 如果是新的调仓日，清除旧的顺延状态（开始新的调仓周期）
        if is_scheduled_rebalance_day and deferral_info is not None:
            self._clear_deferral_info()
            deferral_info = None
            has_pending_deferral = False

        # 判断是否应该执行调仓（新的调仓日 或 有待处理的顺延）
        should_process = is_scheduled_rebalance_day or has_pending_deferral

        if not should_process:
            return

        # 获取顺延信息
        if deferral_info:
            original_date = deferral_info.get('original_date', current_date)
            deferral_count = deferral_info.get('count', 0)
        else:
            original_date = current_date
            deferral_count = 0
        
        # 打印分隔线和状态信息
        self.s.log("=" * 60)
        if deferral_count > 0:
            self.s.log(f"📅 触发器激活: RebalanceDayTrigger (顺延第{deferral_count}次, 原始日期: {original_date})")
        else:
            self.s.log(f"📅 触发器激活: RebalanceDayTrigger ({current_date})")
        self.s.log("=" * 60)

        # 调用选股器
        all_datas = list(self.s.datas)
        selected_stocks = self.s.selector.select(all_datas)
        self.s.log(f"  ✅ 选股器选中 {len(selected_stocks)} 只股票")

        # 如果没有选中任何股票，处理顺延逻辑
        if not selected_stocks:
            if deferral_count < self.MAX_DEFERRALS:
                # 可以继续顺延
                new_count = deferral_count + 1
                self._set_deferral_info(original_date, new_count)
                
                remaining = self.MAX_DEFERRALS - new_count
                self.s.log("  ⚠️ 未选中任何股票，触发顺延机制")
                self.s.log(f"     原始调仓日: {original_date}")
                self.s.log(f"     当前尝试次数: {new_count}/{self.MAX_DEFERRALS}")
                self.s.log(f"     剩余可顺延次数: {remaining}")
                self.s.log("     将在下一交易日再次尝试选股")
                self.s.log("=" * 60)
                return
            else:
                # 已达到最大顺延次数，仍然无结果
                self.s.log(f"  ⚠️ 已顺延 {deferral_count} 次，仍未选中任何股票")
                self.s.log(f"     原始调仓日: {original_date}")
                self.s.log("     执行空调仓处理（清理所有持仓）")
                
                # 清除顺延状态
                self._clear_deferral_info()
                
                # 生成清仓指令
                close_count = 0
                for data in self._iter_held_stocks():
                    self.s.submit_action(
                        data=data,
                        action=ActionType.CLOSE,
                        reason='调仓清理: 选股器连续返回空结果，执行清仓',
                        priority=ActionPriority.REBALANCE
                    )
                    close_count += 1
                
                if close_count > 0:
                    self.s.log(f"  🔴 提交 {close_count} 个平仓指令（因选股失败）")
                else:
                    self.s.log("  ℹ️ 无持仓需要清理")
                
                self.s.log("=" * 60)
                return

        # 成功选中股票，清除顺延状态
        if deferral_count > 0:
            self.s.log(f"  ✅ 顺延第 {deferral_count} 次成功选中股票，清除顺延状态")
        self._clear_deferral_info()

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
        # 传入数据对象以支持 Kelly 资金管理器计算
        target_cash = self.s.capital_manager.get_allocation(total_value, all_datas)
        self.s.log(f"  💰 当前总市值: {total_value:,.2f} 元")
        self.s.log(f"  💰 目标投资金额: {target_cash:,.2f} 元")

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