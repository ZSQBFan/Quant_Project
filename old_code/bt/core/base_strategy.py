# bt/core/base_strategy.py

"""
Backtrader 策略基类

核心职责：
1. 管理触发器系统
2. 收集所有交易意图到缓冲区
3. 解决冲突（按优先级裁决）
4. 统一执行最终指令
"""

import backtrader as bt
import logging
import sys
from datetime import datetime
from .constants import ActionPriority, ActionType


class BacktestStrategy(bt.Strategy):
    """
    事件驱动回测策略基类
    
    设计模式：
    - 触发器（Trigger）：负责感知市场状态，提交交易意图
    - 缓冲区（Buffer）：收集所有意图，解决冲突
    - 执行器（Executor）：统一执行最终指令
    
    使用示例:
        cerebro.addstrategy(
            BacktestStrategy,
            selector=TopNSelector(top_n=10),
            allocator=EqualWeightAllocator(),
            capital_manager=FullPositionManager(),
            triggers=[
                lambda s: StopLossTrigger(s, loss_threshold=-0.10),
                lambda s: RebalanceDayTrigger(s, trading_days_list=dates)
            ]
        )
    """
    
    params = (
        ('selector', None),         # 选股器实例
        ('allocator', None),        # 权重分配器实例
        ('capital_manager', None),  # 资金管理器实例
        ('triggers', []),           # 触发器工厂函数列表
        ('pbar', None),             # 进度条实例
        ('total_steps', None),      # 预估时间步总数（用于原始行内进度显示）
    )
    
    def __init__(self):
        """初始化策略"""
        # 进度条支持 (必须最先初始化，因为 log 方法会用到)
        self.pbar = self.p.pbar
        self.total_steps = self.p.total_steps
        self.step_counter = 0
        
        # 保存组件引用
        self.selector = self.p.selector
        self.allocator = self.p.allocator
        self.capital_manager = self.p.capital_manager
        
        # 🔧 触发器实例化机制：
        # 注意：self.p.triggers接收的是函数列表，不是触发器实例列表
        # 每个函数应该接收strategy参数并返回触发器实例
        # 例如：lambda s: SimpleTestTrigger(s)
        self.triggers = []
        for trigger_factory in self.p.triggers:
            trigger = trigger_factory(self)  # ⚠️ 这里调用工厂函数创建触发器实例
            self.triggers.append(trigger)
            self.log(f"  已加载触发器: {trigger.__class__.__name__}")
        
        # 中央指令缓冲区
        self.pending_actions = []
        
        self.log("✅ 策略初始化完成")
    
    def submit_action(self, data, action: str, size: int = None, 
                      reason: str = '', priority: int = ActionPriority.OTHER):
        """
        提交交易意图（供触发器调用）
        
        注意：此方法不直接执行交易，而是将意图添加到缓冲区，
        等待 _execute_pending_actions() 统一处理。
        
        参数:
            data: Backtrader 数据对象
            action: 交易动作 ('buy', 'sell', 'close')
            size: 交易数量（股）
            reason: 原因描述（用于日志和调试）
            priority: 优先级（数值越小越优先）
        """
        self.pending_actions.append({
            'data': data,
            'action': action,
            'size': size,
            'reason': reason,
            'priority': priority
        })
    
    def _execute_pending_actions(self):
        """
        核心逻辑：解决冲突并执行交易
        
        处理步骤：
        1. 按优先级排序（数值越小越优先）
        2. 去重：每只股票只保留最高优先级的指令
        3. 执行最终指令
        4. 清空缓冲区
        """
        if not self.pending_actions:
            return
        
        # 1. 按优先级排序
        self.pending_actions.sort(key=lambda x: x['priority'])
        
        # 2. 去重：同一股票保留最高优先级
        final_intents = {}
        for intent in self.pending_actions:
            d = intent['data']
            if d not in final_intents:
                final_intents[d] = intent
            # 注意：由于已排序，第一个遇到的就是最高优先级，后续的会被忽略
        
        # 3. 执行最终指令
        for d, intent in final_intents.items():
            # 检查是否停牌
            if self._is_suspended(d):
                self.log(f"⏸️ 意图被拦截(停牌): {intent['action'].upper()} {d._name} | 原因: {intent['reason']}")
                continue
            
            act = intent['action']
            size = intent['size']
            reason = intent['reason']
            
            self.log(f"📈 执行: {act.upper()} {d._name} | 数量: {size} | 原因: {reason}")
            
            if act == ActionType.CLOSE:
                self.close(data=d)
            elif act == ActionType.BUY:
                if size and size > 0:
                    self.buy(data=d, size=size)
            elif act == ActionType.SELL:
                if size and size > 0:
                    self.sell(data=d, size=size)
        
        # 4. 清空缓冲区
        self.pending_actions.clear()
    
    def _is_suspended(self, data) -> bool:
        """
        检查股票是否停牌
        
        此方法在执行阶段统一调用，触发器不需要自行检查。
        """
        if hasattr(data, 'suspended') and len(data) > 0:
            return bool(data.suspended[0])
        return False
    
    def log(self, txt: str, level: int = logging.INFO):
        """
        日志辅助函数
        
        同时输出到控制台和日志文件（如果配置了的话）
        """
        try:
            # 尝试获取当前日期，如果还没有数据则使用系统时间
            dt = self.datetime.date(0)
        except (IndexError, AttributeError):
            dt = datetime.now().date()
        
        msg = f'[{dt}] {txt}'
        
        # 控制台输出由上层统一控制（避免污染控制台）；此处仅记录到 logging
        logging.log(level, msg)
    
    def next(self):
        """
        Backtrader 每个时间步调用一次
        
        执行顺序：
        1. 触发器感知阶段：遍历所有触发器，收集意图到缓冲区
        2. 执行阶段：裁决并执行缓冲区指令
        
        这种两阶段设计的好处：
        - 所有触发器平等运行，不会因为顺序影响结果
        - 冲突解决逻辑集中在一处，易于维护
        """
        # 🔄 触发器调用机制（核心要点）：
        # 这里遍历self.triggers列表，每个触发器都必须实现check_and_execute()方法
        # 如果触发器没有被正确调用，可能的原因：
        # 1. self.triggers列表为空（触发器实例化失败）
        # 2. 触发器对象没有check_and_execute()方法
        # 3. 触发器内部逻辑错误导致异常
        #
        # ⚠️ 调试方法：
        # - 检查日志中是否出现"已加载触发器: XXX"
        # - 在这里添加调试打印：print(f"触发器数量: {len(self.triggers)}")
        # - 在触发器check_and_execute()开头添加打印

        # 记录当前时间步 (改为 DEBUG 级别)
        current_date = self.datetime.date(0)
        logging.debug("时间步：" + str(current_date))

        # 进度条/行内输出均不写入控制台（由上层统一控制）
        if self.pbar is None:
            self.step_counter += 1
        else:
            self.pbar.update(1)

        # 触发器
        logging.debug(f"触发器数量: {len(self.triggers)}")
        logging.debug(f"触发器列表: {[t.__class__.__name__ for t in self.triggers]}")
        logging.debug("开始触发器感知阶段")
        
        # 阶段 1: 触发器感知
        for trigger in self.triggers:
            trigger.check_and_execute()  # 🚨 这里调用触发器的感知方法
        
        # 阶段 2: 执行指令
        self._execute_pending_actions()

    def stop(self):
        """回测结束时收尾行内进度显示。"""
        # 不在此处写入 stdout，避免污染控制台输出
        return
    
    def notify_order(self, order):
        """订单状态通知（可选覆写）"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"  ✓ 买入完成: {order.data._name} @ {order.executed.price:.2f}")
            else:
                self.log(f"  ✓ 卖出完成: {order.data._name} @ {order.executed.price:.2f}")
    
    def notify_trade(self, trade):
        """交易完成通知（可选覆写）"""
        if trade.isclosed:
            self.log(f"  💰 交易关闭: {trade.data._name} | 盈亏: {trade.pnl:.2f}")
