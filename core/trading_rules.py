# trading_rules.py
import backtrader as bt
import numpy as np

class MultiSignalStrategy(bt.Strategy):
    params = (
        ('printlog', False),
        ('rebalance_days', 20),      # 【新增】调仓周期，例如20个交易日（约每月一次）
        ('signal_configs', None),
        ('stop_loss_configs', None),
        ('decision_logic_config', None),
    )

    def __init__(self):
        """
        【核心改造】初始化以支持多股票和定时调仓
        """
        self.order = None # order 属性在多股票场景下意义不大，但保留以防万一
        self.rebalance_timer = 0 # 调仓计时器

        # --- 模块化加载机制 (改造为支持多数据) ---
        
        # 1. 为每只股票初始化信号处理器
        self.signal_handlers = {} # key: d._name, value: list of signal handlers
        if self.p.signal_configs:
            print("--- 加载信号模块 ---")
            for d in self.datas:
                d_name = d._name
                self.signal_handlers[d_name] = []
                for sig_class, sig_params in self.p.signal_configs:
                    # 【重要】将 data=d 传入，使信号与特定股票数据绑定
                    self.signal_handlers[d_name].append(sig_class(self, data=d, **sig_params))
            print(f"✅ 已为 {len(self.datas)} 只股票加载 {len(self.p.signal_configs)} 个信号模块")
            print("--------------------")

        # 2. 为每只股票初始化止损处理器
        self.stop_loss_handlers = {} # key: d._name, value: list of stop loss handlers
        if self.p.stop_loss_configs:
            print("--- 加载止损模块 ---")
            for d in self.datas:
                d_name = d._name
                self.stop_loss_handlers[d_name] = []
                for sl_class, sl_params in self.p.stop_loss_configs:
                    self.stop_loss_handlers[d_name].append(sl_class(self, data=d, **sl_params))
            print(f"✅ 已为 {len(self.datas)} 只股票加载 {len(self.p.stop_loss_configs)} 个止损模块")
            print("--------------------")
            
        # 3. 初始化决策逻辑处理器
        self.decision_handler = None
        if self.p.decision_logic_config:
            logic_class, logic_params = self.p.decision_logic_config
            self.decision_handler = logic_class(self, **logic_params)
            print("--- 加载决策逻辑模块 ---")
            print(f"  -> {logic_class.__name__} (参数: {logic_params})")
            print("--------------------")
        else:
            raise ValueError("错误：必须配置一个决策逻辑模块 (decision_logic_config)")

    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {txt}')
            
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.log(f'BUY EXECUTED [{order.data._name}], Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED [{order.data._name}], Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected for {order.data._name}')
        
        # 将订单通知分发给对应股票的止损处理器
        if order.data and order.data._name in self.stop_loss_handlers:
            for handler in self.stop_loss_handlers[order.data._name]:
                handler.notify_order(order)
            
        self.order = None

    def next(self):
        # 1. 【高优先级】每日检查所有持仓的止损条件
        for d in self.datas:
            if self.getposition(d): # 如果持有该股票的仓位
                d_name = d._name
                sl_handlers = self.stop_loss_handlers.get(d_name, [])
                for handler in sl_handlers:
                    if handler.check():
                        self.log(f'止损模块 {handler.__class__.__name__} 触发平仓 [{d_name}]')
                        self.close(data=d) # 平掉这只股票的仓位
                        break # 一旦触发，不再检查该股票的其他止损

        # 2. 【常规优先级】检查是否为调仓日
        self.rebalance_timer += 1
        if self.rebalance_timer < self.p.rebalance_days:
            return # 未到调仓日，直接返回

        self.rebalance_timer = 0 # 重置计时器
        
        # --- 执行调仓逻辑 ---
        
        # 3. 获取目标投资组合
        target_portfolio_data = self.decision_handler.decide()
        if not target_portfolio_data:
            # 如果目标组合为空，则清仓所有股票
            self.log("目标组合为空，清仓所有头寸。")
            for d in self.datas:
                if self.getposition(d):
                    self.order_target_percent(data=d, target=0.0)
            return
            
        target_symbols = {d._name for d in target_portfolio_data}
        
        # 4. 调仓 - 卖出逻辑
        # 卖出不在新目标组合中的现有持仓
        for d in self.datas:
            if self.getposition(d) and d._name not in target_symbols:
                self.log(f"调仓卖出: {d._name}")
                self.order_target_percent(data=d, target=0.0)

        # 5. 调仓 - 买入逻辑
        # 为目标组合中的每只股票分配等权重资金
        target_weight = 1.0 / len(target_portfolio_data)
        for d in target_portfolio_data:
            self.log(f"调仓买入/调整: {d._name}至{target_weight:.2%}")
            self.order_target_percent(data=d, target=target_weight)