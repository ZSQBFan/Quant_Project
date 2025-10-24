# stop_loss.py
import backtrader as bt

class StopLossBase:
    params = ()
    
    def __init__(self, strategy, data, **kwargs):
        """
        【核心改造】初始化止损处理器，使其与特定的data feed绑定。
        """
        self.strategy = strategy
        self.data = data
        self._setup_params(**kwargs)

    @property
    def position(self):
        """ 获取与此处理器关联的data feed的持仓 """
        return self.strategy.getposition(self.data)

    def _setup_params(self, **kwargs):
        default_params = dict(self.params)
        for key, value in kwargs.items():
            if key in default_params:
                setattr(self, key, value)
        for key, value in default_params.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def check(self):
        raise NotImplementedError("每个止损策略都必须实现 check 方法")

    def notify_order(self, order):
        pass

class FixedPercentageStopLoss(StopLossBase):
    params = (('pct', 0.08),)

    def check(self):
        if not self.position:
            return False
            
        if self.data.close[0] < self.position.price * (1 - self.pct):
            self.strategy.log(f'[{self.data._name}] 固定百分比止损触发, 止损: {self.pct:.2%}, 入场价: {self.position.price:.2f}, 当前价: {self.data.close[0]:.2f}')
            return True
        return False

class ATRTrailingStopLoss(StopLossBase):
    params = (
        ('atr_period', 30),
        ('atr_multiplier', 3.0),
    )

    def __init__(self, strategy, data, **kwargs):
        super().__init__(strategy, data, **kwargs)
        self.atr = bt.indicators.AverageTrueRange(self.data, period=self.atr_period)
        self.trailing_stop_price = 0.0

    def notify_order(self, order):
        # 仅处理与此处理器关联的data的订单
        if order.data != self.data:
            return

        if order.isbuy() and order.status == order.Completed:
            entry_price = order.executed.price
            initial_stop = entry_price - (self.atr[0] * self.atr_multiplier)
            self.trailing_stop_price = initial_stop
            self.strategy.log(f'[{self.data._name}] ATR止损激活, 入场价: {entry_price:.2f}, 初始止损价: {self.trailing_stop_price:.2f}')

    def check(self):
        if not self.position or self.trailing_stop_price == 0.0:
            return False
        
        potential_new_stop = self.data.close[0] - (self.atr[0] * self.atr_multiplier)
        
        if potential_new_stop > self.trailing_stop_price:
            self.trailing_stop_price = potential_new_stop

        if self.data.close[0] < self.trailing_stop_price:
            self.strategy.log(f'[{self.data._name}] ATR追踪止损触发, 止损价: {self.trailing_stop_price:.2f}, 当前价: {self.data.close[0]:.2f}')
            # 重置止损价，避免在下一次持仓时使用旧值
            self.trailing_stop_price = 0.0
            return True
        
        return False