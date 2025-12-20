# bt/core/constants.py

"""
Backtrader 策略常量定义
"""


class ActionPriority:
    """
    指令优先级定义
    
    数值越小，优先级越高。
    当同一只股票同时触发多个信号时，高优先级指令会覆盖低优先级指令。
    
    设计原理：
    - 风控操作（止损/止盈）必须优先执行，保护资金安全
    - 调仓操作优先级较低，可被风控覆盖
    
    示例场景：
    - 某股票同时触发止损和调仓买入 -> 止损优先，执行卖出
    - 某股票同时触发止盈和普通卖出 -> 止盈优先，记录原因为止盈
    """
    STOP_LOSS = 1      # 风控最高优先级（救命用的）
    TAKE_PROFIT = 2    # 止盈
    REBALANCE = 3      # 正常调仓
    SECTOR_BUY = 4     # 行业轮动等其他信号
    OTHER = 5          # 最低优先级


class ActionType:
    """
    交易动作类型
    """
    BUY = 'buy'
    SELL = 'sell'
    CLOSE = 'close'    # 平仓（无论多空）