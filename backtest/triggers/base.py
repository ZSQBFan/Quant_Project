# bt/triggers/base.py

"""
触发器基类

设计原则：
- 触发器只负责"观察"和"提交意图"
- 触发器不检查停牌状态（由策略基类统一处理）
- 每个触发器只关心一件事（单一职责）
"""


class TriggerBase:
    """
    触发器基类
    
    所有触发器必须继承此类并实现 check_and_execute() 方法。
    
    关于停牌处理：
    - 触发器提交意图时不需要检查停牌
    - 策略基类会在执行阶段统一过滤停牌股票
    - 这样设计可以追踪"哪些意图因停牌被拦截"
    """
    
    def __init__(self, strategy):
        """
        参数:
            strategy: BacktestStrategy 实例的引用
        """
        self.s = strategy
    
    def _iter_held_stocks(self):
        """
        生成器：遍历当前有持仓的股票（不过滤停牌）
        
        用于：止损、止盈等需要检查持仓的触发器
        """
        for d in self.s.datas:
            pos = self.s.getposition(d)
            if pos.size != 0:
                yield d
    
    def _iter_all_stocks(self):
        """
        生成器：遍历全市场所有股票（不过滤停牌）
        
        用于：选股等需要扫描全市场的触发器
        """
        for d in self.s.datas:
            yield d
    
    def _is_suspended(self, data) -> bool:
        """
        检查股票是否停牌（工具方法，非强制使用）
        
        注意：一般情况下触发器不需要调用此方法。
        只有在需要统计停牌数量等特殊场景才使用。
        """
        if hasattr(data, 'suspended') and len(data) > 0:
            return bool(data.suspended[0])
        return False
    
    def check_and_execute(self):
        """
        子类必须实现：检查条件并提交意图
        
        注意：不需要检查停牌，直接提交意图即可。
        """
        raise NotImplementedError("子类必须实现 check_and_execute 方法")