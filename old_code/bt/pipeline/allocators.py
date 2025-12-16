# bt/pipeline/allocators.py

"""
权重分配器模块

负责根据不同的分配策略为选中的股票分配权重。
主要职责：
1. 定义权重分配器接口和基础行为
2. 实现等权重分配策略
3. 处理各种边界情况（空列表、单股票等）
4. 提供详细的分配过程日志记录
"""

import math
import logging
from abc import ABC, abstractmethod
from typing import List, Any, Dict

from logger.logger_config import setup_logging


class AllocatorBase(ABC):
    """
    权重分配器基类
    
    定义权重分配器的标准接口和通用功能。所有分配器必须继承此类并实现 allocate 方法。
    
    设计原则：
    - 单一职责：只负责权重分配，不负责其他逻辑
    - 策略模式：不同分配策略可以互换
    - 错误容错：对异常数据有良好的容错能力
    """
    
    def __init__(self, name: str = None):
        """
        初始化权重分配器
        
        Args:
            name (str, optional): 分配器名称，用于日志区分
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(self.__class__.__module__)
        
    @abstractmethod
    def allocate(self, selected_stocks: List[Any]) -> Dict[Any, float]:
        """
        权重分配核心方法
        
        子类必须实现此方法，根据具体策略为选中的股票分配权重。
        
        Args:
            selected_stocks (List[Any]): 选中的股票数据对象列表
            
        Returns:
            Dict[Any, float]: {股票数据对象: 权重} 字典，所有权重之和应为 1.0
            
        Raises:
            NotImplementedError: 子类未实现此方法时抛出
        """
        raise NotImplementedError("子类必须实现 allocate 方法")
    
    def log_allocation_process(self, stock_count: int, weights: Dict[Any, float] = None):
        """
        记录权重分配过程统计信息
        
        Args:
            stock_count (int): 分配权重的股票数量
            weights (Dict[Any, float], optional): 实际分配的权重字典
        """
        self.logger.debug(f"权重分配器 [{self.name}] 开始分配权重")
        self.logger.debug(f"  待分配股票数量: {stock_count}")
        
        if weights and stock_count > 0:
            total_weight = sum(weights.values())
            self.logger.debug(f"  权重总和: {total_weight:.6f}")
            
            # 记录每只股票的权重（调试级别）
            self.logger.debug("  股票权重分配:")
            for i, (stock, weight) in enumerate(weights.items(), 1):
                stock_name = getattr(stock, '_name', f'Stock_{i}')
                self.logger.debug(f"    {i:2d}. {stock_name}: {weight:.6f}")
        
        if stock_count == 0:
            self.logger.warning(f"权重分配器 [{self.name}] 收到空股票列表，未进行权重分配")


class EqualWeightAllocator(AllocatorBase):
    """
    等权重分配器（简单平均分配）
    
    为所有选中的股票分配相等的权重，每只股票的权重 = 1.0 / N（N为股票数量）。
    
    分配逻辑：
    1. 验证输入列表非空
    2. 计算等权重：weight = 1.0 / stock_count
    3. 构建权重分配字典
    4. 处理边界情况
    
    适用场景：
    - 简单平均分配策略
    - 指数化投资
    - 风险分散化投资
    - 策略回测基线
    """
    
    def __init__(self, name: str = None):
        """
        初始化等权重分配器
        
        等权重分配器不需要额外的配置参数，所有股票都获得相等权重。
        
        Args:
            name (str, optional): 分配器名称
        """
        super().__init__(name or "EqualWeightAllocator")
        self.logger = logging.getLogger(__name__)
        
    def allocate(self, selected_stocks: List[Any]) -> Dict[Any, float]:
        """
        执行等权重分配逻辑
        
        详细处理流程：
        1. 输入验证和边界情况处理
        2. 计算等权重值
        3. 构建权重分配字典
        4. 验证权重总和（理论上应为 1.0）
        
        Args:
            selected_stocks (List[Any]): 选中的股票数据对象列表
            
        Returns:
            Dict[Any, float]: {股票数据对象: 权重} 字典，所有权重之和为 1.0
            
        Raises:
            ValueError: 当输入列表包含重复对象时
        """
        # 1. 输入验证
        if not isinstance(selected_stocks, list):
            self.logger.error(f"分配器 [{self.name}] 收到非列表输入: {type(selected_stocks)}")
            return {}
        
        stock_count = len(selected_stocks)
        
        # 2. 边界情况处理：空列表
        if stock_count == 0:
            self.logger.warning(f"分配器 [{self.name}] 收到空股票列表，返回空权重字典")
            return {}
        
        # 3. 边界情况处理：单股票
        if stock_count == 1:
            weight = 1.0
            stock = selected_stocks[0]
            stock_name = getattr(stock, '_name', 'Single_Stock')
            self.logger.debug(f"单股票分配: {stock_name} -> {weight:.6f}")
            return {stock: weight}
        
        # 4. 检查重复股票（防止权重分配错误）
        if len(selected_stocks) != len(set(selected_stocks)):
            self.logger.warning(f"分配器 [{self.name}] 检测到重复股票，可能导致权重分配错误")
        
        # 5. 计算等权重
        equal_weight = 1.0 / stock_count
        
        # 6. 构建权重分配字典
        weight_allocation = {}
        for stock in selected_stocks:
            weight_allocation[stock] = equal_weight
        
        # 7. 记录分配过程
        total_weight = sum(weight_allocation.values())
        self.log_allocation_process(stock_count, weight_allocation)
        
        # 8. 验证权重总和（处理浮点数精度问题）
        weight_sum_error = abs(total_weight - 1.0)
        if weight_sum_error > 1e-10:
            self.logger.warning(
                f"分配器 [{self.name}] 权重总和偏差较大: {total_weight:.10f}, 误差: {weight_sum_error:.2e}"
            )
        else:
            self.logger.debug(f"分配器 [{self.name}] 权重分配完成，总和: {total_weight:.10f}")
        
        return weight_allocation
    
    def __repr__(self):
        """返回分配器的字符串表示"""
        return f"{self.name}()"


# 便捷函数：创建标准等权重分配器
def create_equal_weight_allocator(name: str = None) -> EqualWeightAllocator:
    """
    创建等权重分配器的便捷函数
    
    Args:
        name (str, optional): 分配器名称
        
    Returns:
        EqualWeightAllocator: 配置好的分配器实例
    """
    return EqualWeightAllocator(name=name)


# 分配器注册表（便于管理和测试）
ALLOCATOR_REGISTRY = {
    'equal_weight': EqualWeightAllocator,
    'equal': lambda: EqualWeightAllocator(),
    'equal_weight_default': lambda: EqualWeightAllocator("DefaultEqualWeight"),
}


def get_allocator(allocator_type: str, **kwargs) -> AllocatorBase:
    """
    根据类型获取分配器实例
    
    Args:
        allocator_type (str): 分配器类型
        **kwargs: 分配器初始化参数
        
    Returns:
        AllocatorBase: 分配器实例
        
    Raises:
        ValueError: 不支持的分配器类型
    """
    if allocator_type not in ALLOCATOR_REGISTRY:
        raise ValueError(f"不支持的分配器类型: {allocator_type}")
    
    allocator_factory = ALLOCATOR_REGISTRY[allocator_type]
    return allocator_factory(**kwargs)


# 工具函数：验证权重分配结果
def validate_weight_allocation(weight_allocation: Dict[Any, float], 
                             tolerance: float = 1e-6) -> tuple[bool, float]:
    """
    验证权重分配结果的正确性
    
    Args:
        weight_allocation (Dict[Any, float]): 权重分配字典
        tolerance (float, optional): 允许的误差范围，默认为 1e-6
        
    Returns:
        tuple[bool, float]: (是否有效, 权重总和)
    """
    if not weight_allocation:
        return True, 0.0
    
    total_weight = sum(weight_allocation.values())
    is_valid = abs(total_weight - 1.0) <= tolerance
    
    return is_valid, total_weight


# 示例用法（调试和测试用）
if __name__ == "__main__":
    # 设置日志
    setup_logging(log_prefix='allocator_test')
    
    # 创建测试股票数据
    class MockStock:
        def __init__(self, name):
            self._name = name
    
    stocks = [MockStock(f"Stock_{i}") for i in range(1, 6)]
    
    # 测试等权重分配器
    allocator = EqualWeightAllocator()
    result = allocator.allocate(stocks)
    
    # 验证结果
    is_valid, total = validate_weight_allocation(result)
    print(f"分配结果验证: 有效={is_valid}, 总和={total:.10f}")
    print(f"每股权重: {1.0/len(stocks):.6f}")