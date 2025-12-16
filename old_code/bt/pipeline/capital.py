# bt/pipeline/capital.py

"""
资金管理器模块

负责管理交易账户的资金分配，决定可用于交易的资金量。
主要职责：
1. 定义资金管理器接口和基础行为
2. 实现全仓资金分配策略
3. 处理各种边界情况（总价值异常、利用率越界等）
4. 提供详细的资金分配过程日志记录
5. 保留现金缓冲以应对紧急情况和交易成本

设计原则：
- 单一职责：只负责资金分配，不负责其他逻辑
- 策略模式：不同分配策略可以互换
- 错误容错：对异常数据有良好的容错能力
- 风险控制：保留适当的现金缓冲
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from logger.logger_config import setup_logging


class CapitalManagerBase(ABC):
    """
    资金管理器基类
    
    定义资金管理器的标准接口和通用功能。所有资金管理器必须继承此类并实现 get_allocation 方法。
    
    设计原则：
    - 单一职责：只负责资金分配，不负责其他逻辑
    - 策略模式：不同分配策略可以互换
    - 错误容错：对异常数据有良好的容错能力
    """
    
    def __init__(self, name: str = None):
        """
        初始化资金管理器
        
        Args:
            name (str, optional): 资金管理器名称，用于日志区分
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(self.__class__.__module__)
        
    @abstractmethod
    def get_allocation(self, total_value: float) -> float:
        """
        资金分配核心方法
        
        子类必须实现此方法，根据具体策略计算可用于交易的资金量。
        
        Args:
            total_value (float): 账户总价值
            
        Returns:
            float: 可用于交易的资金量
            
        Raises:
            NotImplementedError: 子类未实现此方法时抛出
        """
        raise NotImplementedError("子类必须实现 get_allocation 方法")
    
    def validate_total_value(self, total_value: float) -> tuple[bool, str]:
        """
        验证总价值的有效性
        
        Args:
            total_value (float): 待验证的账户总价值
            
        Returns:
            tuple[bool, str]: (是否有效, 错误信息)
        """
        if total_value is None:
            return False, "总价值不能为 None"
        
        if not isinstance(total_value, (int, float)):
            return False, f"总价值必须为数字类型，实际为: {type(total_value).__name__}"
        
        if total_value <= 0:
            return False, f"总价值必须大于0，实际为: {total_value}"
        
        if total_value != total_value:  # 检查 NaN
            return False, "总价值不能为 NaN"
        
        if abs(total_value) == float('inf'):
            return False, "总价值不能为无穷大"
        
        return True, ""
    
    def log_allocation_process(self, total_value: float, allocation: float, 
                             utilization_ratio: float = None, extra_info: str = ""):
        """
        记录资金分配过程统计信息
        
        Args:
            total_value (float): 账户总价值
            allocation (float): 实际分配的资金量
            utilization_ratio (float, optional): 资金利用率
            extra_info (str, optional): 额外信息
        """
        self.logger.debug(f"资金管理器 [{self.name}] 开始计算资金分配")
        self.logger.debug(f"  账户总价值: {total_value:,.2f}")
        
        if utilization_ratio is not None:
            self.logger.debug(f"  资金利用率: {utilization_ratio:.4f} ({utilization_ratio*100:.2f}%)")
            reserved_cash = total_value * (1 - utilization_ratio)
            self.logger.debug(f"  预留现金缓冲: {reserved_cash:,.2f}")
        
        self.logger.debug(f"  分配资金量: {allocation:,.2f}")
        self.logger.debug(f"  现金保留比例: {(1 - allocation/total_value)*100:.2f}%" if allocation > 0 else "  现金保留比例: 100.00%")
        
        if extra_info:
            self.logger.debug(f"  额外信息: {extra_info}")


class FullPositionManager(CapitalManagerBase):
    """
    全仓资金管理器（MVP 最简实现）
    
    使用指定的资金利用率计算可用于交易的资金量。默认保留5%的现金缓冲。
    
    现金缓冲的作用：
    1. 应对紧急情况：市场突发事件需要额外资金时
    2. 交易成本：佣金、印花税等交易费用
    3. 价格波动：买入时价格可能高于预期
    4. 滑点控制：确保有足够资金完成预期交易
    5. 系统安全：避免因资金不足导致的强制平仓
    
    分配逻辑：
    1. 验证输入参数的有效性
    2. 检查利用率是否在合理范围内
    3. 计算可用交易资金：allocation = total_value * utilization_ratio
    4. 记录详细的分配过程
    5. 处理各种边界情况
    
    适用场景：
    - 高频交易策略（需要预留更多现金）
    - 波动性较大的市场环境
    - 风险控制要求严格的策略
    - 需要应对交易成本的场景
    """
    
    def __init__(self, utilization_ratio: float = 0.95, name: str = None):
        """
        初始化全仓资金管理器
        
        Args:
            utilization_ratio (float, optional): 资金利用率，默认 0.95（保留5%现金缓冲）
            name (str, optional): 资金管理器名称
            
        Raises:
            ValueError: 利用率参数不在合理范围内时抛出
        """
        super().__init__(name or "FullPositionManager")
        
        # 验证利用率参数
        if not isinstance(utilization_ratio, (int, float)):
            raise ValueError(f"利用率必须为数字类型，实际为: {type(utilization_ratio).__name__}")
        
        if utilization_ratio < 0 or utilization_ratio > 1:
            raise ValueError(f"利用率必须在 [0, 1] 范围内，实际为: {utilization_ratio}")
        
        self.utilization_ratio = float(utilization_ratio)
        self.reserved_ratio = 1.0 - self.utilization_ratio
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"资金管理器 [{self.name}] 初始化完成")
        self.logger.info(f"  资金利用率: {self.utilization_ratio:.4f} ({self.utilization_ratio*100:.2f}%)")
        self.logger.info(f"  现金缓冲比例: {self.reserved_ratio:.4f} ({self.reserved_ratio*100:.2f}%)")
        
        if self.utilization_ratio > 0.99:
            self.logger.warning("资金利用率过高，可能存在流动性风险")
        elif self.utilization_ratio < 0.5:
            self.logger.info("资金利用率较低，资金使用效率可能不高")
    
    def get_allocation(self, total_value: float) -> float:
        """
        计算可用于交易的资金量
        
        详细处理流程：
        1. 验证总价值参数的有效性
        2. 检查利用率是否仍然有效
        3. 计算可用资金：allocation = total_value * utilization_ratio
        4. 处理浮点数精度问题
        5. 记录详细的分配过程
        
        Args:
            total_value (float): 账户总价值
            
        Returns:
            float: 可用于交易的资金量
            
        Raises:
            ValueError: 当总价值无效时
        """
        # 1. 验证总价值参数
        is_valid, error_msg = self.validate_total_value(total_value)
        if not is_valid:
            self.logger.error(f"资金管理器 [{self.name}] 总价值验证失败: {error_msg}")
            raise ValueError(f"总价值验证失败: {error_msg}")
        
        # 2. 二次检查利用率有效性（防止外部修改）
        if self.utilization_ratio < 0 or self.utilization_ratio > 1:
            self.logger.error(f"资金管理器 [{self.name}] 利用率越界: {self.utilization_ratio}")
            raise ValueError(f"资金利用率越界: {self.utilization_ratio}")
        
        # 3. 计算可用交易资金
        # 使用 float() 确保精度，处理可能的Decimal或其他数值类型
        total_value = float(total_value)
        allocation = total_value * self.utilization_ratio
        
        # 4. 处理浮点数精度问题
        allocation = round(allocation, 10)  # 保留10位小数精度
        
        # 5. 边界情况处理：确保分配金额不超过总价值
        if allocation > total_value:
            self.logger.warning(
                f"资金管理器 [{self.name}] 计算出的分配金额 {allocation:.2f} 超过总价值 {total_value:.2f}，"
                f"调整为总价值"
            )
            allocation = total_value
        
        # 6. 确保分配金额不为负数
        if allocation < 0:
            self.logger.error(
                f"资金管理器 [{self.name}] 计算出负分配金额 {allocation:.2f}，调整为0"
            )
            allocation = 0.0
        
        # 7. 记录分配过程
        extra_info = ""
        if allocation == 0:
            extra_info = "分配金额为0，可能是因为总价值过小或利用率过低"
        elif allocation == total_value:
            extra_info = "全部分配，无现金缓冲"
        elif abs(allocation - total_value * self.utilization_ratio) > 1e-6:
            extra_info = "经过精度调整"
        
        self.log_allocation_process(total_value, allocation, self.utilization_ratio, extra_info)
        
        # 8. 最终验证
        if allocation > total_value:
            raise RuntimeError(f"资金管理器 [{self.name}] 分配金额 {allocation:.2f} 超过总价值 {total_value:.2f}")
        
        if allocation < 0:
            raise RuntimeError(f"资金管理器 [{self.name}] 分配金额 {allocation:.2f} 为负数")
        
        self.logger.debug(f"资金管理器 [{self.name}] 分配计算完成: {allocation:,.2f}")
        return allocation
    
    def get_utilization_ratio(self) -> float:
        """
        获取当前的资金利用率
        
        Returns:
            float: 资金利用率
        """
        return self.utilization_ratio
    
    def get_reserved_ratio(self) -> float:
        """
        获取当前的现金缓冲比例
        
        Returns:
            float: 现金缓冲比例
        """
        return self.reserved_ratio
    
    def update_utilization_ratio(self, utilization_ratio: float):
        """
        更新资金利用率
        
        Args:
            utilization_ratio (float): 新的资金利用率
            
        Raises:
            ValueError: 利用率参数不在合理范围内时抛出
        """
        if not isinstance(utilization_ratio, (int, float)):
            raise ValueError(f"利用率必须为数字类型，实际为: {type(utilization_ratio).__name__}")
        
        if utilization_ratio < 0 or utilization_ratio > 1:
            raise ValueError(f"利用率必须在 [0, 1] 范围内，实际为: {utilization_ratio}")
        
        old_ratio = self.utilization_ratio
        self.utilization_ratio = float(utilization_ratio)
        self.reserved_ratio = 1.0 - self.utilization_ratio
        
        self.logger.info(f"资金管理器 [{self.name}] 更新利用率: {old_ratio:.4f} -> {self.utilization_ratio:.4f}")
        
        if self.utilization_ratio > 0.99:
            self.logger.warning("更新后资金利用率过高，可能存在流动性风险")
    
    def __repr__(self):
        """返回资金管理器的字符串表示"""
        return f"{self.name}(utilization_ratio={self.utilization_ratio:.4f})"


# 便捷函数：创建标准全仓资金管理器
def create_full_position_manager(utilization_ratio: float = 0.95, name: str = None) -> FullPositionManager:
    """
    创建全仓资金管理器的便捷函数
    
    Args:
        utilization_ratio (float, optional): 资金利用率，默认 0.95
        name (str, optional): 资金管理器名称
        
    Returns:
        FullPositionManager: 配置好的资金管理器实例
    """
    return FullPositionManager(utilization_ratio=utilization_ratio, name=name)


# 资金管理器注册表（便于管理和测试）
CAPITAL_MANAGER_REGISTRY = {
    'full_position': FullPositionManager,
    'full': lambda ratio=0.95: FullPositionManager(utilization_ratio=ratio),
    'conservative': lambda: FullPositionManager(utilization_ratio=0.90, name="ConservativeManager"),
    'aggressive': lambda: FullPositionManager(utilization_ratio=0.98, name="AggressiveManager"),
    'safe': lambda: FullPositionManager(utilization_ratio=0.85, name="SafeManager"),
}


def get_capital_manager(manager_type: str, **kwargs) -> CapitalManagerBase:
    """
    根据类型获取资金管理器实例
    
    Args:
        manager_type (str): 资金管理器类型
        **kwargs: 资金管理器初始化参数
        
    Returns:
        CapitalManagerBase: 资金管理器实例
        
    Raises:
        ValueError: 不支持的资金管理器类型
    """
    if manager_type not in CAPITAL_MANAGER_REGISTRY:
        raise ValueError(f"不支持的资金管理器类型: {manager_type}")
    
    manager_factory = CAPITAL_MANAGER_REGISTRY[manager_type]
    return manager_factory(**kwargs)


# 示例用法（调试和测试用）
if __name__ == "__main__":
    # 设置日志
    setup_logging(log_prefix='capital_manager_test')
    
    # 测试不同场景
    test_cases = [
        (100000.0, 0.95, "标准场景"),
        (100000.0, 0.99, "高利用率"),
        (100000.0, 0.90, "保守策略"),
        (50000.0, 0.95, "中等资金"),
        (1000000.0, 0.85, "大资金保守"),
    ]
    
    print("=== 资金管理器测试 ===")
    
    for total_value, utilization_ratio, description in test_cases:
        print(f"\n测试场景: {description}")
        print(f"总价值: {total_value:,.2f}")
        print(f"利用率: {utilization_ratio:.2%}")
        
        try:
            manager = FullPositionManager(utilization_ratio=utilization_ratio)
            allocation = manager.get_allocation(total_value)
            reserved_cash = total_value - allocation
            
            print(f"分配资金: {allocation:,.2f}")
            print(f"预留现金: {reserved_cash:,.2f}")
            print(f"实际利用率: {allocation/total_value:.2%}")
            
        except Exception as e:
            print(f"错误: {e}")
    
    # 测试边界情况
    print("\n=== 边界情况测试 ===")
    
    edge_cases = [
        (0, 0.95, "零总价值"),
        (-10000, 0.95, "负总价值"),
        (100000, 1.5, "超100%利用率"),
        (100000, -0.1, "负利用率"),
    ]
    
    for total_value, utilization_ratio, description in edge_cases:
        print(f"\n边界测试: {description}")
        try:
            manager = FullPositionManager(utilization_ratio=utilization_ratio)
            allocation = manager.get_allocation(total_value)
            print(f"分配结果: {allocation:,.2f}")
        except Exception as e:
            print(f"预期错误: {e}")