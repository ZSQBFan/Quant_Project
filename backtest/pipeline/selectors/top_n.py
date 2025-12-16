# bt/pipeline/selectors.py

"""
选股器模块

负责从全市场股票中按照特定策略选择出符合条件的股票。
主要职责：
1. 定义选股器接口和基础行为
2. 实现基于信号强度的选股逻辑
3. 处理各种边界情况（停牌、NaN、无数据等）
4. 提供详细的选股过程日志记录
"""

import math
import logging
from abc import ABC, abstractmethod
from typing import List, Any

from utils.logger import setup_logging


class SelectorBase(ABC):
    """
    选股器基类
    
    定义选股器的标准接口和通用功能。所有选股器必须继承此类并实现 select 方法。
    
    设计原则：
    - 单一职责：只负责选股，不负责其他逻辑
    - 策略模式：不同选股策略可以互换
    - 错误容错：对异常数据有良好的容错能力
    """
    
    def __init__(self, name: str = None):
        """
        初始化选股器
        
        Args:
            name (str, optional): 选股器名称，用于日志区分
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(self.__class__.__module__)
        
    @abstractmethod
    def select(self, datas: List[Any]) -> List[Any]:
        """
        选股核心方法
        
        子类必须实现此方法，根据具体策略从数据列表中选出符合条件的股票。
        
        Args:
            datas (List[Any]): 候选股票数据对象列表
            
        Returns:
            List[Any]: 选中的股票数据对象列表
            
        Raises:
            NotImplementedError: 子类未实现此方法时抛出
        """
        raise NotImplementedError("子类必须实现 select 方法")
    
    def log_selection_process(self, total_candidates: int, selected_count: int, 
                            filtered_out: dict = None):
        """
        记录选股过程统计信息
        
        Args:
            total_candidates (int): 候选股票总数
            selected_count (int): 最终选中股票数量
            filtered_out (dict, optional): 被过滤的股票统计 {过滤原因: 数量}
        """
        self.logger.debug(f"选股器 [{self.name}] 开始选股")
        self.logger.debug(f"  候选股票总数: {total_candidates}")
        
        if filtered_out:
            self.logger.debug("  过滤统计:")
            for reason, count in filtered_out.items():
                self.logger.debug(f"    - {reason}: {count} 只")
        
        self.logger.debug(f"  最终选中: {selected_count} 只股票")
        
        if selected_count == 0:
            self.logger.warning(f"选股器 [{self.name}] 未选中任何股票，请检查筛选条件")


class TopNSelector(SelectorBase):
    """
    Top N 选股器（基于信号强度排序）
    
    根据股票的综合信号强度进行排序，选择信号最强的 N 只股票。
    
    筛选逻辑：
    1. 跳过无数据股票（len(d) == 0）
    2. 跳过停牌股票（d.suspended[0] == True）
    3. 跳过信号为 NaN 的股票
    4. 按信号强度降序排序
    5. 返回前 N 只股票
    
    适用场景：
    - 因子选股策略
    - 信号强度排序
    - 市场热点捕捉
    """
    
    def __init__(self, top_n: int = 10, name: str = None):
        """
        初始化 Top N 选股器
        
        Args:
            top_n (int, optional): 要选择的股票数量，默认为 10
            name (str, optional): 选股器名称
        """
        super().__init__(name or f"Top{top_n}Selector")
        self.top_n = top_n
        self.logger = logging.getLogger(__name__)
        
    def select(self, datas: List[Any]) -> List[Any]:
        """
        执行 Top N 选股逻辑
        
        详细处理流程：
        1. 输入验证和初始化统计
        2. 遍历所有候选股票，应用筛选条件
        3. 按信号强度排序
        4. 返回前 N 只股票
        
        Args:
            datas (List[Any]): 候选股票数据对象列表
            
        Returns:
            List[Any]: 按信号强度排序的前 N 只股票
        """
        # 输入验证
        if not datas:
            self.logger.warning(f"选股器 [{self.name}] 收到空候选列表")
            return []
        
        total_candidates = len(datas)
        filtered_stats = {
            "无数据股票": 0,
            "停牌股票": 0,
            "信号为NaN": 0,
            "信号获取失败": 0
        }
        
        valid_stocks = []
        
        for d in datas:
            try:
                # 1. 检查数据长度
                if len(d) == 0:
                    filtered_stats["无数据股票"] += 1
                    continue
                
                # 2. 检查停牌状态
                if hasattr(d, 'suspended') and d.suspended[0]:
                    filtered_stats["停牌股票"] += 1
                    continue
                
                # 3. 获取综合信号
                if not hasattr(d, 'combined_signal'):
                    filtered_stats["信号获取失败"] += 1
                    self.logger.debug(f"股票 {getattr(d, '_name', 'Unknown')} 缺少 combined_signal 属性")
                    continue
                
                signal = d.combined_signal[0]
                
                # 4. 检查信号有效性
                if math.isnan(signal):
                    filtered_stats["信号为NaN"] += 1
                    self.logger.debug(f"股票 {getattr(d, '_name', 'Unknown')} 信号为 NaN")
                    continue
                
                # 5. 收集有效股票及其信号
                stock_name = getattr(d, '_name', 'Unknown')
                valid_stocks.append((d, signal, stock_name))
                
            except (AttributeError, IndexError, TypeError) as e:
                filtered_stats["信号获取失败"] += 1
                stock_name = getattr(d, '_name', 'Unknown')
                self.logger.debug(f"股票 {stock_name} 信号获取失败: {str(e)}")
                continue
        
        # 按信号强度降序排序
        valid_stocks.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前 N 只股票
        selected_stocks = [stock[0] for stock in valid_stocks[:self.top_n]]
        selected_count = len(selected_stocks)
        
        # 记录选股过程
        self.log_selection_process(total_candidates, selected_count, filtered_stats)
        
        # 记录选中的股票（调试级别）
        if selected_count > 0:
            self.logger.debug("选中的股票列表:")
            for i, (stock, signal, name) in enumerate(valid_stocks[:selected_count], 1):
                self.logger.debug(f"  {i:2d}. {name} (信号: {signal:.4f})")
        
        return selected_stocks
    
    def __repr__(self):
        """返回选股器的字符串表示"""
        return f"{self.name}(top_n={self.top_n})"


# 便捷函数：创建标准选股器
def create_top_n_selector(top_n: int = 10, name: str = None) -> TopNSelector:
    """
    创建 Top N 选股器的便捷函数
    
    Args:
        top_n (int, optional): 要选择的股票数量
        name (str, optional): 选股器名称
        
    Returns:
        TopNSelector: 配置好的选股器实例
    """
    return TopNSelector(top_n=top_n, name=name)


# 选股器注册表（便于管理和测试）
SELECTOR_REGISTRY = {
    'top_n': TopNSelector,
    'top10': lambda: TopNSelector(10),
    'top20': lambda: TopNSelector(20),
    'top50': lambda: TopNSelector(50),
}


def get_selector(selector_type: str, **kwargs) -> SelectorBase:
    """
    根据类型获取选股器实例
    
    Args:
        selector_type (str): 选股器类型
        **kwargs: 选股器初始化参数
        
    Returns:
        SelectorBase: 选股器实例
        
    Raises:
        ValueError: 不支持的选股器类型
    """
    if selector_type not in SELECTOR_REGISTRY:
        raise ValueError(f"不支持的选股器类型: {selector_type}")
    
    selector_factory = SELECTOR_REGISTRY[selector_type]
    return selector_factory(**kwargs)