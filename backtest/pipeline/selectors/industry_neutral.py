# backtest/pipeline/selectors/industry_neutral.py

"""
行业中性化选股器

在各行业中均衡选股，避免行业集中风险。
主要职责：
1. 从全市场股票中按行业分组
2. 在每个行业内按信号强度排序
3. 按配置的策略在各行业间分配选股数量
4. 返回行业分散的股票组合
"""

import os
import json
import math
import logging
from abc import ABC, abstractmethod
from typing import List, Any, Dict
from collections import defaultdict

from .top_n import SelectorBase


class IndustryNeutralSelector(SelectorBase):
    """
    行业中性化选股器

    在各行业中均衡选股，通过控制行业暴露来降低行业风险。

    选股逻辑：
    1. 过滤：跳过无数据、停牌、信号NaN的股票
    2. 行业分组：按行业字段分组
    3. 行业内排序：在每个行业内按信号强度降序排序
    4. 行业间分配：按配置策略在各行业间分配选股数量
       - 'equal': 每个行业选择相同数量的股票
       - 'proportional': 按各行业股票数量比例分配
       - 'top_industries': 只在信号最强的N个行业中选股
    5. 返回最终选中的股票

    适用场景：
    - 需要控制行业暴露的策略
    - 避免行业集中风险
    - 行业轮动策略
    """

    def __init__(self,
                 total_stocks: int = 10,
                 strategy: str = 'equal',
                 stocks_per_industry: int = None,
                 top_industries: int = None,
                 min_stocks_per_industry: int = 1,
                 name: str = None,
                 industry_mapping_path: str = None):
        """
        初始化行业中性化选股器

        Args:
            total_stocks (int): 总共要选择的股票数量，默认为 10
            strategy (str): 行业间分配策略，可选：
                           'equal' - 每个行业选择相同数量（默认）
                           'proportional' - 按行业股票数量比例分配
                           'top_industries' - 只在头部行业中选股
            stocks_per_industry (int, optional): 每个行业选择的股票数量
                                                 仅在 strategy='equal' 时有效
                                                 如不指定，将根据 total_stocks 自动计算
            top_industries (int, optional): 选择的行业数量
                                           仅在 strategy='top_industries' 时有效
            min_stocks_per_industry (int): 每个行业最少选择的股票数量，默认为 1
            name (str, optional): 选股器名称
            industry_mapping_path (str, optional): 行业编码映射文件路径
        """
        super().__init__(name or f"IndustryNeutral{total_stocks}Selector")
        self.total_stocks = total_stocks
        self.strategy = strategy
        self.stocks_per_industry = stocks_per_industry
        self.top_industries = top_industries
        self.min_stocks_per_industry = min_stocks_per_industry
        self.logger = logging.getLogger(__name__)

        # 加载行业编码映射
        self.code_to_industry = {}
        if industry_mapping_path is None:
            industry_mapping_path = 'temp/backtest_data/industry_mapping.json'  # 默认路径

        if os.path.exists(industry_mapping_path):
            with open(industry_mapping_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
                # 将字符串键转换为整数
                self.code_to_industry = {int(k): v for k, v in mapping_data.get('code_to_industry', {}).items()}
            self.logger.info(f"✅ 加载行业编码映射: {len(self.code_to_industry)} 个行业")
        else:
            self.logger.warning(f"⚠️ 行业编码映射文件不存在: {industry_mapping_path}")

        # 参数验证
        if strategy not in ['equal', 'proportional', 'top_industries']:
            raise ValueError(f"不支持的分配策略: {strategy}")

        if strategy == 'top_industries' and top_industries is None:
            raise ValueError("strategy='top_industries' 时必须指定 top_industries 参数")

    def select(self, datas: List[Any]) -> List[Any]:
        """
        执行行业中性化选股逻辑

        详细处理流程：
        1. 输入验证和初始化统计
        2. 遍历所有候选股票，应用筛选条件并按行业分组
        3. 在每个行业内按信号强度排序
        4. 根据策略在各行业间分配选股数量
        5. 返回最终选中的股票

        Args:
            datas (List[Any]): 候选股票数据对象列表

        Returns:
            List[Any]: 行业分散的股票列表
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
            "信号获取失败": 0,
            "无行业信息": 0
        }

        # 按行业分组存储有效股票
        industry_stocks = defaultdict(list)

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

                # 5. 获取行业信息
                if not hasattr(d, 'industry'):
                    filtered_stats["无行业信息"] += 1
                    self.logger.debug(f"股票 {getattr(d, '_name', 'Unknown')} 缺少 industry 属性")
                    continue

                industry_code = d.industry[0]

                # 检查行业信息有效性（-1 表示无行业信息）
                if industry_code == -1 or (isinstance(industry_code, float) and math.isnan(industry_code)):
                    filtered_stats["无行业信息"] += 1
                    self.logger.debug(f"股票 {getattr(d, '_name', 'Unknown')} 行业信息为空")
                    continue

                # 将行业编码转换为行业名称（用于分组）
                industry_name = self.code_to_industry.get(int(industry_code), f"未知行业{int(industry_code)}")

                # 6. 收集有效股票并按行业分组
                stock_name = getattr(d, '_name', 'Unknown')
                industry_stocks[industry_name].append((d, signal, stock_name))

            except (AttributeError, IndexError, TypeError) as e:
                filtered_stats["信号获取失败"] += 1
                stock_name = getattr(d, '_name', 'Unknown')
                self.logger.debug(f"股票 {stock_name} 处理失败: {str(e)}")
                continue

        # 在每个行业内按信号强度降序排序
        for industry in industry_stocks:
            industry_stocks[industry].sort(key=lambda x: x[1], reverse=True)

        # 根据策略分配选股
        selected_stocks = self._allocate_stocks(industry_stocks)

        selected_count = len(selected_stocks)

        # 记录选股过程
        self.log_selection_process(total_candidates, selected_count, filtered_stats)

        # 记录行业分布
        self._log_industry_distribution(selected_stocks, industry_stocks)

        return selected_stocks

    def _allocate_stocks(self, industry_stocks: Dict[str, List]) -> List[Any]:
        """
        根据策略在各行业间分配选股数量

        Args:
            industry_stocks: {行业名: [(data, signal, name), ...]} 字典

        Returns:
            选中的股票数据对象列表
        """
        if not industry_stocks:
            return []

        selected_stocks = []

        if self.strategy == 'equal':
            # 策略1: 每个行业选择相同数量
            selected_stocks = self._equal_allocation(industry_stocks)

        elif self.strategy == 'proportional':
            # 策略2: 按行业股票数量比例分配
            selected_stocks = self._proportional_allocation(industry_stocks)

        elif self.strategy == 'top_industries':
            # 策略3: 只在头部行业中选股
            selected_stocks = self._top_industries_allocation(industry_stocks)

        return selected_stocks

    def _equal_allocation(self, industry_stocks: Dict[str, List]) -> List[Any]:
        """
        等权分配策略：每个行业选择相同数量的股票

        Args:
            industry_stocks: {行业名: [(data, signal, name), ...]} 字典

        Returns:
            选中的股票数据对象列表
        """
        num_industries = len(industry_stocks)

        # 计算每个行业应选择的股票数量
        if self.stocks_per_industry is not None:
            stocks_per_ind = self.stocks_per_industry
        else:
            stocks_per_ind = max(self.min_stocks_per_industry,
                                math.ceil(self.total_stocks / num_industries))

        selected_stocks = []

        for industry, stocks in industry_stocks.items():
            # 从每个行业中选择前 N 只信号最强的股票
            industry_selected = [stock[0] for stock in stocks[:stocks_per_ind]]
            selected_stocks.extend(industry_selected)

            self.logger.debug(f"  行业 [{industry}]: 选择 {len(industry_selected)}/{len(stocks)} 只股票")

        # 如果总数超过目标，按信号强度排序后截取
        if len(selected_stocks) > self.total_stocks:
            # 收集所有股票及其信号，重新排序
            all_with_signals = []
            for industry, stocks in industry_stocks.items():
                for stock_data, signal, name in stocks:
                    if stock_data in selected_stocks:
                        all_with_signals.append((stock_data, signal))

            all_with_signals.sort(key=lambda x: x[1], reverse=True)
            selected_stocks = [stock[0] for stock in all_with_signals[:self.total_stocks]]

        return selected_stocks

    def _proportional_allocation(self, industry_stocks: Dict[str, List]) -> List[Any]:
        """
        比例分配策略：按行业股票数量比例分配选股数量

        Args:
            industry_stocks: {行业名: [(data, signal, name), ...]} 字典

        Returns:
            选中的股票数据对象列表
        """
        total_valid_stocks = sum(len(stocks) for stocks in industry_stocks.values())

        selected_stocks = []
        allocated_counts = {}

        # 第一轮：按比例分配
        for industry, stocks in industry_stocks.items():
            proportion = len(stocks) / total_valid_stocks
            allocated = max(self.min_stocks_per_industry,
                          int(self.total_stocks * proportion))
            allocated_counts[industry] = min(allocated, len(stocks))

        # 确保不超过总数
        total_allocated = sum(allocated_counts.values())
        if total_allocated > self.total_stocks:
            # 按行业股票数量降序排列，从大行业开始减少
            industries_sorted = sorted(allocated_counts.items(),
                                     key=lambda x: len(industry_stocks[x[0]]),
                                     reverse=True)

            excess = total_allocated - self.total_stocks
            for industry, count in industries_sorted:
                if excess <= 0:
                    break
                reduce_by = min(excess, count - self.min_stocks_per_industry)
                if reduce_by > 0:
                    allocated_counts[industry] -= reduce_by
                    excess -= reduce_by

        # 从每个行业选股
        for industry, stocks in industry_stocks.items():
            count = allocated_counts.get(industry, 0)
            industry_selected = [stock[0] for stock in stocks[:count]]
            selected_stocks.extend(industry_selected)

            self.logger.debug(f"  行业 [{industry}]: 选择 {len(industry_selected)}/{len(stocks)} 只股票 (比例: {len(stocks)/total_valid_stocks:.1%})")

        return selected_stocks

    def _top_industries_allocation(self, industry_stocks: Dict[str, List]) -> List[Any]:
        """
        头部行业策略：只在信号最强的N个行业中选股

        Args:
            industry_stocks: {行业名: [(data, signal, name), ...]} 字典

        Returns:
            选中的股票数据对象列表
        """
        # 计算每个行业的平均信号强度
        industry_avg_signals = {}
        for industry, stocks in industry_stocks.items():
            if stocks:
                avg_signal = sum(stock[1] for stock in stocks) / len(stocks)
                industry_avg_signals[industry] = avg_signal

        # 选择信号最强的 N 个行业
        top_industries_list = sorted(industry_avg_signals.items(),
                                    key=lambda x: x[1],
                                    reverse=True)[:self.top_industries]

        top_industry_names = [ind[0] for ind in top_industries_list]

        # 在选中的行业中等权分配
        filtered_industry_stocks = {
            ind: stocks for ind, stocks in industry_stocks.items()
            if ind in top_industry_names
        }

        self.logger.info(f"  选择头部 {len(top_industry_names)} 个行业: {top_industry_names}")

        # 使用等权分配策略
        return self._equal_allocation(filtered_industry_stocks)

    def _log_industry_distribution(self, selected_stocks: List[Any],
                                   industry_stocks: Dict[str, List]):
        """
        记录行业分布信息

        Args:
            selected_stocks: 选中的股票列表
            industry_stocks: 全部行业股票字典
        """
        if not selected_stocks:
            return

        # 统计选中股票的行业分布
        selected_by_industry = defaultdict(int)
        for stock in selected_stocks:
            if hasattr(stock, 'industry') and len(stock) > 0:
                industry_code = stock.industry[0]
                if industry_code != -1 and not (isinstance(industry_code, float) and math.isnan(industry_code)):
                    # 将行业编码转换为行业名称
                    industry_name = self.code_to_industry.get(int(industry_code), f"未知行业{int(industry_code)}")
                    selected_by_industry[industry_name] += 1

        self.logger.debug(f"行业分布统计:")
        self.logger.debug(f"  总行业数: {len(industry_stocks)}")
        self.logger.debug(f"  覆盖行业数: {len(selected_by_industry)}")

        for industry, count in sorted(selected_by_industry.items(),
                                     key=lambda x: x[1],
                                     reverse=True):
            total_in_industry = len(industry_stocks.get(industry, []))
            self.logger.debug(f"    {industry}: {count} 只 (总共 {total_in_industry} 只)")

    def __repr__(self):
        """返回选股器的字符串表示"""
        return f"{self.name}(total={self.total_stocks}, strategy={self.strategy})"


# 便捷函数：创建行业中性化选股器
def create_industry_neutral_selector(total_stocks: int = 10,
                                     strategy: str = 'equal',
                                     **kwargs) -> IndustryNeutralSelector:
    """
    创建行业中性化选股器的便捷函数

    Args:
        total_stocks (int): 要选择的股票总数
        strategy (str): 分配策略
        **kwargs: 其他参数

    Returns:
        IndustryNeutralSelector: 配置好的选股器实例
    """
    return IndustryNeutralSelector(total_stocks=total_stocks, strategy=strategy, **kwargs)
