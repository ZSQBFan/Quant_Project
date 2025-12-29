# backtest/pipeline/allocators/risk_parity.py

"""
风险平价权重分配器

使用风险平价（Risk Parity）方法分配权重，使得每个资产对组合风险的贡献相等。

核心思想：
- 高波动率的资产分配较低权重
- 低波动率的资产分配较高权重
- 权重与波动率成反比（逆波动率加权）

适用场景：
- 风险分散化投资
- 追求稳定收益的策略
- 波动率差异较大的资产组合
"""

import numpy as np
import logging
from typing import List, Any, Dict
from .equal_weight import AllocatorBase
from backtest.utils.roller import BacktestDataRoller


class RiskParityAllocator(AllocatorBase):
    """
    风险平价分配器

    使用逆波动率加权方法分配权重，使每个资产的风险贡献相等。

    分配逻辑：
    1. 计算每只股票的历史波动率
    2. 权重与波动率成反比
    3. 归一化使权重总和为 1.0

    公式：
        weight_i = (1/volatility_i) / sum(1/volatility_j)

    参数:
        lookback_period: 计算波动率的回看周期（默认60天）
        min_periods: 最小有效数据点数（默认30天）
        volatility_floor: 波动率下限，防止除零（默认0.0001）
    """

    def __init__(self,
                 lookback_period: int = 60,
                 min_periods: int = 30,
                 volatility_floor: float = 0.0001,
                 name: str = None):
        """
        初始化风险平价分配器

        Args:
            lookback_period: 计算波动率的回看周期（天数）
            min_periods: 最小有效数据点数
            volatility_floor: 波动率下限
            name: 分配器名称
        """
        super().__init__(name or "RiskParityAllocator")
        self.lookback_period = lookback_period
        self.min_periods = min_periods
        self.volatility_floor = volatility_floor
        self.roller = BacktestDataRoller()
        self.logger = logging.getLogger(__name__)

    def allocate(self, selected_stocks: List[Any]) -> Dict[Any, float]:
        """
        执行风险平价权重分配

        Args:
            selected_stocks: 选中的股票数据对象列表

        Returns:
            Dict[Any, float]: {股票数据对象: 权重} 字典
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
            stock = selected_stocks[0]
            stock_name = getattr(stock, '_name', 'Single_Stock')
            self.logger.debug(f"单股票分配: {stock_name} -> 1.0")
            return {stock: 1.0}

        # 4. 获取历史收益率数据
        returns = self.roller.get_returns(selected_stocks, self.lookback_period)

        # 5. 如果无法获取足够的历史数据，回退到等权重
        if returns is None or len(returns) < self.min_periods:
            self.logger.warning(
                f"分配器 [{self.name}] 历史数据不足，回退到等权重分配"
            )
            return self._fallback_equal_weight(selected_stocks)

        # 6. 计算每只股票的波动率（收益率标准差）
        volatilities = returns.std().values

        # 7. 应用波动率下限，防止除零或过大权重
        volatilities = np.maximum(volatilities, self.volatility_floor)

        # 8. 计算逆波动率权重
        inverse_vol = 1.0 / volatilities
        weights_array = inverse_vol / inverse_vol.sum()

        # 9. 构建权重字典
        weight_allocation = {}
        stock_names = returns.columns.tolist()

        for i, stock in enumerate(selected_stocks):
            stock_name = getattr(stock, '_name', f'Stock_{i}')

            # 找到对应的权重
            if stock_name in stock_names:
                idx = stock_names.index(stock_name)
                weight = float(weights_array[idx])
            else:
                # 如果找不到对应数据，分配最小权重
                self.logger.warning(f"无法找到股票 {stock_name} 的数据，分配最小权重")
                weight = 1.0 / stock_count

            weight_allocation[stock] = weight

        # 10. 重新归一化（确保权重总和为1）
        total_weight = sum(weight_allocation.values())
        if total_weight > 0:
            weight_allocation = {k: v/total_weight for k, v in weight_allocation.items()}

        # 11. 记录分配过程
        self.log_allocation_process(stock_count, weight_allocation)

        # 12. 记录波动率信息
        self.logger.debug(f"  波动率范围: [{volatilities.min():.4f}, {volatilities.max():.4f}]")
        self.logger.debug(f"  权重范围: [{weights_array.min():.4f}, {weights_array.max():.4f}]")

        return weight_allocation

    def _fallback_equal_weight(self, selected_stocks: List[Any]) -> Dict[Any, float]:
        """
        回退到等权重分配

        Args:
            selected_stocks: 股票列表

        Returns:
            Dict[Any, float]: 等权重字典
        """
        equal_weight = 1.0 / len(selected_stocks)
        return {stock: equal_weight for stock in selected_stocks}

    def __repr__(self):
        """返回分配器的字符串表示"""
        return (f"{self.name}(lookback={self.lookback_period}, "
                f"min_periods={self.min_periods})")


def create_risk_parity_allocator(lookback_period: int = 60,
                                 min_periods: int = 30,
                                 volatility_floor: float = 0.0001,
                                 name: str = None) -> RiskParityAllocator:
    """
    创建风险平价分配器的便捷函数

    Args:
        lookback_period: 回看周期
        min_periods: 最小数据点数
        volatility_floor: 波动率下限
        name: 分配器名称

    Returns:
        RiskParityAllocator: 配置好的分配器实例
    """
    return RiskParityAllocator(
        lookback_period=lookback_period,
        min_periods=min_periods,
        volatility_floor=volatility_floor,
        name=name
    )
