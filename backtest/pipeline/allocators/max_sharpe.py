# backtest/pipeline/allocators/max_sharpe.py

"""
最大夏普比率权重分配器

优化投资组合权重以最大化夏普比率（Sharpe Ratio）。

核心思想：
- 夏普比率 = (组合收益率 - 无风险利率) / 组合波动率
- 通过数值优化找到使夏普比率最大化的权重组合
- 约束条件：权重和为1，权重非负（long-only）

适用场景：
- 追求风险调整后收益最大化
- 理性投资组合构建
- 均值-方差优化框架
"""

import numpy as np
import logging
from typing import List, Any, Dict, Optional
from scipy.optimize import minimize
from .equal_weight import AllocatorBase
from backtest.utils.roller import BacktestDataRoller


class MaxSharpeAllocator(AllocatorBase):
    """
    最大夏普比率分配器

    使用均值-方差优化找到最大化夏普比率的权重组合。

    分配逻辑：
    1. 计算历史收益率和协方差矩阵
    2. 定义优化目标：最大化夏普比率（等价于最小化负夏普比率）
    3. 约束条件：权重和为1，权重非负
    4. 使用数值优化求解

    参数:
        lookback_period: 计算统计量的回看周期（默认120天）
        risk_free_rate: 年化无风险利率（默认0.02，即2%）
        min_periods: 最小有效数据点数（默认60天）
        max_weight: 单只股票的最大权重（默认0.3，即30%）
    """

    def __init__(self,
                 lookback_period: int = 120,
                 risk_free_rate: float = 0.02,
                 min_periods: int = 60,
                 max_weight: float = 0.3,
                 name: str = None):
        """
        初始化最大夏普比率分配器

        Args:
            lookback_period: 回看周期（天数）
            risk_free_rate: 年化无风险利率
            min_periods: 最小有效数据点数
            max_weight: 单只股票的最大权重
            name: 分配器名称
        """
        super().__init__(name or "MaxSharpeAllocator")
        self.lookback_period = lookback_period
        self.risk_free_rate = risk_free_rate
        self.min_periods = min_periods
        self.max_weight = max_weight
        self.roller = BacktestDataRoller()
        self.logger = logging.getLogger(__name__)

    def allocate(self, selected_stocks: List[Any]) -> Dict[Any, float]:
        """
        执行最大夏普比率权重分配

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

        # 6. 计算期望收益率和协方差矩阵
        mean_returns = returns.mean().values  # 日收益率
        cov_matrix = returns.cov().values

        # 7. 年化收益率和协方差矩阵（假设252个交易日）
        annual_returns = mean_returns * 252
        annual_cov = cov_matrix * 252

        # 8. 执行优化
        optimal_weights = self._optimize_sharpe(annual_returns, annual_cov, stock_count)

        # 9. 如果优化失败，回退到等权重
        if optimal_weights is None:
            self.logger.warning(
                f"分配器 [{self.name}] 优化失败，回退到等权重分配"
            )
            return self._fallback_equal_weight(selected_stocks)

        # 10. 构建权重字典
        weight_allocation = {}
        stock_names = returns.columns.tolist()

        for i, stock in enumerate(selected_stocks):
            stock_name = getattr(stock, '_name', f'Stock_{i}')

            # 找到对应的权重
            if stock_name in stock_names:
                idx = stock_names.index(stock_name)
                weight = float(optimal_weights[idx])
            else:
                self.logger.warning(f"无法找到股票 {stock_name} 的数据，分配最小权重")
                weight = 1.0 / stock_count

            weight_allocation[stock] = weight

        # 11. 重新归一化（确保权重总和为1）
        total_weight = sum(weight_allocation.values())
        if total_weight > 0:
            weight_allocation = {k: v/total_weight for k, v in weight_allocation.items()}

        # 12. 记录分配过程
        self.log_allocation_process(stock_count, weight_allocation)

        # 13. 计算并记录组合统计信息
        portfolio_return = np.dot(optimal_weights, annual_returns)
        portfolio_volatility = np.sqrt(np.dot(optimal_weights, np.dot(annual_cov, optimal_weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        self.logger.debug(f"  优化结果 - 年化收益: {portfolio_return:.4f}, "
                         f"年化波动: {portfolio_volatility:.4f}, "
                         f"夏普比率: {sharpe_ratio:.4f}")
        self.logger.debug(f"  权重范围: [{optimal_weights.min():.4f}, {optimal_weights.max():.4f}]")

        return weight_allocation

    def _optimize_sharpe(self,
                        expected_returns: np.ndarray,
                        cov_matrix: np.ndarray,
                        n_assets: int) -> Optional[np.ndarray]:
        """
        优化权重以最大化夏普比率

        Args:
            expected_returns: 期望收益率向量（年化）
            cov_matrix: 协方差矩阵（年化）
            n_assets: 资产数量

        Returns:
            np.ndarray: 最优权重向量，如果优化失败则返回 None
        """

        def neg_sharpe_ratio(weights):
            """负夏普比率（优化器最小化此函数）"""
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

            # 防止除零
            if portfolio_volatility == 0:
                return 1e10

            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            return -sharpe  # 最小化负值等于最大化正值

        # 约束条件：权重和为1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

        # 边界条件：每个权重在 [0, max_weight] 之间
        bounds = tuple((0, self.max_weight) for _ in range(n_assets))

        # 初始猜测：等权重
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        try:
            # 执行优化
            result = minimize(
                neg_sharpe_ratio,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )

            if result.success:
                return result.x
            else:
                self.logger.warning(f"优化未收敛: {result.message}")
                return None

        except Exception as e:
            self.logger.error(f"优化过程出错: {e}")
            return None

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
                f"risk_free_rate={self.risk_free_rate}, "
                f"max_weight={self.max_weight})")


def create_max_sharpe_allocator(lookback_period: int = 120,
                                risk_free_rate: float = 0.02,
                                min_periods: int = 60,
                                max_weight: float = 0.3,
                                name: str = None) -> MaxSharpeAllocator:
    """
    创建最大夏普比率分配器的便捷函数

    Args:
        lookback_period: 回看周期
        risk_free_rate: 无风险利率
        min_periods: 最小数据点数
        max_weight: 单只股票的最大权重
        name: 分配器名称

    Returns:
        MaxSharpeAllocator: 配置好的分配器实例
    """
    return MaxSharpeAllocator(
        lookback_period=lookback_period,
        risk_free_rate=risk_free_rate,
        min_periods=min_periods,
        max_weight=max_weight,
        name=name
    )
