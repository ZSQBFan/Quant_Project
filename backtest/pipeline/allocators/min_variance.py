# backtest/pipeline/allocators/min_variance.py

"""
全局最小方差权重分配器 (Global Minimum Variance, GMV)

通过优化找到使组合方差最小的权重分配，充分利用股票间的相关性进行风险对冲。

核心思想：
- 利用股票间的相关性来降低组合风险
- 负相关的股票可以互相对冲，降低整体波动
- 不仅考虑单个股票的波动率，更关注股票之间的协方差

适用场景：
- 追求低波动、稳健收益
- 股票池中存在相关性较低或负相关的股票
- 希望通过分散化降低系统性风险
- 对冲高相关性资产的集中风险
"""

import numpy as np
import logging
from typing import List, Any, Dict, Optional
from scipy.optimize import minimize
from .equal_weight import AllocatorBase
from backtest.utils.roller import BacktestDataRoller


class MinimumVarianceAllocator(AllocatorBase):
    """
    全局最小方差分配器

    通过求解二次规划问题找到使组合方差最小的权重组合。

    优化问题：
        minimize: w^T * Σ * w  (组合方差)
        subject to:
            - sum(w) = 1  (权重和为1)
            - w >= 0      (权重非负，long-only)
            - w <= max_weight  (单只股票最大权重限制)

    分配逻辑：
    1. 计算股票收益率的协方差矩阵
    2. 定义优化目标：最小化组合方差
    3. 约束条件：权重和为1，权重非负，单只股票不超过最大权重
    4. 使用数值优化求解

    参数:
        lookback_period: 计算协方差矩阵的回看周期（默认120天）
        min_periods: 最小有效数据点数（默认60天）
        max_weight: 单只股票的最大权重（默认0.3，即30%）
        regularization: L2正则化系数，用于协方差矩阵的稳定性（默认0.001）
    """

    def __init__(self,
                 lookback_period: int = 120,
                 min_periods: int = 60,
                 max_weight: float = 0.3,
                 regularization: float = 0.001,
                 name: str = None):
        """
        初始化全局最小方差分配器

        Args:
            lookback_period: 回看周期（天数）
            min_periods: 最小有效数据点数
            max_weight: 单只股票的最大权重
            regularization: L2正则化系数
            name: 分配器名称
        """
        super().__init__(name or "MinimumVarianceAllocator")
        self.lookback_period = lookback_period
        self.min_periods = min_periods
        self.max_weight = max_weight
        self.regularization = regularization
        self.roller = BacktestDataRoller()
        self.logger = logging.getLogger(__name__)

    def allocate(self, selected_stocks: List[Any]) -> Dict[Any, float]:
        """
        执行全局最小方差权重分配

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

        # 4. 获取协方差矩阵
        cov_matrix = self.roller.get_covariance_matrix(selected_stocks, self.lookback_period)

        # 5. 如果无法获取足够的历史数据，回退到等权重
        if cov_matrix is None:
            self.logger.warning(
                f"分配器 [{self.name}] 无法计算协方差矩阵，回退到等权重分配"
            )
            return self._fallback_equal_weight(selected_stocks)

        # 6. 添加正则化以提高数值稳定性（岭回归）
        # 协方差矩阵可能奇异或病态，添加小的对角项改善条件数
        cov_matrix_reg = cov_matrix + self.regularization * np.eye(stock_count)

        # 7. 执行优化
        optimal_weights = self._optimize_min_variance(cov_matrix_reg, stock_count)

        # 8. 如果优化失败，回退到等权重
        if optimal_weights is None:
            self.logger.warning(
                f"分配器 [{self.name}] 优化失败，回退到等权重分配"
            )
            return self._fallback_equal_weight(selected_stocks)

        # 9. 构建权重字典
        weight_allocation = {}
        for i, stock in enumerate(selected_stocks):
            weight = float(optimal_weights[i])
            weight_allocation[stock] = weight

        # 10. 重新归一化（确保权重总和为1）
        total_weight = sum(weight_allocation.values())
        if total_weight > 0:
            weight_allocation = {k: v/total_weight for k, v in weight_allocation.items()}

        # 11. 记录分配过程
        self.log_allocation_process(stock_count, weight_allocation)

        # 12. 计算并记录组合统计信息
        portfolio_variance = np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))
        portfolio_volatility = np.sqrt(portfolio_variance) * np.sqrt(252)  # 年化波动率

        self.logger.debug(f"  优化结果 - 组合年化波动率: {portfolio_volatility:.4f}")
        self.logger.debug(f"  权重范围: [{optimal_weights.min():.4f}, {optimal_weights.max():.4f}]")

        # 计算权重集中度（基尼系数）
        sorted_weights = np.sort(optimal_weights)
        n = len(sorted_weights)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_weights) / (n * np.sum(sorted_weights))) - (n + 1) / n
        self.logger.debug(f"  权重集中度(Gini): {gini:.4f}")

        return weight_allocation

    def _optimize_min_variance(self,
                              cov_matrix: np.ndarray,
                              n_assets: int) -> Optional[np.ndarray]:
        """
        优化权重以最小化组合方差

        Args:
            cov_matrix: 协方差矩阵
            n_assets: 资产数量

        Returns:
            np.ndarray: 最优权重向量，如果优化失败则返回 None
        """

        def portfolio_variance(weights):
            """组合方差（优化器最小化此函数）"""
            return np.dot(weights, np.dot(cov_matrix, weights))

        # 约束条件：权重和为1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

        # 边界条件：每个权重在 [0, max_weight] 之间
        bounds = tuple((0, self.max_weight) for _ in range(n_assets))

        # 初始猜测：等权重
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        try:
            # 执行优化
            result = minimize(
                portfolio_variance,
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
                f"max_weight={self.max_weight}, "
                f"regularization={self.regularization})")


def create_min_variance_allocator(lookback_period: int = 120,
                                  min_periods: int = 60,
                                  max_weight: float = 0.3,
                                  regularization: float = 0.001,
                                  name: str = None) -> MinimumVarianceAllocator:
    """
    创建全局最小方差分配器的便捷函数

    Args:
        lookback_period: 回看周期
        min_periods: 最小数据点数
        max_weight: 单只股票的最大权重
        regularization: L2正则化系数
        name: 分配器名称

    Returns:
        MinimumVarianceAllocator: 配置好的分配器实例
    """
    return MinimumVarianceAllocator(
        lookback_period=lookback_period,
        min_periods=min_periods,
        max_weight=max_weight,
        regularization=regularization,
        name=name
    )
