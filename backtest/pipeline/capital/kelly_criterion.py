# backtest/pipeline/capital/kelly_criterion.py

"""
凯利公式仓位管理器

凯利公式（Kelly Criterion）是一种动态仓位管理方法，通过最大化长期资本增长率来确定最优投资比例。

核心公式：
- Full Kelly: f* = (μ - r) / σ²
  其中：
  - f* 是最优仓位比例
  - μ 是资产的期望收益率
  - r 是无风险利率
  - σ² 是收益率的方差

凯利公式的变体：
1. Full Kelly (f*): 理论上最优，但实际中波动较大
2. Half Kelly (f*/2): 在保持较高收益的同时显著降低波动
3. Quarter Kelly (f*/4): 更加保守，适合风险厌恶型投资者

优点：
- 理论上最优：最大化长期资本增长率
- 自适应：根据市场状况动态调整仓位
- 防止破产：理论上永远不会导致完全亏损

缺点：
- 波动较大：Full Kelly可能导致较大回撤
- 参数敏感：对期望收益和方差的估计非常敏感
- 需要历史数据：依赖足够长的历史数据来估计参数

适用场景：
- 趋势明确的市场：期望收益率显著为正
- 有充分历史数据的资产
- 能够承受一定波动的投资者
"""

import numpy as np
import logging
from typing import Optional
from .full_position import CapitalManagerBase
from backtest.utils.roller import BacktestDataRoller


class KellyCriterionManager(CapitalManagerBase):
    """
    凯利公式仓位管理器基类

    提供基于凯利公式的动态仓位管理功能。

    分配逻辑：
    1. 获取历史收益率数据
    2. 计算期望收益率 μ 和方差 σ²
    3. 应用凯利公式计算最优仓位比例 f* = (μ - r) / σ²
    4. 根据具体策略调整仓位（Full/Half/Quarter Kelly）
    5. 应用仓位上下限约束

    参数:
        lookback_period: 计算统计量的回看周期（默认120天）
        risk_free_rate: 年化无风险利率（默认0.02，即2%）
        min_periods: 最小有效数据点数（默认60天）
        kelly_fraction: 凯利分数（1.0=Full Kelly, 0.5=Half Kelly, 0.25=Quarter Kelly）
        max_position: 最大仓位比例（默认0.95，即95%）
        min_position: 最小仓位比例（默认0.0，即0%）
    """

    def __init__(self,
                 lookback_period: int = 120,
                 risk_free_rate: float = 0.02,
                 min_periods: int = 60,
                 kelly_fraction: float = 1.0,
                 max_position: float = 0.95,
                 min_position: float = 0.0,
                 name: str = None):
        """
        初始化凯利公式仓位管理器

        Args:
            lookback_period: 回看周期（天数）
            risk_free_rate: 年化无风险利率
            min_periods: 最小有效数据点数
            kelly_fraction: 凯利分数（1.0/0.5/0.25）
            max_position: 最大仓位比例
            min_position: 最小仓位比例
            name: 管理器名称
        """
        super().__init__(name or f"KellyCriterion({kelly_fraction:.2f})")

        # 参数验证
        if lookback_period <= 0:
            raise ValueError(f"回看周期必须大于0，实际为: {lookback_period}")
        if min_periods <= 0 or min_periods > lookback_period:
            raise ValueError(f"最小有效数据点数必须在 (0, {lookback_period}] 范围内，实际为: {min_periods}")
        if kelly_fraction <= 0:
            raise ValueError(f"凯利分数必须大于0，实际为: {kelly_fraction}")
        if not (0 <= min_position <= max_position <= 1):
            raise ValueError(f"仓位范围必须满足 0 <= min_position <= max_position <= 1")

        self.lookback_period = lookback_period
        self.risk_free_rate = risk_free_rate
        self.min_periods = min_periods
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.min_position = min_position

        self.roller = BacktestDataRoller()
        self.logger = logging.getLogger(__name__)

        # 记录初始化信息
        self.logger.info(f"资金管理器 [{self.name}] 初始化完成")
        self.logger.info(f"  凯利分数: {self.kelly_fraction:.4f}")
        self.logger.info(f"  回看周期: {self.lookback_period} 天")
        self.logger.info(f"  仓位范围: [{self.min_position:.2%}, {self.max_position:.2%}]")

    def get_allocation(self, total_value: float, data_objects: list = None) -> float:
        """
        计算可用于交易的资金量

        Args:
            total_value: 账户总价值
            data_objects: backtrader数据对象列表（用于计算组合收益率和方差）

        Returns:
            float: 可用于交易的资金量
        """
        # 验证总价值
        is_valid, error_msg = self.validate_total_value(total_value)
        if not is_valid:
            self.logger.error(f"资金管理器 [{self.name}] 总价值验证失败: {error_msg}")
            raise ValueError(f"总价值验证失败: {error_msg}")

        # 如果没有提供数据对象，回退到最大仓位
        if not data_objects:
            self.logger.warning(f"资金管理器 [{self.name}] 未提供数据对象，使用最大仓位 {self.max_position:.2%}")
            allocation = total_value * self.max_position
            self.log_allocation_process(total_value, allocation, self.max_position,
                                       extra_info="缺少数据对象，使用最大仓位")
            return allocation

        # 计算凯利仓位比例
        kelly_position = self._calculate_kelly_position(data_objects)

        # 如果计算失败，回退到最大仓位
        if kelly_position is None:
            self.logger.warning(f"资金管理器 [{self.name}] 凯利仓位计算失败，使用最大仓位")
            kelly_position = self.max_position

        # 应用约束
        constrained_position = self._apply_constraints(kelly_position)

        # 计算分配金额
        allocation = total_value * constrained_position

        # 记录分配过程
        extra_info = f"凯利仓位: {kelly_position:.4f}, 约束后仓位: {constrained_position:.4f}"
        self.log_allocation_process(total_value, allocation, constrained_position, extra_info)

        return allocation

    def _calculate_kelly_position(self, data_objects: list) -> Optional[float]:
        """
        计算凯利仓位比例

        Args:
            data_objects: backtrader数据对象列表

        Returns:
            Optional[float]: 凯利仓位比例，如果计算失败则返回None
        """
        # 获取组合的历史收益率
        returns = self.roller.get_returns(data_objects, self.lookback_period)

        # 检查数据是否足够
        if returns is None or len(returns) < self.min_periods:
            self.logger.warning(
                f"资金管理器 [{self.name}] 历史数据不足: "
                f"需要至少 {self.min_periods} 天，实际 {len(returns) if returns is not None else 0} 天"
            )
            return None

        # 计算组合的平均收益率（假设等权重组合）
        # 这里简化处理：计算所有股票的平均收益率
        portfolio_returns = returns.mean(axis=1)  # 每天的平均收益率

        # 计算统计量
        mean_return = portfolio_returns.mean()  # 日平均收益率
        variance = portfolio_returns.var()      # 日方差

        # 年化
        annual_return = mean_return * 252
        annual_variance = variance * 252

        # 防止除零
        if annual_variance <= 0 or np.isnan(annual_variance) or np.isinf(annual_variance):
            self.logger.warning(f"资金管理器 [{self.name}] 方差无效: {annual_variance}")
            return None

        # 应用凯利公式: f* = (μ - r) / σ²
        kelly_full = (annual_return - self.risk_free_rate) / annual_variance

        # 应用凯利分数
        kelly_position = kelly_full * self.kelly_fraction

        # 记录计算过程
        self.logger.debug(f"  年化收益率: {annual_return:.4f} ({annual_return*100:.2f}%)")
        self.logger.debug(f"  年化方差: {annual_variance:.4f}")
        self.logger.debug(f"  年化波动率: {np.sqrt(annual_variance):.4f} ({np.sqrt(annual_variance)*100:.2f}%)")
        self.logger.debug(f"  Full Kelly: {kelly_full:.4f} ({kelly_full*100:.2f}%)")
        self.logger.debug(f"  {self.kelly_fraction:.2f} Kelly: {kelly_position:.4f} ({kelly_position*100:.2f}%)")

        return kelly_position

    def _apply_constraints(self, position: float) -> float:
        """
        应用仓位约束

        Args:
            position: 原始仓位比例

        Returns:
            float: 约束后的仓位比例
        """
        # 处理异常值
        if position is None or np.isnan(position) or np.isinf(position):
            self.logger.warning(f"资金管理器 [{self.name}] 仓位值异常: {position}，使用最小仓位")
            return self.min_position

        # 应用上下限
        constrained = max(self.min_position, min(position, self.max_position))

        # 如果被约束，记录日志
        if constrained != position:
            if constrained == self.max_position:
                self.logger.debug(f"  仓位被上限约束: {position:.4f} -> {constrained:.4f}")
            elif constrained == self.min_position:
                self.logger.debug(f"  仓位被下限约束: {position:.4f} -> {constrained:.4f}")

        # 如果仓位为负，特别提示
        if position < 0:
            self.logger.warning(
                f"资金管理器 [{self.name}] 计算出负仓位 {position:.4f}，"
                f"这表明期望收益率低于无风险利率，建议不持仓"
            )

        return constrained

    def __repr__(self):
        """返回管理器的字符串表示"""
        return (f"{self.name}(kelly_fraction={self.kelly_fraction:.2f}, "
                f"lookback={self.lookback_period}, "
                f"position_range=[{self.min_position:.2f}, {self.max_position:.2f}])")


class FullKellyManager(KellyCriterionManager):
    """
    Full Kelly 仓位管理器

    使用完整的凯利公式进行仓位管理（kelly_fraction=1.0）。

    特点：
    - 理论上最优：最大化长期资本增长率
    - 波动较大：可能导致较大的回撤
    - 风险较高：对参数估计误差敏感

    适用场景：
    - 对自己的预测模型有很高信心
    - 能够承受较大波动
    - 追求长期收益最大化
    """

    def __init__(self,
                 lookback_period: int = 120,
                 risk_free_rate: float = 0.02,
                 min_periods: int = 60,
                 max_position: float = 0.95,
                 min_position: float = 0.0,
                 name: str = None):
        """
        初始化 Full Kelly 管理器

        Args:
            lookback_period: 回看周期（天数）
            risk_free_rate: 年化无风险利率
            min_periods: 最小有效数据点数
            max_position: 最大仓位比例
            min_position: 最小仓位比例
            name: 管理器名称
        """
        super().__init__(
            lookback_period=lookback_period,
            risk_free_rate=risk_free_rate,
            min_periods=min_periods,
            kelly_fraction=1.0,  # Full Kelly
            max_position=max_position,
            min_position=min_position,
            name=name or "FullKellyManager"
        )


class HalfKellyManager(KellyCriterionManager):
    """
    Half Kelly 仓位管理器

    使用半凯利公式进行仓位管理（kelly_fraction=0.5）。

    特点：
    - 平衡收益与风险：在保持较高收益的同时降低波动
    - 更加稳健：对参数估计误差的容忍度更高
    - 实践中更常用：许多专业投资者的选择

    优势：
    - 收益率仅略低于Full Kelly
    - 波动率显著降低（理论上降低约25%）
    - 最大回撤更小

    适用场景：
    - 追求收益与风险的平衡
    - 中等风险承受能力
    - 实际应用中的推荐选择
    """

    def __init__(self,
                 lookback_period: int = 120,
                 risk_free_rate: float = 0.02,
                 min_periods: int = 60,
                 max_position: float = 0.95,
                 min_position: float = 0.0,
                 name: str = None):
        """
        初始化 Half Kelly 管理器

        Args:
            lookback_period: 回看周期（天数）
            risk_free_rate: 年化无风险利率
            min_periods: 最小有效数据点数
            max_position: 最大仓位比例
            min_position: 最小仓位比例
            name: 管理器名称
        """
        super().__init__(
            lookback_period=lookback_period,
            risk_free_rate=risk_free_rate,
            min_periods=min_periods,
            kelly_fraction=0.5,  # Half Kelly
            max_position=max_position,
            min_position=min_position,
            name=name or "HalfKellyManager"
        )


class QuarterKellyManager(KellyCriterionManager):
    """
    Quarter Kelly 仓位管理器

    使用四分之一凯利公式进行仓位管理（kelly_fraction=0.25）。

    特点：
    - 非常保守：显著降低波动和回撤
    - 稳定性高：对参数估计误差非常不敏感
    - 收益率较低：牺牲部分收益换取稳定性

    优势：
    - 最大回撤很小
    - 波动率最低
    - 适合风险厌恶型投资者

    适用场景：
    - 风险厌恶型投资者
    - 资金规模较大，追求稳定
    - 对回撤有严格限制
    - 初学者或保守策略
    """

    def __init__(self,
                 lookback_period: int = 120,
                 risk_free_rate: float = 0.02,
                 min_periods: int = 60,
                 max_position: float = 0.95,
                 min_position: float = 0.0,
                 name: str = None):
        """
        初始化 Quarter Kelly 管理器

        Args:
            lookback_period: 回看周期（天数）
            risk_free_rate: 年化无风险利率
            min_periods: 最小有效数据点数
            max_position: 最大仓位比例
            min_position: 最小仓位比例
            name: 管理器名称
        """
        super().__init__(
            lookback_period=lookback_period,
            risk_free_rate=risk_free_rate,
            min_periods=min_periods,
            kelly_fraction=0.25,  # Quarter Kelly
            max_position=max_position,
            min_position=min_position,
            name=name or "QuarterKellyManager"
        )


# 便捷函数
def create_full_kelly_manager(lookback_period: int = 120,
                              risk_free_rate: float = 0.02,
                              min_periods: int = 60,
                              max_position: float = 0.95,
                              min_position: float = 0.0,
                              name: str = None) -> FullKellyManager:
    """创建 Full Kelly 管理器的便捷函数"""
    return FullKellyManager(
        lookback_period=lookback_period,
        risk_free_rate=risk_free_rate,
        min_periods=min_periods,
        max_position=max_position,
        min_position=min_position,
        name=name
    )


def create_half_kelly_manager(lookback_period: int = 120,
                              risk_free_rate: float = 0.02,
                              min_periods: int = 60,
                              max_position: float = 0.95,
                              min_position: float = 0.0,
                              name: str = None) -> HalfKellyManager:
    """创建 Half Kelly 管理器的便捷函数"""
    return HalfKellyManager(
        lookback_period=lookback_period,
        risk_free_rate=risk_free_rate,
        min_periods=min_periods,
        max_position=max_position,
        min_position=min_position,
        name=name
    )


def create_quarter_kelly_manager(lookback_period: int = 120,
                                 risk_free_rate: float = 0.02,
                                 min_periods: int = 60,
                                 max_position: float = 0.95,
                                 min_position: float = 0.0,
                                 name: str = None) -> QuarterKellyManager:
    """创建 Quarter Kelly 管理器的便捷函数"""
    return QuarterKellyManager(
        lookback_period=lookback_period,
        risk_free_rate=risk_free_rate,
        min_periods=min_periods,
        max_position=max_position,
        min_position=min_position,
        name=name
    )
