# core/strategy.py

from typing import Any, Dict, List, Type
from .abstractions import BaseFactorCombiner, RollingCalculatorBase


class StrategyConfig:
    """
    【框架组件】: 策略配置容器
    
    这个类是一个容器，它持有策略所需的所有信息，例如使用哪个
    因子合成器 (combiner) 以及（如果是滚动策略）如何配置滚动逻辑。
    `main_analyzer.py` 与这个类交互，而不是与具体的策略实现细节交互，
    从而实现了核心逻辑与策略定义的解耦。
    """

    def __init__(self, combiner_class: Type[BaseFactorCombiner],
                 combiner_kwargs: Dict[str, Any], rolling_config: Dict[str,
                                                                       Any]):
        self.combiner_class = combiner_class
        self.combiner_kwargs = combiner_kwargs
        self.rolling_config = rolling_config

    def is_rolling(self) -> bool:
        """检查此配置是否为【滚动策略】。"""
        return self.rolling_config is not None

    def create_rolling_calculator(
            self, forward_return_periods: List[int],
            factor_names: List[str]) -> RollingCalculatorBase:
        """
        【【【核心工厂方法】】】
        
        根据 'CALCULATOR_TYPE' 创建【对应】的滚动计算器实例。
        这个方法是连接策略定义与策略执行的关键桥梁。
        """
        if not self.is_rolling(): return None

        calc_type = self.rolling_config.get('CALCULATOR_TYPE', 'ICIR')
        rolling_window = self.rolling_config.get('ROLLING_WINDOW_DAYS')
        rebalance_freq = self.rolling_config.get('REBALANCE_FREQUENCY', 'MS')

        if rolling_window is None:
            raise ValueError(
                "滚动策略必须在 rolling_config 中指定 'ROLLING_WINDOW_DAYS'。")

        # 注意：这里我们不能直接 import 具体的实现类，因为 core 是框架，不应依赖于 strategies。
        # 具体的实例化逻辑将在 strategy_configs.py 中处理，这里只是一个示例结构。
        # 为了让代码能运行，我们将在这里进行实例化，但在一个更严格的框架中，
        # 可能会使用依赖注入或插件系统。

        from strategies.rolling_calculators import RollingICIRCalculator, RollingRegressionCalculator, RollingAITrainer

        if calc_type == 'ICIR':
            config = self.rolling_config.get('FACTOR_WEIGHTING_CONFIG', {})
            return RollingICIRCalculator(
                factor_weight_config=config,
                forward_return_periods=forward_return_periods,
                factor_names=factor_names,
                rolling_window_days=rolling_window,
                rebalance_frequency=rebalance_freq)

        elif calc_type == 'Regression':
            target_period = self.rolling_config.get('TARGET_RETURN_PERIOD')
            if target_period not in forward_return_periods:
                raise ValueError(
                    f"目标周期 {target_period}d 不在 main_analyzer.py 定义的 {forward_return_periods} 中。"
                )
            return RollingRegressionCalculator(
                target_return_period=target_period,
                factor_names=factor_names,
                rolling_window_days=rolling_window,
                rebalance_frequency=rebalance_freq)

        elif calc_type == 'AI_Trainer':
            trainer = self.rolling_config.get('TRAINING_CALCULATOR')
            return RollingAITrainer(training_calculator=trainer,
                                    factor_names=factor_names,
                                    rolling_window_days=rolling_window,
                                    rebalance_frequency=rebalance_freq)

        else:
            raise ValueError(f"未知的滚动计算器类型: '{calc_type}'")
