# strategy_configs.py

import logging
from core.factor_combiner import (EqualWeightCombiner, FixedWeightCombiner,
                                  DynamicSignificanceCombiner,
                                  DynamicWeightCombiner)
from core.rolling_weight_calculator import (RollingICIRCalculator,
                                            RollingRegressionCalculator)


class StrategyConfig:
    """封装一个完整策略的所有配置组件。"""

    def __init__(self, combiner_class, combiner_kwargs, rolling_config):
        self.combiner_class = combiner_class
        self.combiner_kwargs = combiner_kwargs
        self.rolling_config = rolling_config

    def is_rolling(self):
        return self.rolling_config is not None

    def create_rolling_calculator(self, forward_return_periods, factor_names):
        if not self.is_rolling():
            return None

        calc_type = self.rolling_config.get('CALCULATOR_TYPE', 'ICIR')
        rolling_window = self.rolling_config.get('ROLLING_WINDOW_DAYS')
        if rolling_window is None:
            raise ValueError(
                "滚动策略必须在 rolling_config 中指定 'ROLLING_WINDOW_DAYS'。")

        if calc_type == 'ICIR':
            config = self.rolling_config.get('FACTOR_WEIGHTING_CONFIG', {})
            return RollingICIRCalculator(
                factor_weight_config=config,
                forward_return_periods=forward_return_periods,
                factor_names=factor_names,
                rolling_window_days=rolling_window)
        elif calc_type == 'Regression':
            target_period = self.rolling_config.get('TARGET_RETURN_PERIOD')
            if target_period not in forward_return_periods:
                raise ValueError(
                    f"目标周期 {target_period}d 不在 main_analyzer.py 定义的 {forward_return_periods} 中。"
                )
            return RollingRegressionCalculator(
                target_return_period=target_period,
                factor_names=factor_names,
                rolling_window_days=rolling_window)
        else:
            raise ValueError(f"未知的滚动计算器类型: {calc_type}")


# --- 策略 1: 动态滚动 ICIR 加权 ---
ROLLING_ICIR_STRATEGY = StrategyConfig(combiner_class=DynamicWeightCombiner,
                                       combiner_kwargs={'factor_weights': {}},
                                       rolling_config={
                                           'CALCULATOR_TYPE': 'ICIR',
                                           'ROLLING_WINDOW_DAYS': 90,
                                           'FACTOR_WEIGHTING_CONFIG': {
                                               'Momentum': 'ic_mean_30d',
                                               'Reversal20D': 'ir_30d',
                                               'IndNeu_Momentum': 'ir_60d',
                                               'IndNeu_Reversal20D': 'ir_30d',
                                               'IndNeu_VolumeCV': 'ic_30d',
                                           }
                                       })

# --- 策略 2: 静态固定权重 ---
FIXED_WEIGHT_STRATEGY = StrategyConfig(combiner_class=FixedWeightCombiner,
                                       combiner_kwargs={
                                           'factor_weights': {
                                               'BollingerBands': 0.3,
                                               'Momentum': 0.7
                                           }
                                       },
                                       rolling_config=None)

# --- 策略 3: 静态等权重 ---
EQUAL_WEIGHT_STRATEGY = StrategyConfig(combiner_class=EqualWeightCombiner,
                                       combiner_kwargs={},
                                       rolling_config=None)

# --- 策略 4: 动态显著性加权 ---
DYNAMIC_SIG_STRATEGY = StrategyConfig(
    combiner_class=DynamicSignificanceCombiner,
    combiner_kwargs={},
    rolling_config=None)

# --- 策略 5: 动态滚动回归加权 ---
ROLLING_REGRESSION_STRATEGY = StrategyConfig(
    combiner_class=DynamicWeightCombiner,
    combiner_kwargs={'factor_weights': {}},
    rolling_config={
        'CALCULATOR_TYPE': 'Regression',
        'ROLLING_WINDOW_DAYS': 90,  # <-- 【【【修改】】】 回看窗口
        'TARGET_RETURN_PERIOD': 30
    })

# ==============================================================================
#                 --- 【【【用户配置区 2】】】 ---
#                 ---    在这里注册您的策略    ---
# ==============================================================================
#
# main_analyzer.py 将从这个字典中，根据您选择的
# 【字符串名称】 (例如 "RollingICIR") 来查找对应的策略配置对象。
#
# 【【如何添加新策略】】
# 1. 在上方 "用户配置区 1" 定义您的 `MY_NEW_STRATEGY = StrategyConfig(...)`
# 2. 在下方这个字典中添加一行: `"MyNewStrategyName": MY_NEW_STRATEGY`

STRATEGY_REGISTRY = {
    # 策略名称 (str) : 策略配置对象 (StrategyConfig)
    "RollingICIR": ROLLING_ICIR_STRATEGY,
    "FixedWeights": FIXED_WEIGHT_STRATEGY,
    "EqualWeights": EQUAL_WEIGHT_STRATEGY,
    "DynamicSignificance": DYNAMIC_SIG_STRATEGY,
    "RollingRegression": ROLLING_REGRESSION_STRATEGY,
}
