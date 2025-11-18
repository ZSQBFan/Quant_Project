# strategy_configs.py

import logging
from typing import Any, Dict, List, Type

# 导入所有需要的组件
from core.factor_combiner import (BaseFactorCombiner, EqualWeightCombiner,
                                  FixedWeightCombiner,
                                  DynamicSignificanceCombiner,
                                  DynamicWeightCombiner, AICombiner)
from core.rolling_weight_calculator import (RollingCalculatorBase,
                                            RollingICIRCalculator,
                                            RollingRegressionCalculator,
                                            RollingAITrainer)
from core.ai_model_trainer import LightGBMTrainer


class StrategyConfig:
    """
    封装一个完整策略的所有配置组件。
    这个类是连接策略定义与策略执行的关键桥梁。
    """

    def __init__(self, combiner_class, combiner_kwargs, rolling_config):
        self.combiner_class = combiner_class
        self.combiner_kwargs = combiner_kwargs
        self.rolling_config = rolling_config

    def is_rolling(self) -> bool:
        return self.rolling_config is not None

    def create_rolling_calculator(
            self, forward_return_periods: List[int],
            factor_names: List[str]) -> RollingCalculatorBase:
        if not self.is_rolling(): return None

        calc_type = self.rolling_config.get('CALCULATOR_TYPE', 'ICIR')
        rolling_window = self.rolling_config.get('ROLLING_WINDOW_DAYS')
        # 【【【核心修改】】】: 获取调仓频率配置，默认为 'MS'
        rebalance_freq = self.rolling_config.get('REBALANCE_FREQUENCY', 'MS')

        if rolling_window is None:
            raise ValueError(
                "滚动策略必须在 rolling_config 中指定 'ROLLING_WINDOW_DAYS'。")

        # 根据类型创建不同的计算器实例，并传入所有需要的参数
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


# ==============================================================================
#                 --- 在这里定义你的所有策略配置 ---
# ==============================================================================

# --- 策略 1: 动态滚动 IC/IR 加权 (每日更新) ---
ROLLING_ICIR_STRATEGY = StrategyConfig(
    combiner_class=DynamicWeightCombiner,
    combiner_kwargs={'factor_weights': {}},
    rolling_config={
        'CALCULATOR_TYPE': 'ICIR',
        'ROLLING_WINDOW_DAYS': 90,
        'REBALANCE_FREQUENCY': 'D',  # <-- 【【【修改为每日更新】】】
        'FACTOR_WEIGHTING_CONFIG': {
            'Momentum': 'ic_mean_30d',
            'Reversal20D': 'ir_30d',
            'IndNeu_Momentum': 'ir_60d',
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

# --- 策略 5: 动态滚动回归加权 (每日更新) ---
ROLLING_REGRESSION_STRATEGY = StrategyConfig(
    combiner_class=DynamicWeightCombiner,
    combiner_kwargs={'factor_weights': {}},
    rolling_config={
        'CALCULATOR_TYPE': 'Regression',
        'ROLLING_WINDOW_DAYS': 90,
        'REBALANCE_FREQUENCY': 'D',  # <-- 【【【修改为每日更新】】】
        'TARGET_RETURN_PERIOD': 30
    })

# --- 策略 6: AI 动态周期性重训练 (每月更新) ---
AI_PERIODIC_RETRAIN_STRATEGY = StrategyConfig(
    combiner_class=AICombiner,
    combiner_kwargs={'initial_model_path': None},
    rolling_config={
        'CALCULATOR_TYPE':
        'AI_Trainer',
        'ROLLING_WINDOW_DAYS':
        250,
        'REBALANCE_FREQUENCY':
        'MS',  # <-- 【【【保持每月更新】】】
        'TRAINING_CALCULATOR':
        LightGBMTrainer(target_return_period=30,
                        model_params={
                            'objective': 'regression_l1',
                            'metric': 'rmse',
                            'n_estimators': 200,
                            'learning_rate': 0.05,
                            'num_leaves': 31,
                            'max_depth': -1,
                            'min_child_samples': 20,
                            'subsample': 0.8,
                            'colsample_bytree': 0.8,
                            'random_state': 42,
                            'n_jobs': -1,
                            'verbose': -1,
                        })
    })

# ==============================================================================
#                 --- 在这里注册您的策略 ---
# ==============================================================================
STRATEGY_REGISTRY = {
    "RollingICIR_Daily": ROLLING_ICIR_STRATEGY,
    "FixedWeights": FIXED_WEIGHT_STRATEGY,
    "EqualWeights": EQUAL_WEIGHT_STRATEGY,
    "DynamicSignificance": DYNAMIC_SIG_STRATEGY,
    "RollingRegression_Daily": ROLLING_REGRESSION_STRATEGY,
    "AI_Periodic_Retrain": AI_PERIODIC_RETRAIN_STRATEGY,
}
