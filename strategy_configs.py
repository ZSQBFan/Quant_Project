# strategy_configs.py

import logging

# 1. 从【core】导入框架组件
from core.strategy import StrategyConfig

# 2. 从【strategies】导入所有具体的实现
from strategies.combiners import (EqualWeightCombiner, FixedWeightCombiner,
                                  DynamicSignificanceCombiner,
                                  DynamicWeightCombiner, AICombiner)
from strategies.ai_trainers import LightGBMTrainer
from strategies.rolling_calculators import AdversarialLLMCombiner

# ==============================================================================
#                 --- 在这里定义你的所有策略配置 ---
# ==============================================================================

ROLLING_ICIR_STRATEGY = StrategyConfig(combiner_class=DynamicWeightCombiner,
                                       combiner_kwargs={'factor_weights': {}},
                                       rolling_config={
                                           'CALCULATOR_TYPE': 'ICIR',
                                           'ROLLING_WINDOW_DAYS': 90,
                                           'REBALANCE_FREQUENCY': 'D',
                                           'FACTOR_WEIGHTING_CONFIG': {
                                               'Momentum': 'ic_mean_30d',
                                               'Reversal20D': 'ir_30d',
                                               'IndNeu_Momentum': 'ir_20d',
                                               'IndNeu_Reversal20D': 'ir_5d',
                                               'IndNeu_VolumeCV': 'ic_mean_10d'
                                           }
                                       })

FIXED_WEIGHT_STRATEGY = StrategyConfig(combiner_class=FixedWeightCombiner,
                                       combiner_kwargs={
                                           'factor_weights': {
                                               'BollingerBands': 0.3,
                                               'Momentum': 0.7
                                           }
                                       },
                                       rolling_config={})

EQUAL_WEIGHT_STRATEGY = StrategyConfig(combiner_class=EqualWeightCombiner,
                                       combiner_kwargs={},
                                       rolling_config={})

DYNAMIC_SIG_STRATEGY = StrategyConfig(
    combiner_class=DynamicSignificanceCombiner,
    combiner_kwargs={},
    rolling_config={})

ROLLING_REGRESSION_STRATEGY = StrategyConfig(
    combiner_class=DynamicWeightCombiner,
    combiner_kwargs={'factor_weights': {}},
    rolling_config={
        'CALCULATOR_TYPE': 'Regression',
        'ROLLING_WINDOW_DAYS': 90,
        'REBALANCE_FREQUENCY': 'D',
        'TARGET_RETURN_PERIOD': 30
    })

LIGHTGBM_STRATEGY = StrategyConfig(
    combiner_class=AICombiner,
    combiner_kwargs={'initial_model_path': None},
    rolling_config={
        'CALCULATOR_TYPE':
        'AI_Trainer',
        'ROLLING_WINDOW_DAYS':
        250,
        'REBALANCE_FREQUENCY':
        'MS',
        'TRAINING_CALCULATOR':
        LightGBMTrainer(target_return_period=30,
                        model_params={
                            'objective': 'regression_l1',
                            'metric': 'rmse',
                            'n_estimators': 300,
                            'learning_rate': 0.03,
                            'num_leaves': 31,
                            'max_depth': 6,
                            'min_child_samples': 50,
                            'subsample': 0.8,
                            'colsample_bytree': 0.7,
                            'random_state': 42,
                            'n_jobs': -1,
                            'verbose': -1,
                            'reg_alpha': 0.1,
                            'reg_lambda': 0.1
                        })
    })

ADVERSARIAL_LLM_STRATEGY = StrategyConfig(
    combiner_class=DynamicWeightCombiner,
    combiner_kwargs={'factor_weights': {}},
    rolling_config={
        'CALCULATOR_TYPE': 'AdversarialLLM',
        'ROLLING_WINDOW_DAYS': 90,
        'REBALANCE_FREQUENCY': 'MS',
        'API_URL':
        'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
        'API_KEY': 'sk-***',  # 请替换为实际的API密钥
        'MAX_ROUNDS': 2,
        'INCLUDE_FACTOR_VALUES': True,
        'INCLUDE_CONVERSATION_HISTORY': True,
        'ALLOW_NEGATIVE_WEIGHTS': True
    })

# ==============================================================================
#                 --- 在这里注册您的策略 ---
# ==============================================================================
STRATEGY_REGISTRY = {
    "RollingICIR": ROLLING_ICIR_STRATEGY,
    "FixedWeights": FIXED_WEIGHT_STRATEGY,
    "EqualWeights": EQUAL_WEIGHT_STRATEGY,
    "DynamicSignificance": DYNAMIC_SIG_STRATEGY,
    "RollingRegression": ROLLING_REGRESSION_STRATEGY,
    "LightGBM_Periodic": LIGHTGBM_STRATEGY,
    "AdversarialLLM": ADVERSARIAL_LLM_STRATEGY,
}
