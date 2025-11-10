# strategy_configs.py
import pandas as pd
from core.factor_combiner import (
    DynamicWeightCombiner,  # <-- 【【重命名】】
    FixedWeightCombiner,  #
    EqualWeightCombiner,  #
    DynamicSignificanceCombiner  #
)
from core.rolling_weight_calculator import (
    RollingICIRCalculator,  # <-- 【【重命名】】
    RollingRegressionCalculator  # <-- 【【新增】】
)
"""
【【【策略配置中心】】】

欢迎使用！这是您配置多因子策略的核心文件。

您无需再编辑 main_analyzer.py 中的复杂逻辑，只需在此文件中定义
您的策略，然后在 main_analyzer.py 的 "1a" 部分选择您注册的
【策略名称 (STRATEGY_NAME)】 即可。

【【如何添加一个新策略？】】
1.  在下方 "在这里定义你的所有策略" 部分，仿照示例创建一个新的 `StrategyConfig` 实例。
2.  确保您已经：
    a.  (如果需要) 在 `factor_combiner.py` 中定义了新的合成器 (Combiner) 类。
    b.  (如果需要) 在本文件顶部 `import` 了它。
3.  在文件底部的 `STRATEGY_REGISTRY` 字典中，为您
    的新策略注册一个唯一的“字符串名称”。
4.  去 `main_analyzer.py` 的 "1a" 部分，将 `STRATEGY_NAME` 
    设置为您新注册的字符串。
"""


class StrategyConfig:
    """
    【【架构文件 - 策略容器】】
    
    一个用于封装策略所需【全部配置】的容器类。
    
    `main_analyzer.py` 会创建一个此类的实例，然后调用它的
    `create_combiner` 和 `is_rolling` 等方法，
    而无需关心内部的配置细节。
    """

    def __init__(self, combiner_class, combiner_kwargs, rolling_config=None):
        """
        初始化策略配置。

        Args:
            combiner_class (BaseFactorCombiner): 
                要使用的合成器类 (例如 DynamicWeightCombiner)。
            
            combiner_kwargs (dict): 
                在创建合成器实例时需要传入的参数 (例如 'factor_weights')。
            
            rolling_config (dict, optional): 
                如果这是一个滚动策略，这里应包含滚动逻辑所需的所有参数。
                如果为 None，则表明这是一个静态策略。
        """
        self.combiner_class = combiner_class
        self.combiner_kwargs = combiner_kwargs
        self.rolling_config = rolling_config

    def create_combiner(self):
        """
        根据配置创建合成器 (Combiner) 实例。
        """
        return self.combiner_class(**self.combiner_kwargs)

    def create_rolling_calculator(self, forward_return_periods, factor_names):
        """
        【【【核心重构：计算器工厂】】】
        
        如果配置了滚动，则根据 'CALCULATOR_TYPE' 创建【对应】的
        滚动计算器 (Rolling Calculator) 实例。
        """
        if not self.is_rolling():
            return None

        # 从配置中获取计算器类型，默认为 'ICIR' 以保持向后兼容
        calc_type = self.rolling_config.get('CALCULATOR_TYPE', 'ICIR')

        # --- 1. IC/IR 计算器 ---
        if calc_type == 'ICIR':
            config = self.rolling_config.get('FACTOR_WEIGHTING_CONFIG', {})
            if not config:
                print("警告: 滚动ICIR策略未配置 'FACTOR_WEIGHTING_CONFIG'。")

            return RollingICIRCalculator(
                factor_weight_config=config,
                forward_return_periods=forward_return_periods,
                factor_names=factor_names)

        # --- 2. 回归 计算器 ---
        elif calc_type == 'Regression':
            target_period = self.rolling_config.get('TARGET_RETURN_PERIOD',
                                                    None)
            if target_period is None:
                raise ValueError(
                    "回归策略必须在 rolling_config 中指定 'TARGET_RETURN_PERIOD'。")
            if target_period not in forward_return_periods:
                raise ValueError(
                    f"目标周期 {target_period}d 不在 main_analyzer.py 定义的 {forward_return_periods} 中。"
                )

            # (传递 forward_return_periods 和 factor_names 以满足通用接口)
            # (kwargs 使得 RegressionCalculator 可以只接收它需要的参数)
            return RollingRegressionCalculator(
                target_return_period=target_period,
                factor_names=factor_names,
                forward_return_periods=forward_return_periods)

        # --- (未来可在此添加 'ML' 或 'Optimization' 等计算器) ---

        else:
            raise ValueError(f"未知的滚动计算器类型: {calc_type}")

    def is_rolling(self) -> bool:
        """
        检查此配置是否为【滚动策略】。
        """
        # 核心检测逻辑：
        # 1. 必须在上方定义了 rolling_config (非 None)
        # 2. 必须是 DynamicWeightCombiner (原ICIRWeightCombiner)
        return (self.rolling_config is not None
                and self.combiner_class == DynamicWeightCombiner)

    def get_rolling_param(self, key, default=None):
        """
        安全地从滚动配置中获取参数 (例如 ROLLING_WINDOW_DAYS)。
        """
        if self.is_rolling():
            return self.rolling_config.get(key, default)
        return default


# ==============================================================================
#                 --- 【【【用户配置区 1】】】 ---
#                 ---  在这里定义你的所有策略配置  ---
# ==============================================================================

# --- 策略 1: 动态滚动 IC/IR 加权 ---
# 描述: 这是一个【滚动】策略。
#       它会自动计算因子的历史表现 (IC/IR)，并【定期更新】因子权重。
ROLLING_ICIR_STRATEGY = StrategyConfig(
    # 1. 合成器: 使用 【重命名后】 的 DynamicWeightCombiner
    combiner_class=DynamicWeightCombiner,
    # 2. 合成器参数: 初始权重设置为空字典，等待滚动逻辑填充
    combiner_kwargs={'factor_weights': {}},
    # 3. 滚动配置: (因为这是滚动策略，所以【必须】提供此项)
    rolling_config={
        'CALCULATOR_TYPE': 'ICIR',  # <-- 【新增】明确指定计算器类型
        # 回看多少天的数据来计算 IC/IR
        'ROLLING_WINDOW_DAYS': 90,
        # 多久重新计算一次权重 (MS = Month Start)
        'REBALANCE_FREQUENCY': 'MS',
        # 【重要】: 用哪个指标来决定权重？
        # 格式: '因子名': '指标key_周期d'
        # 指标key: 'ir' (推荐), 'ic_mean', 'ic_t_stat'
        # 周期d: 必须是 main_analyzer.py "2a" 中定义的周期
        'FACTOR_WEIGHTING_CONFIG': {
            'RSI': 'ir_5d',
            'BollingerBands': 'ir_5d',
            'Momentum': 'ir_5d',
            'ADXDMI': 'ic_mean_10d',
            'Reversal20D': 'ir_10d',  # 反转因子用10日滚动IR
            'IndNeu_Momentum': 'ir_30d',  # 动量因子用30日滚动IR
            'IndNeu_Reversal20D': 'ir_10d',
            'IndNeu_VolumeCV': 'ir_10d',
            'IndNeu_ADXDMI': 'ic_mean_10d'
        }
    })

# --- 策略 2: 静态固定权重 ---
# 描述: 这是一个【静态】策略。
#       您需要在这里手动指定【固定】的因子权重，它在回测中【不会】改变。
FIXED_WEIGHT_STRATEGY = StrategyConfig(
    # 1. 合成器: 使用 FixedWeightCombiner
    combiner_class=FixedWeightCombiner,
    # 2. 合成器参数: 【重要】在这里填入您的固定权重
    combiner_kwargs={
        'factor_weights': {
            'BollingerBands': 0.3,
            'Momentum': 0.7,
            # (注意: 如果因子在 main_analyzer.py "1b" 中未启用，
            #  这里的权重会被安全地忽略)
        }
    },
    # 3. 滚动配置: (静态策略，设为 None)
    rolling_config=None)

# --- 策略 3: 静态等权重 ---
# 描述: 这是一个【静态】策略。
#       所有在 main_analyzer.py "1b" 中启用的因子将获得【相同】的权重。
EQUAL_WEIGHT_STRATEGY = StrategyConfig(
    # 1. 合成器: 使用 EqualWeightCombiner
    combiner_class=EqualWeightCombiner,
    # 2. 合成器参数: (无需特定参数)
    combiner_kwargs={},
    # 3. 滚动配置: (静态策略，设为 None)
    rolling_config=None)

# --- 策略 4: 动态显著性加权 ---
# 描述: 这是一个【静态】策略 (虽然名字叫动态)。
#       它【逐日】查看哪个因子的【信号更强】(Z-Score 绝对值更大)，
#       就【实时】给哪个因子更高的权重。它不需要历史回看。
DYNAMIC_SIG_STRATEGY = StrategyConfig(
    # 1. 合成器: 使用 DynamicSignificanceCombiner
    combiner_class=DynamicSignificanceCombiner,
    # 2. 合成器参数: (无需特定参数)
    combiner_kwargs={},
    # 3. 滚动配置: (静态策略，设为 None)
    rolling_config=None)

# --- 【【【策略 5: (新增) 动态滚动回归加权】】】 ---
# 描述: 这是一个【滚动】策略。
#       它使用过去90天的数据，运行一个多元线性回归
#       (Y=30日收益, X=因子值)，并将回归系数 (Betas)
#       的绝对值作为新的因子权重。
ROLLING_REGRESSION_STRATEGY = StrategyConfig(
    # 1. 合成器: 使用 DynamicWeightCombiner (与 ICIR 策略相同)
    combiner_class=DynamicWeightCombiner,
    # 2. 合成器参数: 初始权重为空
    combiner_kwargs={'factor_weights': {}},
    # 3. 滚动配置:
    rolling_config={
        'CALCULATOR_TYPE': 'Regression',  # <-- 【【指定使用回归计算器】】
        'ROLLING_WINDOW_DAYS': 90,  # 回看窗口
        'REBALANCE_FREQUENCY': 'MS',  # 调仓频率

        # 【重要】: 指定回归的目标Y值 (必须是 main_analyzer "2a" 中定义的周期)
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
