"""
Core Framework

提供：
- 注册表管理 (registry)
- 动态加载器 (loader)
- 配置加载器 (config)
- 抽象基类 (abstractions)
"""

from .registry import (
    registry,
    register_factor,
    register_combiner,
    register_standardizer,
    register_rolling_calculator,
    register_selector,
    register_allocator,
    register_capital_manager,
    register_trigger,
    get_factor,
    get_combiner,
    get_standardizer,
    get_rolling_calculator,
    get_selector,
    get_allocator,
    get_capital_manager,
    get_trigger,
    list_all_components,
    print_registry_stats
)

from .loader import (
    ModuleLoader,
    auto_load_all,
    print_registry_stats
)

from .config import (
    ConfigLoader,
    FactorConfig,
    StrategyConfig,
    BacktestConfig,
    load_config,
    get_factor_config,
    get_strategy_config
)

from .abstractions import (
    BaseStandardizer,
    BaseFactorCombiner,
    RollingCalculatorBase,
    AITrainerBase
)

__all__ = [
    # Registry
    'registry',
    'register_factor',
    'register_combiner',
    'register_standardizer',
    'register_rolling_calculator',
    'register_selector',
    'register_allocator',
    'register_capital_manager',
    'register_trigger',
    'get_factor',
    'get_combiner',
    'get_standardizer',
    'get_rolling_calculator',
    'get_selector',
    'get_allocator',
    'get_capital_manager',
    'get_trigger',
    'list_all_components',
    'print_registry_stats',

    # Loader
    'ModuleLoader',
    'auto_load_all',
    'print_registry_stats',
]
