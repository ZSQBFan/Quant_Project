"""
数据处理流水线
Pandas因子数据 -> Backtrader DataFeed 的数据转换和加载
"""

from .selectors import (
    SelectorBase,
    TopNSelector,
    create_top_n_selector,
    get_selector,
    SELECTOR_REGISTRY
)

from .allocators import (
    AllocatorBase,
    EqualWeightAllocator,
    create_equal_weight_allocator,
    get_allocator,
    ALLOCATOR_REGISTRY,
    validate_weight_allocation
)

from .capital import (
    CapitalManagerBase,
    FullPositionManager,
    create_full_position_manager,
    get_capital_manager,
    CAPITAL_MANAGER_REGISTRY
)

__all__ = [
    # Selectors
    'SelectorBase',
    'TopNSelector',
    'create_top_n_selector',
    'get_selector',
    'SELECTOR_REGISTRY',
    # Allocators
    'AllocatorBase',
    'EqualWeightAllocator',
    'create_equal_weight_allocator',
    'get_allocator',
    'ALLOCATOR_REGISTRY',
    'validate_weight_allocation',
    # Capital Managers
    'CapitalManagerBase',
    'FullPositionManager',
    'create_full_position_manager',
    'get_capital_manager',
    'CAPITAL_MANAGER_REGISTRY'
]

