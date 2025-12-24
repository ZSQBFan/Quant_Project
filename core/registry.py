"""
核心注册表管理系统

提供装饰器和注册表机制，用于自动发现和管理：
- 因子（Factors）
- 策略组件（Combiners, Standardizers, Rolling Calculators）
- 回测组件（Selectors, Allocators, Capital Managers, Triggers）

设计原则：
- 单一注册表实例（Singleton模式）
- 支持装饰器自动注册
- 支持手动注册和注销
- 提供查询和验证功能
"""

import logging
from typing import Dict, Type, Any, Optional, Callable
from collections import defaultdict


class ComponentRegistry:
    """
    组件注册表（单例）

    管理所有可插拔组件的注册和查询。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.logger = logging.getLogger(__name__)

        # 各领域的注册表
        self._registries: Dict[str, Dict[str, Any]] = {
            'factors': {},              # 因子库
            'combiners': {},            # 因子合成器
            'standardizers': {},        # 标准化器
            'rolling_calculators': {},  # 滚动计算器
            'selectors': {},            # 选股器
            'allocators': {},           # 权重分配器
            'capital_managers': {},     # 资金管理器
            'triggers': {},             # 触发器
        }

        # 元数据存储（可选，用于存储额外信息）
        self._metadata: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

        self._initialized = True
        self.logger.info("✅ ComponentRegistry 已初始化")

    def register(self, category: str, name: str, component: Any,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        注册组件

        Args:
            category: 组件类别（如 'factors', 'combiners'）
            name: 组件名称（唯一标识）
            component: 组件类或实例
            metadata: 可选的元数据（描述、参数等）
        """
        if category not in self._registries:
            raise ValueError(f"未知的组件类别: {category}")

        if name in self._registries[category]:
            # 检查是否为相同的组件（通过类名和模块判断）
            existing_component = self._registries[category][name]

            # 如果是同一个对象实例，直接跳过
            if existing_component is component:
                self.logger.debug(f"组件 '{name}' 已存在（同一实例），跳过重复注册")
                return

            # 如果是同名的类（可能由于模块重新加载导致），静默跳过
            # 比较类名和模块名来判断是否为"相同"的组件
            if (hasattr(existing_component, '__name__') and
                hasattr(component, '__name__') and
                existing_component.__name__ == component.__name__):
                self.logger.debug(f"组件 '{name}' 已存在（同名类），跳过重复注册")
                return

            # 真正不同的组件，发出警告
            self.logger.warning(f"组件 '{name}' 在类别 '{category}' 中已存在，将被覆盖")

        self._registries[category][name] = component

        if metadata:
            self._metadata[category][name] = metadata

        self.logger.debug(f"✅ 已注册 {category}/{name}")

    def unregister(self, category: str, name: str) -> None:
        """注销组件"""
        if category in self._registries and name in self._registries[category]:
            del self._registries[category][name]
            if category in self._metadata and name in self._metadata[category]:
                del self._metadata[category][name]
            self.logger.debug(f"❌ 已注销 {category}/{name}")

    def get(self, category: str, name: str) -> Any:
        """
        获取组件

        Args:
            category: 组件类别
            name: 组件名称

        Returns:
            组件类或实例

        Raises:
            KeyError: 组件不存在
        """
        if category not in self._registries:
            raise ValueError(f"未知的组件类别: {category}")

        if name not in self._registries[category]:
            available = list(self._registries[category].keys())
            raise KeyError(
                f"组件 '{name}' 在类别 '{category}' 中未找到。"
                f"可用组件: {available}"
            )

        return self._registries[category][name]

    def get_all(self, category: str) -> Dict[str, Any]:
        """获取某类别下的所有组件"""
        if category not in self._registries:
            raise ValueError(f"未知的组件类别: {category}")
        return self._registries[category].copy()

    def get_metadata(self, category: str, name: str) -> Dict[str, Any]:
        """获取组件元数据"""
        return self._metadata.get(category, {}).get(name, {})

    def exists(self, category: str, name: str) -> bool:
        """检查组件是否存在"""
        return category in self._registries and name in self._registries[category]

    def list_categories(self) -> list:
        """列出所有类别"""
        return list(self._registries.keys())

    def list_components(self, category: str) -> list:
        """列出某类别下的所有组件名称"""
        if category not in self._registries:
            raise ValueError(f"未知的组件类别: {category}")
        return list(self._registries[category].keys())

    def clear(self, category: Optional[str] = None) -> None:
        """清空注册表（慎用，主要用于测试）"""
        if category:
            if category in self._registries:
                self._registries[category].clear()
                self._metadata[category].clear()
                self.logger.warning(f"⚠️ 已清空类别: {category}")
        else:
            for cat in self._registries:
                self._registries[cat].clear()
            self._metadata.clear()
            self.logger.warning("⚠️ 已清空所有注册表")

    def get_stats(self) -> Dict[str, int]:
        """获取注册表统计信息"""
        return {
            category: len(components)
            for category, components in self._registries.items()
        }


# 全局单例实例
registry = ComponentRegistry()


# ============================================================================
# 装饰器：自动注册组件
# ============================================================================

def register_factor(name: str, **metadata):
    """
    因子注册装饰器

    用法:
        @register_factor('Momentum', category='simple', description='动量因子')
        class MomentumFactor(BaseFactor):
            ...
    """
    def decorator(cls):
        registry.register('factors', name, cls, metadata)
        return cls
    return decorator


def register_combiner(name: str, **metadata):
    """因子合成器注册装饰器"""
    def decorator(cls):
        registry.register('combiners', name, cls, metadata)
        return cls
    return decorator


def register_standardizer(name: str, **metadata):
    """标准化器注册装饰器"""
    def decorator(cls):
        registry.register('standardizers', name, cls, metadata)
        return cls
    return decorator


def register_rolling_calculator(name: str, **metadata):
    """滚动计算器注册装饰器"""
    def decorator(cls):
        registry.register('rolling_calculators', name, cls, metadata)
        return cls
    return decorator


def register_selector(name: str, **metadata):
    """选股器注册装饰器"""
    def decorator(cls):
        registry.register('selectors', name, cls, metadata)
        return cls
    return decorator


def register_allocator(name: str, **metadata):
    """权重分配器注册装饰器"""
    def decorator(cls):
        registry.register('allocators', name, cls, metadata)
        return cls
    return decorator


def register_capital_manager(name: str, **metadata):
    """资金管理器注册装饰器"""
    def decorator(cls):
        registry.register('capital_managers', name, cls, metadata)
        return cls
    return decorator


def register_trigger(name: str, **metadata):
    """触发器注册装饰器"""
    def decorator(cls):
        registry.register('triggers', name, cls, metadata)
        return cls
    return decorator


# ============================================================================
# 便捷函数
# ============================================================================

def get_factor(name: str) -> Type:
    """获取因子类"""
    return registry.get('factors', name)


def get_combiner(name: str) -> Type:
    """获取因子合成器类"""
    return registry.get('combiners', name)


def get_standardizer(name: str) -> Type:
    """获取标准化器类"""
    return registry.get('standardizers', name)


def get_rolling_calculator(name: str) -> Type:
    """获取滚动计算器类"""
    return registry.get('rolling_calculators', name)


def get_selector(name: str) -> Type:
    """获取选股器类"""
    return registry.get('selectors', name)


def get_allocator(name: str) -> Type:
    """获取权重分配器类"""
    return registry.get('allocators', name)


def get_capital_manager(name: str) -> Type:
    """获取资金管理器类"""
    return registry.get('capital_managers', name)


def get_trigger(name: str) -> Type:
    """获取触发器类"""
    return registry.get('triggers', name)


def list_all_components() -> Dict[str, list]:
    """列出所有组件"""
    return {
        category: registry.list_components(category)
        for category in registry.list_categories()
    }


def print_registry_stats():
    """输出注册表统计信息到日志（避免 print 污染控制台）。"""
    stats = registry.get_stats()
    logger = logging.getLogger(__name__)
    logger.debug("组件注册表统计：")
    for category, count in stats.items():
        logger.debug(f"- {category}: {count} 个组件")


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # 测试注册
    @register_factor('TestFactor', category='simple', description='测试因子')
    class TestFactor:
        pass

    @register_combiner('TestCombiner', description='测试合成器')
    class TestCombiner:
        pass

    # 测试查询
    print("✅ 测试注册成功")
    print(f"因子列表: {registry.list_components('factors')}")
    print(f"合成器列表: {registry.list_components('combiners')}")

    # 测试获取
    factor_cls = get_factor('TestFactor')
    print(f"✅ 获取因子类: {factor_cls}")

    # 测试统计
    print_registry_stats()
