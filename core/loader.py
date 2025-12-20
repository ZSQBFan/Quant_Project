"""
动态模块加载器

提供模块的自动加载和统计功能，支持层级结构显示
"""

import os
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ModuleLoader:
    """模块加载器"""
    
    def __init__(self):
        self.loaded_modules: Dict[str, List[str]] = {}  # package_name -> module_list
        self.package_stats: Dict[str, Any] = {}  # package_name -> stats_info
        
    def load_modules_from_directory(self, directory: str, package_name: str) -> Tuple[int, List[str]]:
        """
        从指定目录加载模块
        
        Args:
            directory: 模块目录路径
            package_name: 包名（用于层级结构）
            
        Returns:
            Tuple[int, List[str]]: (模块数量, 模块名列表)
        """
        if not os.path.exists(directory):
            logger.warning(f"目录不存在: {directory}")
            return 0, []
            
        modules = []
        module_files = []
        
        # 查找所有 Python 文件
        for file_path in Path(directory).glob("*.py"):
            if file_path.name == "__init__.py":
                continue
            module_files.append(file_path)
            
        # 加载每个模块
        for file_path in module_files:
            module_name = file_path.stem
            
            try:
                # 构建模块导入路径
                if package_name:
                    full_module_name = f"{package_name}.{module_name}"
                else:
                    full_module_name = module_name
                    
                # 动态导入模块
                spec = importlib.util.spec_from_file_location(full_module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    
                    # 执行模块（仅加载，不运行）
                    spec.loader.exec_module(module)
                    modules.append(module_name)
                    
                    logger.debug(f"成功加载模块: {full_module_name}")
                else:
                    logger.warning(f"无法创建模块规范: {file_path}")
                    
            except Exception as e:
                logger.warning(f"加载模块失败 {module_name}: {e}")
                continue
                
        # 保存加载结果
        self.loaded_modules[package_name] = modules
        self.package_stats[package_name] = {
            'count': len(modules),
            'modules': modules,
            'directory': directory
        }
        
        logger.info(f"从 {package_name} 加载了 {len(modules)} 个模块")
        return len(modules), modules
    
    def load_all_components(self, base_path: str = ".") -> Dict[str, Any]:
        """
        加载所有组件模块
        
        Args:
            base_path: 基础路径
            
        Returns:
            Dict: 加载统计信息
        """
        components = {
            'factors': {
                'library': 'factors/library',
                'library.industry_neutral': 'factors/library/industry_neutral',
                'pipeline.combiners': 'factors/pipeline/combiners',
                'pipeline.standardizers': 'factors/pipeline/standardizers',
                'analysis': 'factors/analysis'
            },
            'backtest': {
                'pipeline.selectors': 'backtest/pipeline/selectors',
                'pipeline.allocators': 'backtest/pipeline/allocators',
                'pipeline.capital': 'backtest/pipeline/capital',
                'triggers': 'backtest/triggers'
            },
            'data': {
                'providers': 'data/providers',
                'handlers': 'data/handlers'
            }
        }
        
        total_modules = 0
        total_packages = 0
        
        logger.info("开始加载所有组件...")
        
        for category, packages in components.items():
            for package_name, relative_path in packages.items():
                full_path = os.path.join(base_path, relative_path)
                count, modules = self.load_modules_from_directory(full_path, f"{category}.{package_name}")
                
                total_modules += count
                total_packages += 1
                
        logger.info(f"加载完成: {total_packages} 个包, {total_modules} 个模块")
        
        return {
            'total_packages': total_packages,
            'total_modules': total_modules,
            'package_stats': self.package_stats.copy()
        }
    
    def _print_stats(self):
        """打印详细的模块加载统计信息"""
        if not self.package_stats:
            print("暂无加载的模块统计信息")
            return
            
        print("=" * 60)
        print("模块加载统计")
        print("=" * 60)
        
        # 构建树形结构数据
        tree_data = {}
        
        # 处理每个包和其模块
        for package_name, stats in self.package_stats.items():
            modules = stats['modules']
            
            # 解析包名路径
            path_parts = package_name.split('.')
            current = tree_data
            
            # 构建目录结构
            for i, part in enumerate(path_parts):
                if part not in current:
                    current[part] = {}
                current = current[part]
                
                # 如果是最后一个路径部分，添加模块
                if i == len(path_parts) - 1:
                    current['__modules__'] = modules
                    current['__package_name__'] = package_name
        
        # 递归打印树形结构
        def print_tree(node, prefix="", is_last=True, level=0):
            """递归打印树形结构"""
            items = list(node.items())
            # 过滤掉特殊键
            items = [(k, v) for k, v in items if not k.startswith('__')]
            
            for i, (name, value) in enumerate(items):
                is_last_item = (i == len(items) - 1)
                
                # 选择合适的连接符
                if level == 0:
                    # 根级别
                    if is_last_item:
                        connector = "└── "
                    else:
                        connector = "├── "
                    print(f"{prefix}{connector}{name}")
                else:
                    # 子级别
                    if is_last_item:
                        connector = "└── "
                    else:
                        connector = "├── "
                    
                    print(f"{prefix}{connector}{name}")
                
                # 递归打印子项
                if isinstance(value, dict):
                    # 打印模块
                    if '__modules__' in value:
                        modules = value['__modules__']
                        if modules:
                            module_prefix = prefix + ("    " if is_last_item else "│   ")
                            
                            for j, module in enumerate(modules):
                                is_last_module = (j == len(modules) - 1)
                                if is_last_module:
                                    module_connector = "└── "
                                else:
                                    module_connector = "├── "
                                print(f"{module_prefix}{module_connector}{module}")
                    
                    # 递归打印子目录
                    sub_node = {k: v for k, v in value.items() if not k.startswith('__')}
                    if sub_node:
                        sub_prefix = prefix + ("    " if is_last_item else "│   ")
                        print_tree(sub_node, sub_prefix, is_last_item, level + 1)
        
        # 打印根结构
        print_tree(tree_data)
        
        print("=" * 60)
        
        # 显示每个包的模块计数
        print("📊 各包模块计数：")
        for package_name, stats in sorted(self.package_stats.items()):
            count = stats['count']
            # 格式化输出，保持对齐
            print(f"✅ {package_name:<35} : {count:>2} 个模块")
                
        print("=" * 60)
        
        # 显示总计信息
        total_packages = len(self.package_stats)
        total_modules = sum(stats['count'] for stats in self.package_stats.values())
        
        print(f"总计: {total_packages} 个包, {total_modules} 个模块")
        print("=" * 60)


# 全局加载器实例，避免重复加载
_global_loader = None

def auto_load_all() -> ModuleLoader:
    """
    自动加载所有组件
    
    Returns:
        ModuleLoader: 模块加载器实例
    """
    global _global_loader
    if _global_loader is None:
        _global_loader = ModuleLoader()
        _global_loader.load_all_components()
    return _global_loader


def print_registry_stats():
    """打印注册表统计信息（使用全局加载器实例避免重复加载）"""
    global _global_loader
    if _global_loader is None:
        _global_loader = ModuleLoader()
        _global_loader.load_all_components()
    _global_loader._print_stats()
