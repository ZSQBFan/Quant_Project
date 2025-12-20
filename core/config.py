"""
配置加载器

从 YAML 文件加载配置，支持新的分层配置结构：

configs/
├── backtest/           # 回测配置
│   ├── config.yaml     # 总编排
│   ├── pipeline/       # Pipeline组件
│   └── triggers/       # 触发器
├── factors/            # 因子评测配置
│   ├── config.yaml     # 总编排
│   ├── pipeline/       # Pipeline组件
│   ├── library/        # 因子库
│   └── strategies.yaml # 策略配置
├── data/               # 数据下载配置
│   ├── config.yaml     # 总编排
│   ├── providers/      # 数据源
│   └── calendar.yaml   # 日历
└── universe.yaml       # 股票池

设计原则：
- 配置验证
- 默认值处理
- 支持环境变量替换
- 向后兼容旧配置结构
"""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class FactorConfig:
    """因子配置"""
    name: str
    category: str  # 'simple' or 'complex'
    params: Dict[str, Any] = field(default_factory=dict)
    required_columns: List[str] = field(default_factory=list)
    description: str = ""
    enabled: bool = True


@dataclass
class StrategyConfig:
    """策略配置"""
    name: str
    combiner: str
    combiner_params: Dict[str, Any] = field(default_factory=dict)
    standardizer: str = "ZScore"
    rolling: Optional[Dict[str, Any]] = None
    description: str = ""


@dataclass
class TriggerConfig:
    """触发器配置"""
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class PipelineConfig:
    """Pipeline 组件配置"""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    enabled: bool = True


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: str
    end_date: str
    initial_cash: float = 1000000.0
    commission: float = 0.0003

    # Pipeline 组件
    selector: str = "TopN"
    selector_params: Dict[str, Any] = field(default_factory=dict)

    allocator: str = "EqualWeight"
    allocator_params: Dict[str, Any] = field(default_factory=dict)

    capital_manager: str = "FullPosition"
    capital_manager_params: Dict[str, Any] = field(default_factory=dict)

    # 触发器
    triggers: List[TriggerConfig] = field(default_factory=list)

    # 其他参数
    benchmark: Optional[str] = None
    output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorAnalysisConfig:
    """因子分析配置"""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    forward_return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 30])
    factor_calculation: Dict[str, Any] = field(default_factory=dict)
    default_strategy: str = "EqualWeights"
    default_standardizer: str = "ZScore"
    benchmark: str = "600519"
    output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """数据配置"""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    use_all_stocks: bool = False
    database: Dict[str, Any] = field(default_factory=dict)
    provider_priority: List[str] = field(default_factory=lambda: ["sqlite", "tushare", "akshare"])
    download: Dict[str, Any] = field(default_factory=dict)
    integrity: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    """数据提供者配置"""
    name: str
    enabled: bool = True
    priority: int = 1
    config: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 配置加载器
# ============================================================================

class ConfigLoader:
    """
    配置加载器

    支持新的分层配置结构，同时向后兼容旧的平级配置结构。
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化配置加载器

        Args:
            config_dir: 配置文件目录，默认为项目根目录下的 configs/
        """
        self.logger = logging.getLogger(__name__)

        if config_dir is None:
            # 默认：项目根目录/configs
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            config_dir = project_root / 'configs'

        self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            self.logger.warning(f"配置目录不存在: {self.config_dir}")
            self.logger.info(f"将创建配置目录: {self.config_dir}")
            self.config_dir.mkdir(parents=True, exist_ok=True)

        # 检测配置结构类型
        self._is_new_structure = self._detect_structure()
        structure_type = "新分层结构" if self._is_new_structure else "旧平级结构"
        self.logger.info(f"ConfigLoader 初始化，配置目录: {self.config_dir} ({structure_type})")

    def _detect_structure(self) -> bool:
        """检测配置结构类型"""
        new_structure_dirs = ['backtest', 'factors', 'data']
        return all((self.config_dir / d).is_dir() for d in new_structure_dirs)

    def _load_yaml(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        加载 YAML 文件

        Args:
            filepath: 文件路径（相对于config_dir或绝对路径）

        Returns:
            配置字典
        """
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        # 如果是相对路径，则相对于 config_dir
        if not filepath.is_absolute():
            filepath = self.config_dir / filepath

        if not filepath.exists():
            self.logger.debug(f"配置文件不存在: {filepath}")
            return {}

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # 支持环境变量替换 ${VAR_NAME}
                content = self._expand_env_vars(content)
                config = yaml.safe_load(content)
                self.logger.debug(f"已加载配置: {filepath.name}")
                return config or {}
        except yaml.YAMLError as e:
            self.logger.error(f"YAML 解析错误 ({filepath}): {e}")
            raise
        except Exception as e:
            self.logger.error(f"读取配置文件失败 ({filepath}): {e}")
            raise

    def _expand_env_vars(self, content: str) -> str:
        """展开环境变量"""
        pattern = r'\$\{([^}]+)\}'

        def replacer(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(pattern, replacer, content)

    def _load_directory_configs(self, subdir: str) -> Dict[str, Dict[str, Any]]:
        """
        加载目录下所有 YAML 文件

        Args:
            subdir: 子目录路径

        Returns:
            {文件名(无扩展名): 配置字典}
        """
        dir_path = self.config_dir / subdir
        if not dir_path.is_dir():
            return {}

        configs = {}
        for filepath in dir_path.glob('*.yaml'):
            name = filepath.stem
            # 传递相对于 config_dir 的路径，而不是完整路径
            # 因为 _load_yaml 会自动拼接 config_dir
            relative_path = f"{subdir}/{filepath.name}"
            configs[name] = self._load_yaml(relative_path)

        return configs

    # ========================================================================
    # 因子配置加载
    # ========================================================================

    def load_factors(self) -> Dict[str, FactorConfig]:
        """
        加载因子配置
        
        仅从主配置文件的 enabled_factors 字段读取因子定义

        Returns:
            {因子名称: FactorConfig} 字典
        """
        factors = {}

        # 加载主配置文件
        main_config = self._load_yaml('factors/config.yaml')
        enabled_factors_config = main_config.get('enabled_factors', [])
        
        if not enabled_factors_config:
            raise RuntimeError(
                "配置错误：enabled_factors 字段为空或不存在！\n"
                "请在 configs/factors/config.yaml 中至少配置一个因子。"
            )

        # 从因子库获取默认配置
        default_factor_configs = self._get_all_default_factor_configs()

        for factor_config in enabled_factors_config:
            try:
                # 支持两种格式：
                # 1. 简单格式：'- FactorName'  # 直接使用因子名称
                # 2. 完整格式：'- FactorName: {category: simple, params: {...}, ...}'
                if isinstance(factor_config, str):
                    # 简单格式：直接使用因子名称
                    factor_name = factor_config
                    user_config = {}
                elif isinstance(factor_config, dict):
                    # 完整格式：解析因子定义
                    factor_name = list(factor_config.keys())[0]
                    user_config = factor_config[factor_name]
                else:
                    self.logger.warning(f"跳过无效的因子配置格式: {factor_config}")
                    continue
                
                # 从默认配置中获取因子定义
                default_config = default_factor_configs.get(factor_name, {})
                if not default_config:
                    self.logger.warning(f"因子 '{factor_name}' 在因子库中未找到，跳过")
                    continue
                
                # 合并配置：默认配置 + 用户配置
                final_config = default_config.copy()
                final_config.update(user_config)
                
                factors[factor_name] = FactorConfig(
                    name=factor_name,
                    category=final_config.get('category', 'simple'),
                    params=final_config.get('params', {}),
                    required_columns=final_config.get('required_columns', []),
                    description=final_config.get('description', ''),
                    enabled=True
                )
                
            except Exception as e:
                self.logger.error(f"因子配置解析失败: {e}")

        if not factors:
            raise RuntimeError(
                f"没有成功加载任何因子！\n"
                f"请检查 enabled_factors 配置是否正确: {enabled_factors_config}"
            )

        self.logger.info(f"已加载 {len(factors)} 个因子配置")
        self.logger.info(f"启用的因子: {list(factors.keys())}")
        return factors
    
    def _get_all_default_factor_configs(self) -> Dict[str, Any]:
        """
        从因子库获取所有因子的默认配置
        
        Returns:
            所有因子默认配置的字典
        """
        try:
            if self._is_new_structure:
                # 从因子库文件加载默认配置
                simple_factors = self._load_yaml('factors/library/simple.yaml')
                complex_factors = self._load_yaml('factors/library/complex.yaml')
                return {**simple_factors, **complex_factors}
            else:
                # 旧结构
                return self._load_yaml('factors.yaml')
        except Exception as e:
            self.logger.error(f"加载因子库配置失败: {e}")
            return {}

    # ========================================================================
    # 策略配置加载
    # ========================================================================

    def load_strategies(self) -> Dict[str, StrategyConfig]:
        """
        加载策略配置

        Returns:
            {策略名称: StrategyConfig} 字典
        """
        if self._is_new_structure:
            config_dict = self._load_yaml('factors/pipeline/combiners.yaml')
        else:
            config_dict = self._load_yaml('strategies.yaml')

        strategies = {}
        for name, cfg in config_dict.items():
            try:
                strategies[name] = StrategyConfig(
                    name=name,
                    combiner=cfg.get('combiner', 'EqualWeight'),
                    combiner_params=cfg.get('combiner_params', {}),
                    standardizer=cfg.get('standardizer', 'ZScore'),
                    rolling=cfg.get('rolling'),
                    description=cfg.get('description', '')
                )
            except Exception as e:
                self.logger.error(f"策略配置解析失败 ({name}): {e}")

        self.logger.info(f"已加载 {len(strategies)} 个策略配置")
        return strategies

    # ========================================================================
    # 回测配置加载
    # ========================================================================

    def load_backtest(self) -> BacktestConfig:
        """
        加载回测配置

        日期范围继承顺序：backtest -> data（如backtest未指定则从data继承）

        Returns:
            BacktestConfig 实例
        """
        if self._is_new_structure:
            # 新结构：从多个文件合并
            main_config = self._load_yaml('backtest/config.yaml')

            # 加载数据配置以获取默认日期（继承链：data -> factors -> backtest）
            data_config = self._load_yaml('data/config.yaml')

            # 加载 Pipeline 组件配置
            selectors = self._load_yaml('backtest/pipeline/selectors.yaml')
            allocators = self._load_yaml('backtest/pipeline/allocators.yaml')
            capital = self._load_yaml('backtest/pipeline/capital.yaml')

            # 加载触发器配置
            rebalance = self._load_yaml('backtest/triggers/rebalance.yaml')
            stop_loss = self._load_yaml('backtest/triggers/stop_loss.yaml')

            # 获取选中的组件参数
            selector_name = main_config.get('selector', 'TopN')
            selector_params = selectors.get(selector_name, {}).get('params', {})

            allocator_name = main_config.get('allocator', 'EqualWeight')
            allocator_params = allocators.get(allocator_name, {}).get('params', {})

            capital_manager_name = main_config.get('capital_manager', 'FullPosition')
            capital_manager_params = capital.get(capital_manager_name, {}).get('params', {})

            # 构建触发器列表
            triggers = []
            for trigger_cfg in main_config.get('triggers', []):
                trigger_type = trigger_cfg.get('type')
                if trigger_type == 'RebalanceDay':
                    trigger_data = rebalance.get('RebalanceDay', {})
                elif trigger_type == 'StopLoss':
                    trigger_data = stop_loss.get('StopLoss', {})
                else:
                    trigger_data = {}

                if trigger_data.get('enabled', True):
                    triggers.append(TriggerConfig(
                        type=trigger_type,
                        params=trigger_data.get('params', {}),
                        enabled=True
                    ))

            # 日期继承：backtest -> data
            start_date = main_config.get('start_date') or data_config.get('start_date', '2024-01-01')
            end_date = main_config.get('end_date') or data_config.get('end_date', '2024-12-31')

            config_dict = main_config
        else:
            # 旧结构：从单个文件加载
            config_dict = self._load_yaml('backtest.yaml')

            selector_name = config_dict.get('selector', 'TopN')
            selector_params = config_dict.get('selector_params', {})
            allocator_name = config_dict.get('allocator', 'EqualWeight')
            allocator_params = config_dict.get('allocator_params', {})
            capital_manager_name = config_dict.get('capital_manager', 'FullPosition')
            capital_manager_params = config_dict.get('capital_manager_params', {})

            triggers = []
            for trigger_cfg in config_dict.get('triggers', []):
                triggers.append(TriggerConfig(
                    type=trigger_cfg.get('type'),
                    params=trigger_cfg.get('params', {}),
                    enabled=True
                ))

            start_date = config_dict.get('start_date', '2024-01-01')
            end_date = config_dict.get('end_date', '2024-12-31')

        try:
            return BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_cash=config_dict.get('initial_cash', 1000000.0),
                commission=config_dict.get('commission', 0.0003),
                selector=selector_name,
                selector_params=selector_params,
                allocator=allocator_name,
                allocator_params=allocator_params,
                capital_manager=capital_manager_name,
                capital_manager_params=capital_manager_params,
                triggers=triggers,
                benchmark=config_dict.get('benchmark'),
                output=config_dict.get('output', {})
            )
        except Exception as e:
            self.logger.error(f"回测配置解析失败: {e}")
            raise

    # ========================================================================
    # 因子分析配置加载
    # ========================================================================

    def load_factor_analysis(self) -> FactorAnalysisConfig:
        """
        加载因子分析配置

        日期范围继承顺序：factors -> data（如factors未指定则从data继承）

        Returns:
            FactorAnalysisConfig 实例
        """
        if self._is_new_structure:
            config_dict = self._load_yaml('factors/config.yaml')
            # 加载数据配置以获取默认日期
            data_config = self._load_yaml('data/config.yaml')

            # 日期继承：factors -> data
            start_date = config_dict.get('start_date') or data_config.get('start_date')
            end_date = config_dict.get('end_date') or data_config.get('end_date')
        else:
            # 旧结构没有专门的因子分析配置，返回默认值
            config_dict = {}
            start_date = None
            end_date = None

        return FactorAnalysisConfig(
            start_date=start_date,
            end_date=end_date,
            forward_return_periods=config_dict.get('forward_return_periods', [1, 5, 10, 20, 30]),
            factor_calculation=config_dict.get('factor_calculation', {}),
            default_strategy=config_dict.get('default_strategy', 'EqualWeights'),
            default_standardizer=config_dict.get('default_standardizer', 'ZScore'),
            benchmark=config_dict.get('benchmark', '600519'),
            output=config_dict.get('output', {})
        )

    # ========================================================================
    # 数据配置加载
    # ========================================================================

    def load_data(self) -> DataConfig:
        """
        加载数据配置

        Returns:
            DataConfig 实例
        """
        if self._is_new_structure:
            config_dict = self._load_yaml('data/config.yaml')
        else:
            config_dict = {}

        return DataConfig(
            start_date=config_dict.get('start_date'),
            end_date=config_dict.get('end_date'),
            use_all_stocks=config_dict.get('use_all_stocks', False),
            database=config_dict.get('database', {}),
            provider_priority=config_dict.get('provider_priority', ['sqlite', 'tushare', 'akshare']),
            download=config_dict.get('download', {}),
            integrity=config_dict.get('integrity', {}),
            output=config_dict.get('output', {})
        )

    def load_providers(self) -> Dict[str, ProviderConfig]:
        """
        加载数据提供者配置

        Returns:
            {提供者名称: ProviderConfig} 字典
        """
        providers = {}

        if self._is_new_structure:
            provider_configs = self._load_directory_configs('data/providers')
            for name, cfg in provider_configs.items():
                providers[name] = ProviderConfig(
                    name=name,
                    enabled=cfg.get('enabled', True),
                    priority=cfg.get('priority', 99),
                    config=cfg
                )

        self.logger.info(f"已加载 {len(providers)} 个数据提供者配置")
        return providers

    def load_calendar(self) -> Dict[str, Any]:
        """
        加载交易日历配置

        Returns:
            日历配置字典
        """
        if self._is_new_structure:
            return self._load_yaml('data/calendar.yaml')
        return {}

    # ========================================================================
    # Pipeline 组件配置加载
    # ========================================================================

    def load_pipeline_components(self, category: str) -> Dict[str, Dict[str, PipelineConfig]]:
        """
        加载指定类别的 Pipeline 组件配置

        Args:
            category: 类别 ('backtest' 或 'factors')

        Returns:
            {组件类型: {组件名称: PipelineConfig}}
        """
        components = {}

        if self._is_new_structure:
            if category == 'backtest':
                dirs = ['selectors', 'allocators', 'capital']
                base_path = 'backtest/pipeline'
            elif category == 'factors':
                dirs = ['combiners', 'standardizers', 'rolling']
                base_path = 'factors/pipeline'
            else:
                return components

            for component_type in dirs:
                filepath = f'{base_path}/{component_type}.yaml'
                config_dict = self._load_yaml(filepath)

                components[component_type] = {}
                for name, cfg in config_dict.items():
                    components[component_type][name] = PipelineConfig(
                        name=name,
                        params=cfg.get('params', {}),
                        description=cfg.get('description', ''),
                        enabled=cfg.get('enabled', True)
                    )

        return components

    # ========================================================================
    # 触发器配置加载
    # ========================================================================

    def load_triggers(self) -> Dict[str, Dict[str, Any]]:
        """
        加载所有触发器配置

        Returns:
            {触发器类型: 配置字典}
        """
        triggers = {}

        if self._is_new_structure:
            trigger_configs = self._load_directory_configs('backtest/triggers')
            for filename, cfg in trigger_configs.items():
                triggers.update(cfg)

        return triggers

    # ========================================================================
    # 股票池配置加载
    # ========================================================================

    def load_universe(self) -> List[str]:
        """
        加载股票池配置

        Returns:
            股票代码列表
        """
        config_dict = self._load_yaml('universe.yaml')

        universe = config_dict.get('symbols', [])
        self.logger.info(f"已加载股票池，共 {len(universe)} 只股票")

        return universe

    def load_universe_full(self) -> Dict[str, Any]:
        """
        加载完整的股票池配置

        Returns:
            完整配置字典
        """
        return self._load_yaml('universe.yaml')

    # ========================================================================
    # 全量加载
    # ========================================================================

    def load_all(self) -> Dict[str, Any]:
        """
        加载所有配置

        Returns:
            包含所有配置的字典
        """
        return {
            'factors': self.load_factors(),
            'strategies': self.load_strategies(),
            'backtest': self.load_backtest(),
            'factor_analysis': self.load_factor_analysis(),
            'data': self.load_data(),
            'providers': self.load_providers(),
            'universe': self.load_universe(),
            'pipeline': {
                'backtest': self.load_pipeline_components('backtest'),
                'factors': self.load_pipeline_components('factors'),
            },
            'triggers': self.load_triggers(),
        }


# ============================================================================
# 便捷函数
# ============================================================================

def load_config(config_dir: Optional[str] = None) -> ConfigLoader:
    """
    创建配置加载器（便捷函数）

    Args:
        config_dir: 配置目录路径

    Returns:
        ConfigLoader 实例
    """
    return ConfigLoader(config_dir)


def get_factor_config(factor_name: str,
                     config_dir: Optional[str] = None) -> Optional[FactorConfig]:
    """
    获取指定因子的配置

    Args:
        factor_name: 因子名称
        config_dir: 配置目录

    Returns:
        FactorConfig 实例，如果不存在返回 None
    """
    loader = ConfigLoader(config_dir)
    factors = loader.load_factors()
    return factors.get(factor_name)


def get_strategy_config(strategy_name: str,
                       config_dir: Optional[str] = None) -> Optional[StrategyConfig]:
    """
    获取指定策略的配置

    Args:
        strategy_name: 策略名称
        config_dir: 配置目录

    Returns:
        StrategyConfig 实例，如果不存在返回 None
    """
    loader = ConfigLoader(config_dir)
    strategies = loader.load_strategies()
    return strategies.get(strategy_name)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 测试加载
    loader = ConfigLoader()

    print("\n" + "="*60)
    print("测试配置加载")
    print("="*60)

    # 加载所有配置
    all_configs = loader.load_all()

    print(f"\n因子配置数量: {len(all_configs['factors'])}")
    print(f"策略配置数量: {len(all_configs['strategies'])}")
    print(f"股票池大小: {len(all_configs['universe'])}")
    print(f"数据提供者数量: {len(all_configs['providers'])}")
    print(f"回测配置: {all_configs['backtest']}")
    print(f"因子分析配置: {all_configs['factor_analysis']}")
    print(f"数据配置: {all_configs['data']}")

    print("="*60 + "\n")
