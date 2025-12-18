"""
因子分析脚本

运行因子计算和分析流程。
"""

import os
import sys
import logging
import datetime
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 默认配置（仅作为后备，优先使用配置文件）
DEFAULT_CONFIG = {
    'start_date': '2024-01-01',
    'end_date': '2024-12-31',
    'forward_return_periods': [1, 5, 10, 20, 30],
    'benchmark': '600519',
    'db_path': './database/quant_data.db',
    'output_dir': 'output/factor_reports',
    'skip_data_preparation': True,
    'factor_calc_processes': 8,
    'strategy_name': 'EqualWeights',
    'standardizer': 'ZScore',
}


def _get_config_value(factor_analysis_config, backtest_config, key, default):
    """从配置中获取值，优先使用 factor_analysis_config"""
    if hasattr(factor_analysis_config, key):
        val = getattr(factor_analysis_config, key)
        if val is not None:
            return val
    if hasattr(backtest_config, key):
        val = getattr(backtest_config, key)
        if val is not None:
            return val
    return default


def run_factor_analysis(config_loader):
    """
    执行因子分析

    Args:
        config_loader: 配置加载器实例
    """
    logger.info("=" * 60)
    logger.info("开始因子分析流程...")
    logger.info("=" * 60)

    # =========================================================================
    # 1. 加载配置
    # =========================================================================
    logger.info("\n--- 步骤 1: 加载配置 ---")

    factors_config = config_loader.load_factors()
    strategies_config = config_loader.load_strategies()
    universe = config_loader.load_universe()
    backtest_config = config_loader.load_backtest()
    factor_analysis_config = config_loader.load_factor_analysis()
    data_config = config_loader.load_data()

    # 从配置中获取参数（优先使用因子分析配置，其次回测配置，最后默认值）
    START_DATE = factor_analysis_config.start_date or backtest_config.start_date
    END_DATE = factor_analysis_config.end_date or backtest_config.end_date
    BENCHMARK = factor_analysis_config.benchmark or backtest_config.benchmark or DEFAULT_CONFIG['benchmark']
    DB_PATH = data_config.database.get('db_path', DEFAULT_CONFIG['db_path'])
    OUTPUT_DIR = factor_analysis_config.output.get('dir', DEFAULT_CONFIG['output_dir'])
    FORWARD_RETURN_PERIODS = factor_analysis_config.forward_return_periods or DEFAULT_CONFIG['forward_return_periods']
    DEFAULT_STRATEGY = factor_analysis_config.default_strategy
    factor_calc_cfg = factor_analysis_config.factor_calculation
    SKIP_DATA_PREPARATION = factor_calc_cfg.get('skip_data_preparation', DEFAULT_CONFIG['skip_data_preparation'])
    FACTOR_CALC_PROCESSES = factor_calc_cfg.get('num_processes', DEFAULT_CONFIG['factor_calc_processes'])

    # 检查是否启用全股票模式
    if data_config.use_all_stocks:
        logger.info("🎯 启用全股票模式，将从缓存数据库获取全部股票代码...")

        # 直接从缓存数据库 quant_data.db 获取股票列表，实现模块分离
        import sqlite3

        try:
            conn = sqlite3.connect(DB_PATH)

            # 查询缓存数据库中的所有股票代码
            query = f"""
                SELECT DISTINCT code
                FROM stock_daily_prices
                WHERE date BETWEEN '{START_DATE}' AND '{END_DATE}'
            """
            cursor = conn.execute(query)
            all_symbols = [row[0] for row in cursor.fetchall()]

            if all_symbols:
                universe = all_symbols
                logger.info(f"[全股票模式] ✅ 从缓存数据库获取成功！")
                logger.info(f"  - 股票数量: {len(universe)}")

                # 获取数据统计信息
                stats_query = f"""
                    SELECT COUNT(*) as total_records,
                           COUNT(DISTINCT date) as date_count
                    FROM stock_daily_prices
                    WHERE date BETWEEN '{START_DATE}' AND '{END_DATE}'
                """
                stats_cursor = conn.execute(stats_query)
                stats = stats_cursor.fetchone()
                logger.info(f"  - 总记录数: {stats[0]}")
                logger.info(f"  - 日期数量: {stats[1]}")
            else:
                error_msg = (
                    "全股票模式启用但缓存数据库中没有数据。请检查：\n"
                    f"1. 缓存数据库路径是否正确: {DB_PATH}\n"
                    "2. 是否已运行 --download_data 模式下载数据\n"
                    f"3. 日期范围是否正确: {START_DATE} ~ {END_DATE}"
                )
                logger.critical(error_msg)
                raise RuntimeError(error_msg)

            conn.close()
            logger.info(f"✅ 全股票模式成功，已从缓存数据库获取股票池: {len(universe)} 只股票")

        except sqlite3.Error as e:
            error_msg = (
                f"从缓存数据库获取股票列表失败: {e}\n"
                f"请确保缓存数据库存在: {DB_PATH}"
            )
            logger.critical(error_msg)
            raise RuntimeError(error_msg)
    else:
        logger.info(f"📋 使用配置文件股票池，共 {len(universe)} 只股票")

    # 获取启用的因子（配置加载器已经处理了白名单机制）
    enabled_factors = factors_config

    logger.info(f"已加载 {len(enabled_factors)} 个启用的因子配置")
    logger.info(f"已加载 {len(strategies_config)} 个策略配置")
    logger.info(f"默认策略: {factor_analysis_config.default_strategy}")
    logger.info(f"最终股票池大小: {len(universe)}")
    
    # 显示可用策略列表
    if strategies_config:
        available_strategies = list(strategies_config.keys())
        logger.info(f"可用策略: {available_strategies}")

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # 2. 初始化数据管理器
    # =========================================================================
    logger.info("\n--- 步骤 2: 初始化数据管理器 ---")

    try:
        from data import DataProviderManager
        from data.providers import SQLiteDataProvider
    except ImportError as e:
        logger.error(f"导入数据模块失败: {e}")
        logger.info("尝试从 old_code 导入...")
        from old_code.data.data_manager import DataProviderManager
        from old_code.data.data_providers import SQLiteDataProvider

    # 数据源配置 - 使用统一的配置处理逻辑
    DATA_PROVIDERS_CONFIG = []
    
    # 重新加载providers_config以确保可用
    if 'providers_config' not in locals():
        providers_config = config_loader.load_providers()
    
    # 处理SQLite数据提供者配置 - 支持多个SQLite数据源
    for provider_name in providers_config:
        provider_config = providers_config[provider_name]
        
        # 支持sqlite, sqlite_jy, sqlite_csmr等所有SQLite提供者
        if provider_name.startswith('sqlite') and provider_config.enabled:
            logger.info(f"处理SQLite数据提供者: {provider_name}")
            
            # 应用与全股票模式相同的配置处理逻辑
            provider_kwargs = provider_config.config.copy()
            
            if 'connection' in provider_kwargs and 'tables' in provider_kwargs:
                conn_config = provider_kwargs['connection']
                tables_config = provider_kwargs['tables']
                daily_config = tables_config.get('daily', {})
                
                processed_kwargs = {
                    'db_path': conn_config.get('db_path'),
                    'table_name': daily_config.get('table_name'),
                    'column_mapping': daily_config.get('column_mapping', {}),
                    'date_format': daily_config.get('date_format', '%Y-%m-%d')
                }
                
                # 使用名称标识符作为元组的第一个元素
                DATA_PROVIDERS_CONFIG.append((provider_name, SQLiteDataProvider, processed_kwargs))
                logger.info(f"✅ {provider_name}数据提供者配置已从配置文件加载: {processed_kwargs}")
            else:
                logger.error(f"❌ {provider_name}配置格式不正确，无法创建数据提供者")
        elif provider_name.startswith('sqlite'):
            logger.warning(f"❌ {provider_name}配置未启用")
    
    # 处理其他数据提供者（Tushare, Akshare等）
    for provider_name in providers_config:
        provider_config = providers_config[provider_name]
        
        if not provider_name.startswith('sqlite') and provider_config.enabled:
            try:
                if provider_name == 'tushare':
                    from data.providers import TushareDataProvider
                    provider_class = TushareDataProvider
                elif provider_name == 'akshare':
                    from data.providers import AkshareDataProvider
                    provider_class = AkshareDataProvider
                else:
                    logger.warning(f"未知的数据提供者: {provider_name}")
                    continue
                
                # 处理特殊配置
                provider_kwargs = provider_config.config.copy()
                
                if provider_name == 'tushare' and 'token' not in provider_kwargs:
                    # 如果是tushare且没有配置token，尝试从环境变量获取
                    provider_kwargs['token'] = os.getenv('TUSHARE_TOKEN')
                    if not provider_kwargs['token']:
                        logger.warning("Tushare需要token配置，跳过...")
                        continue
                
                # 使用名称标识符作为元组的第一个元素
                DATA_PROVIDERS_CONFIG.append((provider_name, provider_class, provider_kwargs))
                logger.info(f"✅ {provider_name}数据提供者配置已加载")
                
            except Exception as e:
                logger.error(f"加载{provider_name}提供者失败: {e}")
    
    # 如果没有配置成功，抛出异常
    if not DATA_PROVIDERS_CONFIG:
        error_msg = (
            "无法创建任何数据提供者。请检查：\n"
            "1. 配置文件 configs/data/providers/*.yaml 是否正确\n"
            "2. 数据提供者配置是否启用\n"
            "3. 配置格式是否正确"
        )
        logger.critical(error_msg)
        raise RuntimeError(error_msg)

    data_manager = DataProviderManager(
        provider_configs=DATA_PROVIDERS_CONFIG,
        symbols=universe,
        start_date=START_DATE,
        end_date=END_DATE,
        db_path=DB_PATH,
        num_checker_threads=16,
        num_downloader_threads=16,
        batch_size=200
    )

    # 确保基准在股票池中
    if BENCHMARK not in data_manager.symbols:
        data_manager.symbols.append(BENCHMARK)

    if not SKIP_DATA_PREPARATION:
        logger.info("模式: 完整数据准备 (检查、下载、写入)...")
        data_manager.prepare_data_for_universe()
    else:
        logger.info("已跳过数据下载流程")

    # 获取基准数据
    logger.info(f"获取基准 '{BENCHMARK}' 数据...")
    benchmark_df = data_manager.get_dataframe(BENCHMARK, columns=['close'])

    # 准备活跃股票池
    active_universe = data_manager.symbols.copy()
    if BENCHMARK in active_universe:
        active_universe.remove(BENCHMARK)

    # =========================================================================
    # 3. 计算未来收益率
    # =========================================================================
    logger.info("\n--- 步骤 3: 预计算未来收益率 ---")

    future_returns_df = data_manager.calculate_universe_forward_returns(
        universe=active_universe,
        forward_return_periods=FORWARD_RETURN_PERIODS
    )

    if future_returns_df is None or future_returns_df.empty:
        logger.critical("致命错误: 未能计算出未来收益率，程序终止。")
        return

    future_returns_df.set_index('date', inplace=True)
    logger.info(f"未来收益率计算完成，共 {len(future_returns_df)} 条记录")

    # =========================================================================
    # 4. 因子分类与路由
    # =========================================================================
    logger.info("\n--- 步骤 4: 因子分类与路由 ---")

    simple_factors_batch = []
    complex_factors_batch = []

    for factor_name, factor_cfg in enabled_factors.items():
        if factor_cfg.category == 'simple':
            simple_factors_batch.append((factor_name, factor_cfg.params))
        elif factor_cfg.category == 'complex':
            complex_factors_batch.append(factor_name)
        else:
            logger.warning(f"跳过: 因子 '{factor_name}' 的 category '{factor_cfg.category}' 无效。")

    logger.info(f"简单因子 (多进程计算): {[f[0] for f in simple_factors_batch]}")
    logger.info(f"复合因子 (全量表计算): {complex_factors_batch}")

    # 收集所有需要的数据列
    all_required_columns = set()
    for factor_name in enabled_factors:
        required = enabled_factors[factor_name].required_columns
        all_required_columns.update(required)

    sorted_cols = sorted(list(all_required_columns))
    logger.info(f"本次运行的数据列需求: {sorted_cols}")

    # =========================================================================
    # 5. 执行因子计算
    # =========================================================================
    all_factors_dfs = {}

    logger.info("\n--- 步骤 5: 执行因子计算 ---")

    # --- 分支 A: 执行简单因子 ---
    if simple_factors_batch:
        try:
            from factors.analysis import FactorCalculator
        except ImportError:
            from old_code.factor_analysis.factor_calculator import FactorCalculator

        for factor_name, user_params in simple_factors_batch:
            factor_cfg = enabled_factors[factor_name]
            required_cols = factor_cfg.required_columns

            logger.info(f"[Simple] 计算因子: {factor_name}...")

            try:
                calculator = FactorCalculator(
                    provider_configs=data_manager.provider_configs,
                    db_path=DB_PATH,
                    universe=active_universe,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    factor_name=factor_name,
                    factor_params=user_params,
                    num_threads=FACTOR_CALC_PROCESSES,
                    required_columns=required_cols
                )

                factor_data_df = calculator.calculate_factor()

                if not factor_data_df.empty:
                    factor_series = factor_data_df.set_index(
                        'asset', append=True)['factor_value']
                    factor_series.name = factor_name
                    all_factors_dfs[factor_name] = factor_series.sort_index()
                    logger.info(f"  > 完成: {factor_name}")
            except Exception as e:
                logger.error(f"计算因子 {factor_name} 失败: {e}", exc_info=True)
    else:
        logger.info("(无简单因子需要计算)")

    # --- 分支 B: 执行复合因子 ---
    if complex_factors_batch:
        logger.info(f"[Complex] 正在为复合因子加载宽表数据...")

        try:
            all_data_df = data_manager.get_all_data_for_universe(
                active_universe, required_columns=sorted_cols
            )

            if all_data_df is not None and not all_data_df.empty:
                from core.registry import registry
                
                for factor_name in complex_factors_batch:
                    if registry.exists('factors', factor_name):
                        logger.info(f"[Complex] 计算: {factor_name}...")
                        
                        try:
                            FactorClass = registry.get('factors', factor_name)
                            # 实例化因子类
                            factor_instance = FactorClass()
                            # 调用 calculate 方法
                            factor_series = factor_instance.calculate(all_data_df)
                            
                            if factor_series is not None:
                                factor_series.name = factor_name
                                all_factors_dfs[factor_name] = factor_series.sort_index()
                                logger.info(f"  > 完成: {factor_name}")
                        except Exception as e:
                            logger.error(f"计算复合因子 {factor_name} 失败: {e}")
                    else:
                        logger.warning(f"因子 {factor_name} 在注册表中未找到")
            else:
                logger.error("无法加载数据，跳过复合因子计算")
        except Exception as e:
            logger.error(f"加载宽表数据失败: {e}")
    else:
        logger.info("(无复合因子需要计算)")

    # =========================================================================
    # 6. 因子合并与分析
    # =========================================================================
    logger.info("\n--- 步骤 6: 因子合并与分析 ---")

    FACTOR_NAMES = list(all_factors_dfs.keys())
    logger.info(f"待合并因子: {FACTOR_NAMES}")

    if not FACTOR_NAMES:
        logger.warning("没有计算出任何因子，流程终止。")
        return

    # 导入标准化器
    try:
        from factors.pipeline.standardizers import CrossSectionalZScoreStandardizer
        STANDARDIZER = CrossSectionalZScoreStandardizer()
    except ImportError:
        try:
            from old_code.strategies.standardizers import CrossSectionalZScoreStandardizer
            STANDARDIZER = CrossSectionalZScoreStandardizer()
        except ImportError:
            logger.warning("无法导入标准化器，将跳过标准化")
            STANDARDIZER = None

    if len(FACTOR_NAMES) == 1:
        logger.info("单因子模式")
        final_factor_name = FACTOR_NAMES[0]
        combined_factors_df = all_factors_dfs[final_factor_name].to_frame()
    else:
        logger.info("步骤 6a: 合并因子数据...")

        # 处理重复索引
        for factor_name, factor_series in all_factors_dfs.items():
            if factor_series.index.duplicated().any():
                dup_count = factor_series.index.duplicated().sum()
                logger.warning(f"因子 {factor_name} 发现 {dup_count} 个重复索引，正在处理...")
                all_factors_dfs[factor_name] = factor_series[~factor_series.index.duplicated(keep='last')]

        # 合并因子
        combined_factors_df = pd.concat(
            all_factors_dfs.values(),
            axis=1,
            keys=all_factors_dfs.keys()
        )
        if isinstance(combined_factors_df.columns, pd.MultiIndex):
            combined_factors_df.columns = combined_factors_df.columns.droplevel(1)

        # 最终检查重复索引
        if combined_factors_df.index.duplicated().any():
            combined_factors_df = combined_factors_df[~combined_factors_df.index.duplicated(keep='last')]

        # 标准化
        if STANDARDIZER is not None:
            logger.info(f"步骤 6b: 执行全局截面标准化 ({STANDARDIZER.__class__.__name__})...")
            combined_factors_df = combined_factors_df.groupby(
                level='date',
                group_keys=False
            ).apply(lambda x: STANDARDIZER.standardize(x))
            logger.info("  > 所有因子已完成标准化处理")

        # 因子合成（使用配置策略）
        logger.info(f"步骤 6c: 因子合成（策略: {DEFAULT_STRATEGY}）...")

        # 从策略配置获取合成器名称
        if DEFAULT_STRATEGY in strategies_config:
            strategy_config = strategies_config[DEFAULT_STRATEGY]
            combiner_name = strategy_config.combiner
            logger.info(f"使用合成器: {combiner_name}")

            # 检查是否是滚动策略
            is_rolling = hasattr(strategy_config, 'rolling') and strategy_config.rolling is not None

            if is_rolling:
                # 滚动策略：使用滚动计算器
                logger.info("  > 模式: 动态滚动 (每日权重计算)")
                rolling_config = strategy_config.rolling
                calculator_type = rolling_config.get('calculator_type', 'Regression')

                # 准备滚动数据：合并因子值和未来收益
                all_data_merged = pd.merge(
                    combined_factors_df.reset_index(),
                    future_returns_df.reset_index(),
                    on=['date', 'asset'],
                    how='inner'
                ).set_index(['date', 'asset']).sort_index()

                # 创建滚动计算器
                try:
                    if calculator_type == 'Regression':
                        from factors.pipeline.combiners.rolling import RollingRegressionCalculator
                        roller = RollingRegressionCalculator(
                            factor_names=FACTOR_NAMES,
                            rolling_window_days=rolling_config.get('rolling_window_days', 90),
                            rebalance_frequency=rolling_config.get('rebalance_frequency', 'D'),
                            target_return_period=rolling_config.get('target_return_period', 30)
                        )
                    elif calculator_type == 'ICIR':
                        from factors.pipeline.combiners.rolling import RollingICIRCalculator
                        roller = RollingICIRCalculator(
                            factor_names=FACTOR_NAMES,
                            rolling_window_days=rolling_config.get('rolling_window_days', 90),
                            rebalance_frequency=rolling_config.get('rebalance_frequency', 'D'),
                            factor_weight_config=rolling_config.get('factor_weighting_config', {}),
                            forward_return_periods=FORWARD_RETURN_PERIODS
                        )
                    else:
                        raise ValueError(f"未知的 calculator_type: {calculator_type}")

                    # 执行滚动合成
                    composite_factor_series = roller.calculate_composite_factor(all_data_merged)
                    composite_factor_series.name = 'factor_value'
                    final_factor_name = f"Composite_{DEFAULT_STRATEGY}"
                    logger.info(f"  > 完成: {DEFAULT_STRATEGY} 策略合成")

                except Exception as e:
                    logger.warning(f"滚动计算器执行失败，回退到等权合成: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    composite_factor_series = combined_factors_df.mean(axis=1)
                    composite_factor_series.name = 'factor_value'
                    final_factor_name = "Composite_EqualWeights_Fallback"
            else:
                # 静态策略：使用 combiner
                logger.info("  > 模式: 静态合成")
                try:
                    from core.registry import get_combiner
                    combiner_class = get_combiner(combiner_name)
                    combiner = combiner_class(**strategy_config.combiner_params)

                    # 执行因子合成
                    composite_factor_series = combiner.combine(combined_factors_df)
                    composite_factor_series.name = 'factor_value'
                    final_factor_name = f"Composite_{DEFAULT_STRATEGY}"

                    logger.info(f"  > 完成: {DEFAULT_STRATEGY} 策略合成")

                except Exception as e:
                    logger.warning(f"合成器 {combiner_name} 执行失败，回退到等权合成: {e}")
                    composite_factor_series = combined_factors_df.mean(axis=1)
                    composite_factor_series.name = 'factor_value'
                    final_factor_name = "Composite_EqualWeights_Fallback"
        else:
            logger.warning(f"策略 '{DEFAULT_STRATEGY}' 未找到配置，回退到等权合成")
            composite_factor_series = combined_factors_df.mean(axis=1)
            composite_factor_series.name = 'factor_value'
            final_factor_name = "Composite_EqualWeights_Fallback"

        combined_factors_df = composite_factor_series.to_frame()

    # =========================================================================
    # 7. 生成报告
    # =========================================================================
    if not combined_factors_df.empty:
        logger.info("\n--- 步骤 7: 生成报告 ---")

        try:
            from factors.analysis import FactorReport
        except ImportError:
            from old_code.factor_analysis.factor_report import FactorReport

        # 准备数据用于报告
        if isinstance(combined_factors_df.index, pd.MultiIndex):
            index_data = {
                'date': combined_factors_df.index.get_level_values('date').values,
                'asset': combined_factors_df.index.get_level_values('asset').values
            }
        else:
            logger.warning("索引格式不符合预期，尝试重建...")
            index_data = {
                'date': combined_factors_df.index.values,
                'asset': ['unknown'] * len(combined_factors_df)
            }

        factor_df_for_merge = pd.DataFrame(index_data)
        factor_df_for_merge['factor_value'] = combined_factors_df.iloc[:, -1].values

        final_factor_data_df = pd.merge(
            factor_df_for_merge,
            future_returns_df.reset_index(),
            on=['date', 'asset'],
            how='inner'
        )
        final_factor_data_df.set_index('date', inplace=True)

        final_report_df = final_factor_data_df.copy()
        final_report_df.dropna(subset=['factor_value'], inplace=True)

        if not final_report_df.empty:
            report_generator = FactorReport(
                factor_name=final_factor_name,
                factor_data=final_report_df,
                forward_return_periods=FORWARD_RETURN_PERIODS,
                benchmark_data=benchmark_df if benchmark_df is not None else pd.DataFrame()
            )

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = os.path.join(
                OUTPUT_DIR,
                f"{timestamp}_factor_report.html"
            )

            logger.info(f"生成 HTML 报告: {output_filename}")
            report_generator.generate_html_report(output_filename)
            logger.info(f"报告已保存到: {output_filename}")
        else:
            logger.warning("最终因子数据为空，无法生成报告。")

    logger.info("\n" + "=" * 60)
    logger.info("因子分析流程执行完毕")
    logger.info("=" * 60)


if __name__ == '__main__':
    from core.config import load_config
    config_loader = load_config()
    run_factor_analysis(config_loader)
