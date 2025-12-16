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

    # 获取启用的因子
    enabled_factors = {
        name: cfg for name, cfg in factors_config.items() if cfg.enabled
    }

    logger.info(f"已加载 {len(enabled_factors)} 个启用的因子配置")
    logger.info(f"已加载 {len(strategies_config)} 个策略配置")
    logger.info(f"股票池大小: {len(universe)}")

    # 从配置中获取参数（优先使用因子分析配置，其次回测配置，最后默认值）
    START_DATE = factor_analysis_config.start_date or backtest_config.start_date
    END_DATE = factor_analysis_config.end_date or backtest_config.end_date
    BENCHMARK = factor_analysis_config.benchmark or backtest_config.benchmark or DEFAULT_CONFIG['benchmark']
    DB_PATH = data_config.database.get('db_path', DEFAULT_CONFIG['db_path'])
    OUTPUT_DIR = factor_analysis_config.output.get('dir', DEFAULT_CONFIG['output_dir'])
    FORWARD_RETURN_PERIODS = factor_analysis_config.forward_return_periods or DEFAULT_CONFIG['forward_return_periods']
    factor_calc_cfg = factor_analysis_config.factor_calculation
    SKIP_DATA_PREPARATION = factor_calc_cfg.get('skip_data_preparation', DEFAULT_CONFIG['skip_data_preparation'])
    FACTOR_CALC_PROCESSES = factor_calc_cfg.get('num_processes', DEFAULT_CONFIG['factor_calc_processes'])

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

    # 数据源配置
    DATA_PROVIDERS_CONFIG = [
        (
            SQLiteDataProvider,
            {
                'db_path': './database/JY_database/sqlite/JY_database.sqlite',
                'table_name': 'JY_t_price_daily'
            }
        ),
    ]

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
                # 尝试导入复合因子注册表
                try:
                    from old_code.factor_analysis.factors_complex import COMPLEX_FACTOR_REGISTRY
                except ImportError:
                    logger.warning("无法导入复合因子注册表，跳过复合因子计算")
                    COMPLEX_FACTOR_REGISTRY = {}

                for factor_name in complex_factors_batch:
                    if factor_name in COMPLEX_FACTOR_REGISTRY:
                        logger.info(f"[Complex] 计算: {factor_name}...")
                        factor_func = COMPLEX_FACTOR_REGISTRY[factor_name]

                        try:
                            factor_series = factor_func(all_data_df)
                            if factor_series is not None:
                                factor_series.name = factor_name
                                all_factors_dfs[factor_name] = factor_series.sort_index()
                                logger.info(f"  > 完成: {factor_name}")
                        except Exception as e:
                            logger.error(f"计算复合因子 {factor_name} 失败: {e}")
                    else:
                        logger.warning(f"因子 {factor_name} 在 COMPLEX_FACTOR_REGISTRY 中未找到")
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

        # 因子合成（默认等权）
        logger.info("步骤 6c: 因子合成（等权）...")
        composite_factor_series = combined_factors_df.mean(axis=1)
        composite_factor_series.name = 'factor_value'
        final_factor_name = "Composite_EqualWeights"

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
