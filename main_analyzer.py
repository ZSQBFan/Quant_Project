# main_analyzer.py (已重构 - 修正多进程调用)

import os
import logging
import pandas as pd
from tqdm import tqdm
import sys

# ==============================================================================
# 1. 策略分析“控制面板” (Strategy Analysis "Control Panel")
# ==============================================================================
#
#   欢迎使用！
#   您几乎所有的【高频】自定义配置都可以在这个版块完成。
#   (对于【低频】的策略定义，请转到 `strategy_configs.py` 文件)
#
# ==============================================================================

# --- 1a. 核心策略选择 (Core Strategy Selection) ---
#
#   这是【最重要】的策略选择点。所有复杂的配置 (滚动周期、固定权重等)
#   都已封装在 `strategy_configs.py` 文件中。
#
from strategy_configs import STRATEGY_REGISTRY

# 【【请在这里选择您的策略名称 (从 strategy_configs.py 复制)】】
# STRATEGY_NAME = "RollingICIR"
# STRATEGY_NAME = "RollingRegression"
# STRATEGY_NAME = "FixedWeights"
# STRATEGY_NAME = "EqualWeights"
# STRATEGY_NAME = "DynamicSignificance"
STRATEGY_NAME = "AI_Periodic_Retrain"

# --- (自动加载配置) ---
if STRATEGY_NAME not in STRATEGY_REGISTRY:
    # (这个日志会在 setup_logging 之前，可能无法被捕获，但 raise 会终止程序)
    raise ValueError(f"策略 '{STRATEGY_NAME}' 未在 strategy_configs.py 中注册。")
# 自动加载所选策略的完整配置对象
STRATEGY_CONFIG = STRATEGY_REGISTRY[STRATEGY_NAME]

# --- 1b. 因子选择 (Factors to Analyze) ---
#
#   【【重要配置】】
#   您希望在本次分析中运行哪些【基础因子 (Type 1)】？
#
FACTORS_TO_ANALYZE = [
    # ('RSI', {
    #     'rsi_period': 22
    # }),
    # ('BollingerBands', {
    #     'period': 30
    # }),
    # ('ADXDMI', {
    #     'period': 14,
    #     'trend_threshold': 22
    # }),
    # ('Momentum', {
    #     'period': 20
    # }),
    # ('Reversal20D', {
    #     'period': 40,
    #     'decay': 20
    # }),
]

# --- 1c. 复合因子选择 (Complex Factors) ---
#
#   【【高级配置】】
#   您希望在 "Type 1" 基础因子计算【之后】运行哪些 "Type 2" 复合因子？
#
from factor_analysis.factors_complex import COMPLEX_FACTOR_REGISTRY

COMPLEX_FACTORS_TO_RUN = [
    "IndNeu_Momentum",
    "IndNeu_Reversal20D",
    "IndNeu_VolumeCV",
    # "IndNeu_ADXDMI",
    # "MktNeu_RSI", # 示例: 如果您在 `factors_complex.py` 中定义了它
]

# --- 1d. 截面数据配置 (Cross-Sectional Data) ---
#
#   【【高级配置】】
#   您是否需要加载【额外】的截面数据 (例如 行业、市值)？
#
LOAD_INDUSTRY_DATA = True
# (如果不需要行业数据，请改为 False)

# --- 1e. 标准化器 (Standardizer) ---
#
#   【【重要配置】】
#   您希望如何对因子值进行【截面标准化】？
#
from strategies.standardizers import (CrossSectionalZScoreStandardizer,
                                      NoStandardizer,
                                      CrossSectionalQuantileStandardizer)
# 【【请在这里三选一】】
STANDARDIZER_CLASS = CrossSectionalZScoreStandardizer
# STANDARDIZER_CLASS = CrossSectionalQuantileStandardizer
# STANDARDIZER_CLASS = NoStandardizer

# ==============================================================================
# 2. 基础回测与路径配置 (Basic Backtest & Path Settings)
# ==============================================================================
#
# 【【【重要】】】数据下载与写入开关：
#
#   当您只是修改因子计算 (factors.py) 或合成逻辑 (factor_combiner.py)，
#   而股票池和时间范围不变时，请将此项设为 True。
#   这将跳过耗时的数据检查和下载流程，直接使用数据库中的现有数据。
#
SKIP_DATA_PREPARATION = True
# SKIP_DATA_PREPARATION = False  # (正常运行时设为 False)
#
# ==============================================================================

# --- 2a. 回测时间与收益周期 ---
START_DATE = '2018-01-01'
END_DATE = '2020-12-31'
FORWARD_RETURN_PERIODS = [1, 5, 10, 30, 60]  # 必须包含 1d 中配置的所有周期

# --- 2b. 基准与股票池 ---
BENCHMARK = '600519'  # 用于报告对比
from universe_config import UNIVERSE  # 导入您的股票池

# 【【【新增】】】: Type 1 因子计算【进程数】
# (这在 factor_calculator.py 中被用作 max_workers)
FACTOR_CALC_PROCESSES = 16  # (根据您的 CPU 核心数调整，例如 8, 16)

# --- 2c. 路径配置 ---
LOG_DIR = "logs"
OUTPUT_DIR = "factor_reports"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

BACKTEST_DB_PATH = './database/quant_data.db'

# --- 2d. 数据源配置 ---
from data.data_providers import SQLiteDataProvider

DATA_PROVIDERS_CONFIG = [
    (SQLiteDataProvider, {
        'db_path': './database/JY_database/sqlite/JY_database.sqlite',
        'table_name': 'JY_t_price_daily'
    }),
]

# ==============================================================================
#
#                     --- 核心程序开始 (Core Logic Starts) ---
#                     --- (!!!) 通常你不需要修改以下内容 (!!!) ---
#
# ==============================================================================

# 导入所有必需的分析模块
from data.data_manager import DataProviderManager
from factor_analysis.factor_calculator import FactorCalculator
from factor_analysis.factor_report import FactorReport
from logger.logger_config import setup_logging

if __name__ == '__main__':

    # =====================
    # 0. 初始化与配置校验
    # =====================
    # (setup_logging 必须在最开始调用)
    setup_logging(log_dir=LOG_DIR, log_prefix='factor_analysis')
    logging.info(f"\n{'='*60}\n--- 步骤 0: 初始化与配置校验 ---\n{'='*60}")
    logging.info("🏁 因子分析程序启动...")

    # 实例化标准化器
    STANDARDIZER = STANDARDIZER_CLASS()
    logging.info(f"✅ 标准化器已加载: {STANDARDIZER.__class__.__name__}")
    logging.info(f"⚙️ 正在加载策略: {STRATEGY_NAME}")
    logging.info("✅ 策略配置加载完毕。")

    # =====================
    # 1. 初始化数据管理器并准备数据
    # =====================
    logging.info(f"\n{'='*60}\n--- 步骤 1: 准备数据 ---\n{'='*60}")
    logging.info("⚙️ 正在初始化 DataProviderManager...")
    data_manager = DataProviderManager(provider_configs=DATA_PROVIDERS_CONFIG,
                                       symbols=UNIVERSE,
                                       start_date=START_DATE,
                                       end_date=END_DATE,
                                       db_path=BACKTEST_DB_PATH,
                                       num_checker_threads=16,
                                       num_downloader_threads=16,
                                       batch_size=200)

    # 将基准添加到下载列表
    if BENCHMARK not in data_manager.symbols:
        data_manager.symbols.append(BENCHMARK)
        logging.info(f"  > 已将基准 {BENCHMARK} 添加到数据管理器任务列表。")

    # 根据配置决定是否跳过数据准备
    if not SKIP_DATA_PREPARATION:
        logging.info("⚙️ 模式: 完整数据准备 (检查、下载、写入)...")
        data_manager.prepare_data_for_universe()
        logging.info("✅ 完整数据准备流程 (ETL) 已完成。")
    else:
        logging.info(
            "🟡 【【跳过】】: 已按配置 (SKIP_DATA_PREPARATION=True) 跳过数据检查与下载流程。")

    # 获取基准数据用于报告对比
    logging.info(f"⚙️ 正在获取基准 '{BENCHMARK}' 数据用于报告对比...")
    benchmark_df = data_manager.get_dataframe(BENCHMARK)
    if benchmark_df is None or benchmark_df.empty:
        logging.warning(f"⚠️ 警告: 未能获取到基准 '{BENCHMARK}' 的数据。")
    else:
        logging.info(f"✅ 成功获取基准 '{BENCHMARK}' 数据。")

    # 定义用于因子计算的有效股票池（排除基准）
    active_universe = data_manager.symbols.copy()
    if BENCHMARK in active_universe:
        active_universe.remove(BENCHMARK)
        logging.info(f"  > 已从因子计算池中移除基准 {BENCHMARK}。")

    # ==============================================================================
    # 【【【【【【 新增步骤 1.5: 统一计算未来收益率 】】】】】】
    # 这是架构优化的核心：将收益率计算与因子计算完全分离。
    # 无论后续运行何种因子，收益率（“答案”）都预先在这里一次性计算好。
    # ==============================================================================
    logging.info(f"\n{'='*60}\n--- 步骤 1.5: 预计算所有未来收益率 ---\n{'='*60}")
    future_returns_df = data_manager.calculate_universe_forward_returns(
        universe=active_universe,
        forward_return_periods=FORWARD_RETURN_PERIODS)
    if future_returns_df is None or future_returns_df.empty:
        logging.critical("⛔ 致命错误: 未能计算出未来收益率，后续分析无法进行。程序终止。")
        sys.exit()

    # 将 date 设为索引以优化后续合并性能
    future_returns_df.set_index('date', inplace=True)
    logging.info(f"✅ 未来收益率预计算完成。")

    # =====================
    # 2. 计算所有因子的原始值
    # =====================
    all_factors_dfs = {}
    all_data_df = None

    logging.info(f"\n{'='*60}\n--- 步骤 2: 计算所有指定因子的原始值 ---\n{'='*60}")

    # --- 步骤 2a: 计算基础因子 (Type 1) ---
    if not FACTORS_TO_ANALYZE:
        logging.info("ℹ️ (跳过: 未在 1b 中配置基础因子)")
    else:
        for factor_name, factor_params in FACTORS_TO_ANALYZE:
            logging.info(
                f"⚙️ 正在启动 (Type 1) 计算器: {factor_name} (参数: {factor_params})..."
            )
            # 【【【修改】】】: FactorCalculator 不再需要 forward_return_periods 参数
            calculator = FactorCalculator(
                provider_configs=data_manager.provider_configs,
                db_path=BACKTEST_DB_PATH,
                universe=active_universe,
                start_date=START_DATE,
                end_date=END_DATE,
                factor_name=factor_name,
                factor_params=factor_params,
                num_threads=FACTOR_CALC_PROCESSES)
            # 【【【修改】】】: 调用新的、更纯粹的 calculate_factor 方法
            factor_data_df = calculator.calculate_factor()

            if not factor_data_df.empty:
                # 【【【修改】】】: 不再有从这里获取 future_returns_df 的逻辑
                factor_series = factor_data_df.set_index(
                    'asset', append=True)['factor_value']
                factor_series.name = factor_name
                all_factors_dfs[factor_name] = factor_series.sort_index()
                logging.info(f"✅ 成功计算并存储 (Type 1) 因子: {factor_name}")

    # --- 步骤 2b: 计算复合因子 (Type 2) ---
    if not COMPLEX_FACTORS_TO_RUN:
        logging.info("ℹ️ (跳过: 未在 1c 中配置复合因子)")
    else:
        logging.info("⚙️ 正在准备 (Type 2) 复合因子计算所需的全量数据...")
        all_data_df = data_manager.get_all_data_for_universe(active_universe)

        if all_data_df is None:
            logging.error("❌ 错误: 无法加载复合因子所需的基础数据，已跳过。")
        else:
            if LOAD_INDUSTRY_DATA:
                logging.info("  > 正在加载并合并行业数据...")
                industry_df = data_manager.get_industry_mapping()
                if industry_df is not None:
                    # 【【【修复】】】: 采用正确的合并与索引重建流程，确保 all_data_df 结构正确
                    all_data_df = all_data_df.reset_index().merge(
                        industry_df, on='asset',
                        how='left').set_index(['date', 'asset']).sort_index()
                    all_data_df['industry'] = all_data_df.groupby(
                        level='asset')['industry'].ffill().bfill()
                    logging.info("  > ✅ 行业数据合并并重建索引完成。")
                else:
                    logging.warning("  > ⚠️ 警告: 未能从 'stock_kind' 加载行业数据。")

            for factor_name in COMPLEX_FACTORS_TO_RUN:
                if factor_name in COMPLEX_FACTOR_REGISTRY:
                    logging.info(f"⚙️ 正在计算 (Type 2) 复合因子: {factor_name}...")
                    factor_func = COMPLEX_FACTOR_REGISTRY[factor_name]
                    # 【【【修复】】】: 复合因子函数是独立的，只需 all_data_df
                    factor_series = factor_func(all_data_df)
                    factor_series.name = factor_name
                    all_factors_dfs[factor_name] = factor_series.sort_index()
                    logging.info(f"✅ 成功计算并存储 (Type 2) 因子: {factor_name}")

    # 【【【移除】】】: 之前为了修复bug而增加的“保险”补算逻辑已不再需要。

    # =====================
    # 3. 因子合并与分析
    # =====================
    final_factor_data_df = pd.DataFrame()
    final_factor_name = ""

    FACTOR_NAMES = list(all_factors_dfs.keys())
    logging.info(f"\n{'='*60}\n--- 步骤 3: 因子合并与分析 ---\n{'='*60}")
    logging.info(f"ℹ️ 即将合并的【所有】因子: {FACTOR_NAMES}")

    if not FACTOR_NAMES:
        logging.warning("⚠️ 没有任何因子被计算，分析流程终止。")
    elif len(FACTOR_NAMES) == 1:
        logging.info("ℹ️ 只有一个因子，直接进入报告生成阶段。")
        final_factor_name = FACTOR_NAMES[0]
        final_factor_series = all_factors_dfs[final_factor_name]
        combined_factors_df = final_factor_series.to_frame()
    else:
        # --- 多因子合成路径 ---
        logging.info("⚙️ 步骤 3a: 合并所有因子数据...")
        combined_factors_df = pd.concat(all_factors_dfs.values(),
                                        axis=1,
                                        keys=all_factors_dfs.keys())
        if isinstance(combined_factors_df.columns, pd.MultiIndex):
            combined_factors_df.columns = combined_factors_df.columns.droplevel(
                1)
        combined_factors_df = combined_factors_df[FACTOR_NAMES]
        logging.info(f"  > ✅ 成功合并 {len(FACTOR_NAMES)} 个因子。")

        # 核心逻辑分支：静态 vs 滚动
        if not STRATEGY_CONFIG.is_rolling():
            # 分支A: 静态权重逻辑
            logging.info(
                f"ℹ️ 模式: 静态合成 (策略: {STRATEGY_CONFIG.combiner_class.__name__})")
            combiner = STRATEGY_CONFIG.combiner_class(
                **STRATEGY_CONFIG.combiner_kwargs)
            logging.info(
                f"⚙️ 步骤 3b: 执行截面标准化 ({STANDARDIZER.__class__.__name__})...")
            standardized_factors_df = combined_factors_df.groupby(
                level='date').apply(lambda x: STANDARDIZER.standardize(x))
            logging.info("⚙️ 步骤 3c: 执行因子合成...")
            composite_factor_series = standardized_factors_df.groupby(
                level='date').apply(lambda x: combiner.combine(x))
            composite_factor_series.name = 'factor_value'
            final_factor_name = f"CompositeFactor_{STRATEGY_NAME}"
        else:
            # 分支B: 动态滚动权重逻辑
            logging.info(f"ℹ️ 模式: 动态滚动 (每日权重计算模式)")
            roller = STRATEGY_CONFIG.create_rolling_calculator(
                forward_return_periods=FORWARD_RETURN_PERIODS,
                factor_names=FACTOR_NAMES)
            logging.info("⚙️ 步骤 3c: 准备滚动数据 (合并因子与未来收益)...")
            all_data_merged = pd.merge(
                combined_factors_df.reset_index(),
                future_returns_df.reset_index(
                ),  # future_returns_df 的索引是 date, reset 后变为列
                on=['date', 'asset'],
                how='inner')
            all_data_merged.set_index(['date', 'asset'], inplace=True)
            all_data_merged.sort_index(inplace=True)
            logging.info(f"  > ✅ 滚动数据准备完毕，共 {len(all_data_merged)} 条合并记录。")
            composite_factor_series = roller.calculate_composite_factor(
                all_data_merged)
            composite_factor_series.name = 'factor_value'
            final_factor_name = f"CompositeFactor_{STRATEGY_NAME}_Rolling"

        combined_factors_df = composite_factor_series.to_frame()

    # --- 合并未来收益以生成报告 ---
    if not combined_factors_df.empty:
        final_factor_data_df = pd.merge(combined_factors_df.reset_index(),
                                        future_returns_df.reset_index(),
                                        on=['date', 'asset'],
                                        how='inner')
        final_factor_data_df.rename(
            columns={'factor_value': final_factor_name}, inplace=True)
        final_factor_data_df.set_index('date', inplace=True)

    # =====================
    # 4. 生成最终的因子分析报告
    # =====================
    if not final_factor_data_df.empty:
        logging.info(
            f"\n{'='*60}\n--- 步骤 4: 为最终因子 '{final_factor_name}' 生成分析报告 ---\n{'='*60}"
        )
        final_report_df = final_factor_data_df.rename(
            columns={final_factor_name: 'factor_value'})
        final_report_df.dropna(subset=['factor_value'], inplace=True)

        if final_report_df.empty:
            logging.warning(f"  > ⚠️ 警告: 最终因子 '{final_factor_name}' 数据在清理后为空。")
        else:
            logging.info(f"  > ✅ 最终因子数据准备完成，共 {len(final_report_df)} 条有效记录。")
            report_generator = FactorReport(
                factor_name=final_factor_name,
                factor_data=final_report_df,
                forward_return_periods=FORWARD_RETURN_PERIODS,
                benchmark_data=benchmark_df)
            output_filename = os.path.join(OUTPUT_DIR,
                                           f"report_{final_factor_name}.html")
            logging.info(f"⚙️ 正在生成 HTML 报告至: {output_filename}")
            report_generator.generate_html_report(output_filename)

    logging.info(f"\n{'='*60}")
    logging.info("🏁 所有因子分析流程执行完毕 🏁")
    logging.info(f"{'='*60}")
