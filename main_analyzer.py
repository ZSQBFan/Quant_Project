# main_analyzer.py

import os
import logging
import pandas as pd
from tqdm import tqdm
import sys

# ==============================================================================
# 1. 策略分析“控制面板” (Strategy Analysis "Control Panel")
# ==============================================================================

# --- 1a. 核心策略选择 (Core Strategy Selection) ---
from strategy_configs import STRATEGY_REGISTRY

# 【【请在这里选择您的策略名称 (从 strategy_configs.py 复制)】】
# STRATEGY_NAME = "RollingICIR_Daily"
# STRATEGY_NAME = "RollingRegression_Daily"
# STRATEGY_NAME = "FixedWeights"
# STRATEGY_NAME = "EqualWeights"
# STRATEGY_NAME = "DynamicSignificance"
STRATEGY_NAME = "AI_Periodic_Retrain"

if STRATEGY_NAME not in STRATEGY_REGISTRY:
    raise ValueError(f"策略 '{STRATEGY_NAME}' 未在 strategy_configs.py 中注册。")
STRATEGY_CONFIG = STRATEGY_REGISTRY[STRATEGY_NAME]

# --- 【【【新】】】 1b. 因子选择 (Factors to Analyze) ---
#
#   现在您只需要列出因子名称。
#   具体的参数 (params) 和数据依赖 (required_columns) 已在 factor_configs.py 中统一定义。
#
from factor_configs import FACTOR_REGISTRY

FACTORS_TO_ANALYZE = [
    # 'Momentum',
    # 'Reversal20D',
    # 'RSI',
    # 'BollingerBands',
]

# --- 1c. 复合因子选择 (Complex Factors) ---
from factor_analysis.factors_complex import COMPLEX_FACTOR_REGISTRY

COMPLEX_FACTORS_TO_RUN = [
    "IndNeu_Momentum",
    "IndNeu_Reversal20D",
    "IndNeu_VolumeCV",
]

# --- 1d. 截面数据配置 (Cross-Sectional Data) ---
#   【全局开关】: 是否在任何情况下都强制加载行业数据？
#   (即使当前计算的因子不需要行业数据，如果您想在生成的报告中按行业分组查看，也需要开启此项)
LOAD_INDUSTRY_DATA = False

# --- 1e. 标准化器 (Standardizer) ---
from strategies.standardizers import (CrossSectionalZScoreStandardizer,
                                      NoStandardizer,
                                      CrossSectionalQuantileStandardizer)

STANDARDIZER_CLASS = CrossSectionalZScoreStandardizer

# ==============================================================================
# 2. 基础回测与路径配置 (Basic Backtest & Path Settings)
# ==============================================================================

#   【【【重要】】】数据下载与写入开关：
#   True: 跳过下载，直接使用数据库 (调试因子逻辑时用)
#   False: 检查并下载缺失数据 (日常更新数据时用)
SKIP_DATA_PREPARATION = True

# --- 2a. 回测时间与收益周期 ---
START_DATE = '2018-01-01'
END_DATE = '2020-12-31'
FORWARD_RETURN_PERIODS = [1, 5, 10, 20, 30, 90]

# --- 2b. 基准与股票池 ---
BENCHMARK = '600519'  # 茅台
from universe_config import UNIVERSE

# 因子计算进程数
FACTOR_CALC_PROCESSES = 8

# --- 2c. 路径配置 ---
LOG_DIR = "logs"
OUTPUT_DIR = "factor_reports"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
BACKTEST_DB_PATH = './database/quant_data.db'

# --- 2d. 数据源配置 ---
from data.data_providers import SQLiteDataProvider

DATA_PROVIDERS_CONFIG = [
    (
        SQLiteDataProvider,
        {
            'db_path': './database/JY_database/sqlite/JY_database.sqlite',
            'table_name': 'JY_t_price_daily'  # 或者是您的数据源表名
        }),
]

# ==============================================================================
#
#                     --- 核心程序开始 (Core Logic Starts) ---
#
# ==============================================================================

from data.data_manager import DataProviderManager
from factor_analysis.factor_calculator import FactorCalculator
from factor_analysis.factor_report import FactorReport
from logger.logger_config import setup_logging

if __name__ == '__main__':

    # =====================
    # 0. 初始化与配置校验
    # =====================
    setup_logging(log_dir=LOG_DIR, log_prefix='factor_analysis')
    logging.info(f"\n{'='*60}\n--- 步骤 0: 初始化与配置校验 ---\n{'='*60}")
    logging.info("🏁 因子分析程序启动...")

    STANDARDIZER = STANDARDIZER_CLASS()
    logging.info(f"✅ 标准化器已加载: {STANDARDIZER.__class__.__name__}")
    logging.info(f"⚙️ 正在加载策略: {STRATEGY_NAME}")

    # =====================
    # 1. 初始化数据管理器
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

    if BENCHMARK not in data_manager.symbols:
        data_manager.symbols.append(BENCHMARK)

    if not SKIP_DATA_PREPARATION:
        logging.info("⚙️ 模式: 完整数据准备 (检查、下载、写入)...")
        data_manager.prepare_data_for_universe()
    else:
        logging.info("🟡 【【跳过】】: 已按配置跳过数据下载流程。")

    # 获取基准数据
    logging.info(f"⚙️ 获取基准 '{BENCHMARK}' 数据...")
    benchmark_df = data_manager.get_dataframe(BENCHMARK, columns=['close'])

    active_universe = data_manager.symbols.copy()
    if BENCHMARK in active_universe:
        active_universe.remove(BENCHMARK)

    # =====================
    # 1.5 统一计算未来收益率
    # =====================
    logging.info(f"\n{'='*60}\n--- 步骤 1.5: 预计算未来收益率 ---\n{'='*60}")
    future_returns_df = data_manager.calculate_universe_forward_returns(
        universe=active_universe,
        forward_return_periods=FORWARD_RETURN_PERIODS)

    if future_returns_df is None or future_returns_df.empty:
        logging.critical("⛔ 致命错误: 未能计算出未来收益率，程序终止。")
        sys.exit()

    future_returns_df.set_index('date', inplace=True)

    # ==============================================================================
    # 【【【新增步骤 1.7: 预计算所需数据列 (按需加载核心)】】】
    # ==============================================================================
    logging.info(f"\n{'='*60}\n--- 步骤 1.7: 预计算所有因子所需的数据列 ---\n{'='*60}")

    all_required_columns = set()
    all_factors_to_run = FACTORS_TO_ANALYZE + COMPLEX_FACTORS_TO_RUN

    for factor_name in all_factors_to_run:
        # 兼容处理：如果用户在 FACTORS_TO_ANALYZE 中写了元组 ('Name', params)，取第一个元素
        if isinstance(factor_name, tuple):
            factor_name = factor_name[0]

        if factor_name in FACTOR_REGISTRY:
            required = FACTOR_REGISTRY[factor_name].get('required_columns', [])
            all_required_columns.update(required)
        else:
            logging.warning(
                f"⚠️ 警告: 因子 '{factor_name}' 未在 FACTOR_REGISTRY 中注册，将无法按需加载。")

    # 处理全局行业数据开关
    if LOAD_INDUSTRY_DATA:
        all_required_columns.add('industry')

    # 排序仅为了日志美观
    sorted_cols = sorted(list(all_required_columns))
    logging.info(f"✅ 本次运行优化后的数据列需求: {sorted_cols}")

    # =====================
    # 2. 计算因子原始值
    # =====================
    all_factors_dfs = {}
    all_data_df = None

    logging.info(f"\n{'='*60}\n--- 步骤 2: 计算所有指定因子的原始值 ---\n{'='*60}")

    # --- 步骤 2a: 计算基础因子 (Type 1) ---
    if not FACTORS_TO_ANALYZE:
        logging.info("ℹ️ (跳过: 未配置基础因子)")
    else:
        for factor_item in FACTORS_TO_ANALYZE:
            # 兼容旧格式 (name, params) 或新格式 name
            if isinstance(factor_item, tuple):
                factor_name = factor_item[0]
                # 如果用户在 main 中指定了参数，优先使用；否则用 Registry 的
                factor_params = factor_item[1] if len(factor_item) > 1 else {}
            else:
                factor_name = factor_item
                factor_params = {}  # 稍后会从 Registry 合并

            # 从 Registry 获取标准配置
            registry_config = FACTOR_REGISTRY.get(factor_name, {})
            registry_params = registry_config.get('params', {})
            required_cols = registry_config.get('required_columns', [])

            # 合并参数 (main 配置覆盖 registry 配置)
            final_params = {**registry_params, **factor_params}

            logging.info(
                f"⚙️ 启动 (Type 1) 计算器: {factor_name} (Cols: {required_cols})..."
            )

            calculator = FactorCalculator(
                provider_configs=data_manager.provider_configs,
                db_path=BACKTEST_DB_PATH,
                universe=active_universe,
                start_date=START_DATE,
                end_date=END_DATE,
                factor_name=factor_name,
                factor_params=final_params,
                num_threads=FACTOR_CALC_PROCESSES,
                required_columns=required_cols  # 【【【核心：按需加载】】】
            )

            factor_data_df = calculator.calculate_factor()

            if not factor_data_df.empty:
                factor_series = factor_data_df.set_index(
                    'asset', append=True)['factor_value']
                factor_series.name = factor_name
                all_factors_dfs[factor_name] = factor_series.sort_index()
                logging.info(f"✅ 成功计算: {factor_name}")

    # --- 步骤 2b: 计算复合因子 (Type 2) ---
    if not COMPLEX_FACTORS_TO_RUN:
        logging.info("ℹ️ (跳过: 未配置复合因子)")
    else:
        logging.info("⚙️ 正在准备 (Type 2) 复合因子计算所需的全量数据...")
        # 【【【核心：只加载所有因子需要的列并集】】】
        all_data_df = data_manager.get_all_data_for_universe(
            active_universe, required_columns=list(all_required_columns))

        if all_data_df is None:
            logging.error("❌ 无法加载复合因子所需的基础数据。")
        else:
            # 这里的行业数据已经在 get_all_data_for_universe 中根据 'industry' 列自动合并了
            # 所以不需要像以前那样手动 merge get_industry_mapping

            for factor_name in COMPLEX_FACTORS_TO_RUN:
                if factor_name in COMPLEX_FACTOR_REGISTRY:
                    logging.info(f"⚙️ 计算 (Type 2) 复合因子: {factor_name}...")
                    factor_func = COMPLEX_FACTOR_REGISTRY[factor_name]
                    factor_series = factor_func(all_data_df)
                    if factor_series is not None:
                        factor_series.name = factor_name
                        all_factors_dfs[
                            factor_name] = factor_series.sort_index()
                        logging.info(f"✅ 成功计算: {factor_name}")

    # =====================
    # 3. 因子合并与分析
    # =====================
    final_factor_data_df = pd.DataFrame()
    final_factor_name = ""
    FACTOR_NAMES = list(all_factors_dfs.keys())

    logging.info(f"\n{'='*60}\n--- 步骤 3: 因子合并与分析 ---\n{'='*60}")
    logging.info(f"ℹ️ 待合并因子: {FACTOR_NAMES}")

    if not FACTOR_NAMES:
        logging.warning("⚠️ 没有计算出任何因子，流程终止。")
    elif len(FACTOR_NAMES) == 1:
        logging.info("ℹ️ 单因子模式。")
        final_factor_name = FACTOR_NAMES[0]
        combined_factors_df = all_factors_dfs[final_factor_name].to_frame()
    else:
        logging.info("⚙️ 步骤 3a: 合并因子数据...")
        combined_factors_df = pd.concat(all_factors_dfs.values(),
                                        axis=1,
                                        keys=all_factors_dfs.keys())
        if isinstance(combined_factors_df.columns, pd.MultiIndex):
            combined_factors_df.columns = combined_factors_df.columns.droplevel(
                1)

        # 核心策略逻辑
        if not STRATEGY_CONFIG.is_rolling():
            # A. 静态策略
            logging.info(
                f"ℹ️ 模式: 静态合成 (策略: {STRATEGY_CONFIG.combiner_class.__name__})")
            combiner = STRATEGY_CONFIG.combiner_class(
                **STRATEGY_CONFIG.combiner_kwargs)

            logging.info(
                f"⚙️ 步骤 3b: 截面标准化 ({STANDARDIZER.__class__.__name__})...")
            standardized_factors_df = combined_factors_df.groupby(
                level='date').apply(lambda x: STANDARDIZER.standardize(x))

            logging.info("⚙️ 步骤 3c: 因子合成...")
            composite_factor_series = standardized_factors_df.groupby(
                level='date').apply(lambda x: combiner.combine(x))
            composite_factor_series.name = 'factor_value'
            final_factor_name = f"Composite_{STRATEGY_NAME}"
        else:
            # B. 动态滚动策略
            logging.info(f"ℹ️ 模式: 动态滚动 (每日权重计算)")
            roller = STRATEGY_CONFIG.create_rolling_calculator(
                forward_return_periods=FORWARD_RETURN_PERIODS,
                factor_names=FACTOR_NAMES)

            logging.info("⚙️ 步骤 3c: 准备滚动数据...")
            # 合并因子值和未来收益 (用于计算 IC/IR 等)
            all_data_merged = pd.merge(combined_factors_df.reset_index(),
                                       future_returns_df.reset_index(),
                                       on=['date', 'asset'],
                                       how='inner').set_index(
                                           ['date', 'asset']).sort_index()

            composite_factor_series = roller.calculate_composite_factor(
                all_data_merged)
            composite_factor_series.name = 'factor_value'
            final_factor_name = f"Composite_{STRATEGY_NAME}_Rolling"

        combined_factors_df = composite_factor_series.to_frame()

    # =====================
    # 4. 生成报告
    # =====================
    if not combined_factors_df.empty:
        # 合并收益率用于最终报告
        final_factor_data_df = pd.merge(combined_factors_df.reset_index(),
                                        future_returns_df.reset_index(),
                                        on=['date', 'asset'],
                                        how='inner')
        final_factor_data_df.rename(
            columns={'factor_value': final_factor_name}, inplace=True)
        final_factor_data_df.set_index('date', inplace=True)

        logging.info(
            f"\n{'='*60}\n--- 步骤 4: 生成报告 ({final_factor_name}) ---\n{'='*60}")

        final_report_df = final_factor_data_df.rename(
            columns={final_factor_name: 'factor_value'})
        final_report_df.dropna(subset=['factor_value'], inplace=True)

        if not final_report_df.empty:
            report_generator = FactorReport(
                factor_name=final_factor_name,
                factor_data=final_report_df,
                forward_return_periods=FORWARD_RETURN_PERIODS,
                benchmark_data=benchmark_df)

            output_filename = os.path.join(OUTPUT_DIR,
                                           f"report_{final_factor_name}.html")
            logging.info(f"⚙️ 生成 HTML 报告: {output_filename}")
            report_generator.generate_html_report(output_filename)
        else:
            logging.warning("⚠️ 最终因子数据为空，无法生成报告。")

    logging.info(f"\n{'='*60}\n🏁 分析流程执行完毕 🏁\n{'='*60}")
