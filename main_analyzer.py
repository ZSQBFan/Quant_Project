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
STRATEGY_NAME = "RollingRegression"
# STRATEGY_NAME = "FixedWeights"
# STRATEGY_NAME = "EqualWeights"
# STRATEGY_NAME = "DynamicSignificance"

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
    ('Reversal20D', {
        'period': 40,
        'decay': 20
    }),
]

# --- 1c. 复合因子选择 (Complex Factors) ---
#
#   【【高级配置】】
#   您希望在 "Type 1" 基础因子计算【之后】运行哪些 "Type 2" 复合因子？
#
from factor_analysis.factors_complex import COMPLEX_FACTOR_REGISTRY

COMPLEX_FACTORS_TO_RUN = [
    "IndNeu_Momentum",
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
from core.factor_standardizer import (CrossSectionalZScoreStandardizer,
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
# SKIP_DATA_PREPARATION = False # (正常运行时设为 False)
#
# ==============================================================================

# --- 2a. 回测时间与收益周期 ---
START_DATE = '2023-01-01'
END_DATE = '2024-12-31'
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

    # 1. 提取因子名称列表 (来自 1b)
    FACTOR_NAMES = [f[0] for f in FACTORS_TO_ANALYZE]

    # 2. 实例化标准化器 (来自 1e)
    STANDARDIZER = STANDARDIZER_CLASS()
    logging.info(f"✅ 标准化器已加载: {STANDARDIZER.__class__.__name__}")

    # 3. 【【重构后的初始化逻辑】】 (来自 1a 和 strategy_configs.py)
    logging.info(f"⚙️ 正在加载策略: {STRATEGY_NAME}")

    # 3a. 实例化合成器 (Combiner)
    COMBINER = STRATEGY_CONFIG.create_combiner()
    logging.info(f"✅ 合成器已加载: {STRATEGY_CONFIG.combiner_class.__name__}")

    # 3b. 检查滚动逻辑 (Rolling)
    _run_rolling = STRATEGY_CONFIG.is_rolling()
    logging.info(f"ℹ️ 自动检测滚动逻辑: {_run_rolling}")

    if _run_rolling:
        logging.info(f"  > 滚动配置: {STRATEGY_CONFIG.rolling_config}")
    else:
        logging.info(f"  > 模式: 静态 (非滚动)")
    logging.info("✅ 策略配置加载完毕。")
    # --- 【【初始化逻辑结束】】 ---

    # =====================
    # 1. 初始化数据管理器并准备数据
    # =====================
    logging.info(f"\n{'='*60}\n--- 步骤 1: 准备数据 ---\n{'='*60}")

    # 1. 初始化 DataProviderManager
    #    (这一步【必须】执行，以便后续步骤可以从数据库读取数据)
    logging.info("⚙️ 正在初始化 DataProviderManager...")
    data_manager = DataProviderManager(
        provider_configs=DATA_PROVIDERS_CONFIG,  # <- 【【重要】】
        symbols=UNIVERSE,
        start_date=START_DATE,
        end_date=END_DATE,
        db_path=BACKTEST_DB_PATH,  # <- 【【重要】】
        num_checker_threads=16,
        num_downloader_threads=16,
        batch_size=200)

    # 2. 将 BENCHMARK (基准) 添加到 "待下载" 列表中
    if BENCHMARK not in data_manager.symbols:
        data_manager.symbols.append(BENCHMARK)
        logging.info(f"  > 已将基准 {BENCHMARK} 添加到数据管理器任务列表。")

    # 3. 【【【核心修改：跳过数据准备】】】
    if not SKIP_DATA_PREPARATION:
        logging.info("⚙️ 模式: 完整数据准备 (检查、下载、写入)...")
        # 3. 执行【完整】数据准备 (ETL 流程)
        data_manager.prepare_data_for_universe()
        logging.info("✅ 完整数据准备流程 (ETL) 已完成。")
    else:
        logging.info(
            "🟡 【【跳过】】: 已按配置 (SKIP_DATA_PREPARATION=True) 跳过数据检查与下载流程。")
        logging.info("ℹ️ 模式: 直接使用数据库现有数据...")

    # 4. 获取基准数据，用于报告对比
    logging.info(f"⚙️ 正在获取基准 '{BENCHMARK}' 数据用于报告对比...")
    benchmark_df = data_manager.get_dataframe(BENCHMARK)
    if benchmark_df is None or benchmark_df.empty:
        logging.warning(f"⚠️ 警告: 未能获取到基准 '{BENCHMARK}' 的数据。报告中将不包含基准对比。")
    else:
        logging.info(f"✅ 成功获取基准 '{BENCHMARK}' 数据。")

    # =====================
    # 2. 计算所有因子的原始值和未来收益
    # =====================
    all_factors_dfs = {}  # <-- 将存储【所有】(Type 1 和 2) 的因子 Series
    future_returns_df = None

    logging.info(
        f"\n{'='*60}\n--- 步骤 2: 计算所有指定因子的原始值 (Type 1 因子) ---\n{'='*60}")

    # 1. 准备股票池
    active_universe = data_manager.symbols.copy()
    if BENCHMARK in active_universe:
        active_universe.remove(BENCHMARK)
        logging.info(f"  > 已从因子计算池中移除基准 {BENCHMARK}。")

    # 2. 【【【步骤 2a: 计算基础因子 (Type 1)】】】
    if not FACTORS_TO_ANALYZE:
        logging.info("ℹ️ (跳过: 未在 1b 中配置基础因子)")
    else:
        for factor_name, factor_params in FACTORS_TO_ANALYZE:
            logging.info(
                f"⚙️ 正在启动 (Type 1) 计算器: {factor_name} (参数: {factor_params})..."
            )

            # 【【【【【【 核心修正 1 】】】】】】
            # 将 data_manager 实例替换为它所包含的配置
            # 以匹配 factor_calculator.py 的新 __init__ 签名
            calculator = FactorCalculator(
                provider_configs=data_manager.provider_configs,  # <- 【【修正】】
                db_path=BACKTEST_DB_PATH,  # <- 【【修正】】
                universe=active_universe,
                start_date=START_DATE,
                end_date=END_DATE,
                factor_name=factor_name,
                factor_params=factor_params,
                forward_return_periods=FORWARD_RETURN_PERIODS,
                num_threads=FACTOR_CALC_PROCESSES  # (这个参数名在内部被映射为进程数)
            )
            # 【【【【【【 修正结束 】】】】】】

            factor_data_df = calculator.calculate_factor_and_returns()

            if factor_data_df.empty:
                logging.warning(f"❌ 警告: 未能为因子 {factor_name} 生成有效数据，已跳过。")
                continue

            # 【重要】将因子 Series (MultiIndex) 存入 all_factors_dfs
            factor_series = factor_data_df.set_index(
                'asset', append=True)['factor_value']
            factor_series.name = factor_name
            all_factors_dfs[factor_name] = factor_series.sort_index()
            logging.info(f"✅ 成功计算并存储 (Type 1) 因子: {factor_name}")

            if future_returns_df is None:
                logging.info("  > 正在缓存未来收益数据...")
                return_cols = ['asset'] + [
                    f'forward_return_{p}d' for p in FORWARD_RETURN_PERIODS
                ]
                future_returns_df = factor_data_df[return_cols].reset_index()

    # 3. 【【【步骤 2.5: 计算复合因子 (Type 2)】】】
    logging.info(
        f"\n{'='*60}\n--- 步骤 2.5: 计算所有指定因子的复合值 (Type 2 因子) ---\n{'='*60}")
    if not COMPLEX_FACTORS_TO_RUN:
        logging.info("ℹ️ (跳过: 未在 1c 中配置复合因子)")
    else:
        logging.info("⚙️ 步骤 2.5a: 加载【全部】股票的日线数据 (用于复合计算)...")
        # (进度条在 data_manager.get_all_data_for_universe 内部)
        all_data_df = data_manager.get_all_data_for_universe(active_universe)

        if all_data_df is None:
            logging.error("❌ 错误: 无法加载复合因子所需的基础数据，已跳过。")
        else:
            # 步骤 2.5b: (可选) 加载并合并截面数据
            if LOAD_INDUSTRY_DATA:
                logging.info("⚙️ 步骤 2.5b: (按配置) 正在加载并合并行业数据 ('stock_kind')...")
                try:
                    industry_df = data_manager.get_industry_mapping()
                    if industry_df is not None:
                        # 将 (asset, industry) 合并到 ('date', 'asset') 的主数据中
                        all_data_df = all_data_df.reset_index().merge(
                            industry_df, on='asset',
                            how='left').set_index(['date',
                                                   'asset']).sort_index()
                        logging.info("  > ✅ 行业数据合并完成。")
                    else:
                        logging.warning("  > ⚠️ 警告: 未能从 'stock_kind' 加载行业数据。")
                except AttributeError:
                    logging.error(
                        "  > ❌ 错误: 'get_industry_mapping' 函数未在 DataProviderManager 中定义。"
                    )

            # (未来可以在此合并 'market_cap' 等)

            # 步骤 2.5c: 循环计算
            logging.info(
                f"⚙️ 步骤 2.5c: 开始计算 {len(COMPLEX_FACTORS_TO_RUN)} 个复合因子...")

            # (使用 tqdm 包裹循环)
            tqdm_loop = tqdm(
                COMPLEX_FACTORS_TO_RUN,
                desc="[主循环] 计算复合因子",
                ncols=100,
                leave=False,
                file=sys.stdout  # (保持与 logger_config.py 一致)
            )

            for factor_key in tqdm_loop:
                tqdm_loop.set_description(f"[主循环] 复合因子 ({factor_key})")

                if factor_key in COMPLEX_FACTOR_REGISTRY:
                    calc_func = COMPLEX_FACTOR_REGISTRY[factor_key]

                    # 【核心】调用复合计算函数
                    complex_factor_series = calc_func(all_data_df)

                    if complex_factor_series is not None:
                        logging.debug(f"    > ✅ 成功计算 (Type 2): {factor_key}")
                        all_factors_dfs[
                            factor_key] = complex_factor_series.sort_index()
                else:
                    logging.warning(f"  > ❌ 警告: 复合因子 '{factor_key}' 在 "
                                    f"factors_complex.py 中未注册，已跳过。")

            logging.info("✅ 所有复合因子计算完毕。")

    # 4. 【【【步骤 2.6: 检查未来收益 (处理边界情况)】】】
    if future_returns_df is None:
        if not all_factors_dfs:
            logging.critical("⛔ 错误: 未计算出任何 (Type 1 或 Type 2) 因子数据。程序终止。")
            exit()
        else:
            logging.warning("⚠️ 警告: 未计算未来收益 (因为 Type 1 因子被跳过)。")
            logging.info("  > ⚙️ 正在【重新】运行一个基础计算器以获取未来收益...")

            # 【【【【【【 核心修正 2 】】】】】】
            temp_calc = FactorCalculator(
                provider_configs=data_manager.provider_configs,  # <- 【【修正】】
                db_path=BACKTEST_DB_PATH,  # <- 【【修正】】
                universe=active_universe,
                start_date=START_DATE,
                end_date=END_DATE,
                factor_name='RSI',
                factor_params={'rsi_period': 14},
                forward_return_periods=FORWARD_RETURN_PERIODS,
                num_threads=FACTOR_CALC_PROCESSES)
            # 【【【【【【 修正结束 】】】】】】

            logging.info("  > ⚙️ 正在为所有股票计算未来收益...")
            all_data_df_with_returns = temp_calc.calculate_factor_and_returns(
                run_factor_calc=False  # (仅计算收益)
            )

            if all_data_df_with_returns.empty:
                logging.critical("  > ❌ 致命错误: 无法补算未来收益。程序终止。")
                exit()

            return_cols = ['asset'] + [
                f'forward_return_{p}d' for p in FORWARD_RETURN_PERIODS
            ]
            future_returns_df = all_data_df_with_returns[
                return_cols].reset_index()
            logging.info("  > ✅ 未来收益已补算。")

    # =====================
    # 3. 核心分析流程：单因子 vs 多因子 (静态/滚动)
    # =====================
    final_factor_data_df = pd.DataFrame()
    final_factor_name = ""

    # 1. 动态生成最终的 FACTOR_NAMES 列表
    FACTOR_NAMES = list(all_factors_dfs.keys())
    logging.info(f"\n{'='*60}\n--- 步骤 3: 因子合并与分析 ---\n{'='*60}")
    logging.info(f"ℹ️ 即将合并的【所有】因子: {FACTOR_NAMES}")

    if len(FACTOR_NAMES) > 1:
        # --- 多因子合成路径 ---
        logging.info("⚙️ 步骤 3a: 合并所有 (Type 1 和 Type 2) 因子数据...")

        all_factors_df_list = []
        for factor_name, factor_series in all_factors_dfs.items():
            df = factor_series.to_frame().reset_index()
            all_factors_df_list.append(df)

        combined_factors_df = all_factors_df_list[0]

        if len(all_factors_df_list) > 1:
            for i in range(1, len(all_factors_df_list)):
                combined_factors_df = pd.merge(combined_factors_df,
                                               all_factors_df_list[i],
                                               on=['date', 'asset'],
                                               how='outer')
        logging.info(f"  > ✅ 成功合并 {len(FACTOR_NAMES)} 个因子。")

        combined_factors_df = combined_factors_df.set_index(['date', 'asset'
                                                             ]).sort_index()

        # 【【【核心逻辑分支：静态 vs 滚动】】】
        if not _run_rolling:
            # --- 分支A: 静态权重逻辑 (GroupBy) ---
            logging.info(f"ℹ️ 模式: 静态 (GroupBy 模式)")
            logging.info(
                f"⚙️ 步骤 3b: 执行截面标准化 ({STANDARDIZER.__class__.__name__})...")

            standardized_factors_df = combined_factors_df.groupby(
                level='date').apply(lambda group: STANDARDIZER.standardize(
                    group.droplevel('date')[FACTOR_NAMES]))
            logging.info("    > ✅ (静态) 标准化完成。")

            logging.info(
                f"⚙️ 步骤 3c: 执行因子合成 ({COMBINER.__class__.__name__})...")

            composite_factor_series = standardized_factors_df.groupby(
                level='date').apply(
                    lambda group: COMBINER.combine(group.droplevel('date')))
            logging.info("    > ✅ (静态) 因子合成完成。")

            composite_factor_series.name = 'factor_value'
            final_factor_name = f"CompositeFactor_{STRATEGY_NAME}_Static"

        else:
            # --- 分支B: 动态滚动权重逻辑 (逐日循环) ---
            logging.info(f"ℹ️ 模式: 动态滚动 (逐日循环模式)")

            logging.info("⚙️ 步骤 3b: 初始化滚动计算器...")
            roller = STRATEGY_CONFIG.create_rolling_calculator(
                forward_return_periods=FORWARD_RETURN_PERIODS,
                factor_names=FACTOR_NAMES)
            if roller is None:
                logging.critical(f"⛔ 策略 {STRATEGY_NAME} 配置错误：应为滚动模式但无法创建滚动器。")
                raise Exception(f"策略 {STRATEGY_NAME} 配置错误：无法创建滚动器。")

            logging.info("⚙️ 步骤 3c: 准备滚动数据 (合并因子与未来收益)...")
            all_data_merged = pd.merge(combined_factors_df.reset_index(),
                                       future_returns_df,
                                       on=['date', 'asset'],
                                       how='inner')
            all_data_merged.set_index(['date', 'asset'],
                                      inplace=True,
                                      drop=False)
            all_data_merged.sort_index(inplace=True)
            logging.info(f"  > ✅ 滚动数据准备完毕，共 {len(all_data_merged)} 条合并记录。")

            trading_dates = all_data_merged.index.get_level_values(
                'date').unique().sort_values()

            all_dates_series = pd.Series(index=trading_dates).index
            rebalance_freq = STRATEGY_CONFIG.get_rolling_param(
                'REBALANCE_FREQUENCY')
            rebalance_dates_ideal = pd.date_range(start=trading_dates.min(),
                                                  end=trading_dates.max(),
                                                  freq=rebalance_freq)

            rebalance_dates_idx = all_dates_series.searchsorted(
                rebalance_dates_ideal)
            rebalance_dates = all_dates_series[rebalance_dates_idx[
                rebalance_dates_idx < len(all_dates_series)]].date
            logging.info(
                f"  > ℹ️ 调仓频率: {rebalance_freq} (共 {len(rebalance_dates)} 个调仓日)"
            )

            all_composite_scores = []
            rolling_window_days = STRATEGY_CONFIG.get_rolling_param(
                'ROLLING_WINDOW_DAYS')
            logging.info(f"  > ℹ️ 回看窗口: {rolling_window_days} 天")

            logging.info(f"⚙️ 步骤 3d: 执行滚动标准化与合成 (共 {len(trading_dates)} 天)...")

            # 【【【使用 TQDM 包裹循环】】】
            for current_date in tqdm(
                    trading_dates, desc="[主循环] 滚动回测中", ncols=100,
                    file=sys.stdout):  # (保持与 logger_config.py 一致)

                logging.debug(
                    f"  > 正在处理日期: {current_date.strftime('%Y-%m-%d')}")

                # 1. 【调仓日】: 重新计算和更新权重
                if current_date.date() in rebalance_dates:
                    # (这个 INFO 日志会导致进度条跳动，但它是必要的低频信息)
                    logging.info(
                        f"  >  pivotal: {current_date.strftime('%Y-%m-%d')} 是调仓日，重新计算权重..."
                    )
                    window_end_date = current_date
                    window_start_date = window_end_date - pd.DateOffset(
                        days=rolling_window_days)

                    historical_window_mask = (
                        (all_data_merged.index.get_level_values('date')
                         >= window_start_date) &
                        (all_data_merged.index.get_level_values('date')
                         < window_end_date))
                    historical_window = all_data_merged.loc[
                        historical_window_mask]

                    if not historical_window.empty:
                        logging.debug(
                            f"    > 正在调用 roller.calculate_new_weights...")
                        new_weights = roller.calculate_new_weights(
                            historical_window)
                        COMBINER.update_weights(new_weights)
                    else:
                        logging.warning(
                            f"    > ⚠️ 警告: {current_date} 的历史窗口数据为空，无法更新权重。")

                # 2. 【每日】: 使用【当前】权重进行合成
                todays_data_slice = combined_factors_df[FACTOR_NAMES].loc[
                    current_date]
                standardized_slice = STANDARDIZER.standardize(
                    todays_data_slice)
                composite_score_series = COMBINER.combine(standardized_slice)

                composite_score_series.index = pd.MultiIndex.from_product(
                    [[current_date], composite_score_series.index],
                    names=['date', 'asset'])
                all_composite_scores.append(composite_score_series)

            logging.info("    > ✅ 滚动合成完成。")
            if not all_composite_scores:
                logging.error("  > ❌ 错误：滚动合成未产生任何结果。")
                composite_factor_series = pd.Series(name='factor_value')
            else:
                composite_factor_series = pd.concat(all_composite_scores)
                composite_factor_series.name = 'factor_value'

            final_factor_name = f"CompositeFactor_{STRATEGY_NAME}_Rolling"

        # --- 滚动逻辑分支结束 ---

        logging.info("⚙️ 步骤 3e: 准备最终报告数据...")
        if composite_factor_series.empty:
            logging.error("  > ❌ 错误：因子合成结果为空，无法生成报告。")
        else:
            final_factor_data_df = pd.merge(
                composite_factor_series.reset_index(),
                future_returns_df,
                on=['date', 'asset'],
                how='inner')
            final_factor_data_df.set_index('date', inplace=True)
            logging.info("  > ✅ 最终报告数据准备完毕。")

    elif len(FACTOR_NAMES) == 1:
        # --- 单因子路径 ---
        logging.info(f"\n{'='*60}\n--- 步骤 3: 单因子评测流程 ---\n{'='*60}")
        factor_name = FACTOR_NAMES[0]
        final_factor_name = f"{factor_name}_Standardized"
        logging.info(f"⚙️ 步骤 3a: 准备单因子数据: {factor_name}...")

        raw_factor_series = all_factors_dfs[factor_name]
        raw_factor_df_indexed = raw_factor_series.to_frame(name=factor_name)

        logging.info(
            f"⚙️ 步骤 3b: 执行截面标准化 ({STANDARDIZER.__class__.__name__})...")

        def apply_standardization(group):
            return STANDARDIZER.standardize(
                group.droplevel('date')[[factor_name]])

        standardized_factor_df = raw_factor_df_indexed.groupby(
            level='date').apply(apply_standardization)
        logging.info("    > ✅ (单因子) 标准化完成。")

        standardized_factor_df.rename(columns={factor_name: 'factor_value'},
                                      inplace=True)

        logging.info("⚙️ 步骤 3c: 准备最终报告数据...")
        final_factor_data_df = pd.merge(standardized_factor_df.reset_index(),
                                        future_returns_df,
                                        on=['date', 'asset'],
                                        how='inner')
        final_factor_data_df.set_index('date', inplace=True)
        logging.info("  > ✅ 最终报告数据准备完毕。")

    else:
        # (此分支在 步骤 2.6 中已被处理，但作为双重保险)
        logging.critical("⛔ 未计算出任何有效的因子数据，程序终止。")
        exit()

    # =====================
    # 4. 生成最终的因子分析报告
    # =====================
    if not final_factor_data_df.empty:
        logging.info(
            f"\n{'='*60}\n--- 步骤 4: 为最终因子 '{final_factor_name}' 生成分析报告 ---\n{'='*60}"
        )

        final_factor_data_df.dropna(inplace=True)
        if final_factor_data_df.empty:
            logging.warning(
                f"  > ⚠️ 警告: 最终因子 '{final_factor_name}' 数据在清理(dropna)后为空。")
        else:
            logging.info(
                f"  > ✅ 最终因子数据准备完成，共 {len(final_factor_data_df)} 条有效记录。")

            report_generator = FactorReport(
                factor_name=final_factor_name,
                factor_data=final_factor_data_df,
                forward_return_periods=FORWARD_RETURN_PERIODS,
                benchmark_data=benchmark_df)

            output_filename = os.path.join(OUTPUT_DIR,
                                           f"report_{final_factor_name}.html")

            logging.info(f"⚙️ 正在生成 HTML 报告...")
            # 【核心】生成 HTML 报告
            report_generator.generate_html_report(output_filename)
            # (日志已移至 report_generator 内部)

    logging.info(f"\n{'='*60}")
    logging.info("🏁 所有因子分析流程执行完毕 🏁")
    logging.info(f"{'='*60}")
