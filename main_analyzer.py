# main_analyzer.py

import os
import logging
import pandas as pd
from tqdm import tqdm
import sys
import datetime

# ==============================================================================
# 1. 策略分析“控制面板” (Strategy Analysis "Control Panel")
# ==============================================================================

# --- 1a. 核心策略选择 (Core Strategy Selection) ---
from strategy_configs import STRATEGY_REGISTRY

# 【【请在这里选择您的策略名称 (从 strategy_configs.py 复制)】】
# STRATEGY_NAME = "RollingICIR"
# STRATEGY_NAME = "RollingRegression"
# STRATEGY_NAME = "FixedWeights"
# STRATEGY_NAME = "EqualWeights"
# STRATEGY_NAME = "DynamicSignificance"
STRATEGY_NAME = "LightGBM_Periodic"
# STRATEGY_NAME = "AdversarialLLM"

if STRATEGY_NAME not in STRATEGY_REGISTRY:
    raise ValueError(f"策略 '{STRATEGY_NAME}' 未在 strategy_configs.py 中注册。")
STRATEGY_CONFIG = STRATEGY_REGISTRY[STRATEGY_NAME]

# ---  1b. 因子选择 (Factors to Analyze) ---
#
#   现在您只需要列出因子名称。
#   具体的参数 (params) 和数据依赖 (required_columns) 已在 factor_configs.py 中统一定义。
#
from factor_configs import FACTOR_REGISTRY
from factor_analysis.factors_complex import COMPLEX_FACTOR_REGISTRY

FACTORS_TO_RUN = [
    # 'Momentum',
    # 'Reversal20D',
    "IndNeu_Momentum",
    "IndNeu_Reversal20D",
    "IndNeu_VolumeCV",
    "IndNeu_EP",
    # "IndNeu_BP"
    # "IndNeu_ROE",
    "IndNeu_SalesGrowth",  
    "IndNeu_CFOP",
    "IndNeu_GPM",
    # "IndNeu_AssetTurnover",
    # "IndNeu_CurrentRatio",
]

# --- 1d. 截面数据配置 (Cross-Sectional Data) ---
#   【全局开关】: 是否在任何情况下都强制加载行业数据？
LOAD_INDUSTRY_DATA = True
# LOAD_INDUSTRY_DATA = False

# --- 1e. 标准化器 (Standardizer) ---
from strategies.standardizers import (CrossSectionalZScoreStandardizer,
                                      NoStandardizer,
                                      CrossSectionalQuantileStandardizer)

STANDARDIZER_CLASS = CrossSectionalZScoreStandardizer  #跨度Z分数标准化
# STANDARDIZER_CLASS = NoStandardizer  #不进行标准化
# STANDARDIZER_CLASS = CrossSectionalQuantileStandardizer  #截面分位数标准化

# ==============================================================================
# 2. 基础回测与路径配置 (Basic Backtest & Path Settings)
# ==============================================================================

#   【【【重要】】】数据下载与写入开关：
#   True: 跳过下载，直接使用数据库 (调试因子逻辑时用)
#   False: 检查并下载缺失数据 (日常更新数据时用)
SKIP_DATA_PREPARATION = True

# --- 2a. 回测时间与收益周期 ---
START_DATE = '2023-01-01'
END_DATE = '2025-12-31'
FORWARD_RETURN_PERIODS = [1, 5, 10, 20, 30, 90]

# --- 2b. 基准与股票池 ---
BENCHMARK = '600519'  # 茅台
from universe_config import UNIVERSE

# 因子计算进程数
FACTOR_CALC_PROCESSES = 16

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
    # 【【【新增步骤 1.6: 因子分类与路由】】】
    # ==============================================================================
    simple_factors_batch = []
    complex_factors_batch = []

    logging.info(f"\n{'='*60}\n--- 步骤 1.6: 因子分类与路由 ---\n{'='*60}")

    # 注意：这里使用的是统一的 FACTORS_TO_RUN 列表
    for item in FACTORS_TO_RUN:
        # 解析配置 (支持 'Name' 或 ('Name', params) 格式)
        if isinstance(item, tuple):
            f_name = item[0]
            f_params = item[1]
        else:
            f_name = item
            f_params = {}

        # 检查注册表
        if f_name not in FACTOR_REGISTRY:
            logging.warning(f"⚠️ 跳过: 因子 '{f_name}' 未在 factor_configs.py 中注册。")
            continue

        config = FACTOR_REGISTRY[f_name]
        # 获取因子类别，默认为 'simple'
        category = config.get('category', 'simple')

        # 分流逻辑
        if category == 'simple':
            simple_factors_batch.append((f_name, f_params))
        elif category == 'complex':
            complex_factors_batch.append(f_name)
        else:
            logging.warning(
                f"⚠️ 跳过: 因子 '{f_name}' 的 category '{category}' 无效。")

    logging.info(f"📋 简单因子 (多进程计算): {[f[0] for f in simple_factors_batch]}")
    logging.info(f"📋 复合因子 (全量表计算): {complex_factors_batch}")

    # ==============================================================================
    # 【【【新增步骤 1.7: 预计算所需数据列】】】
    # ==============================================================================
    logging.info(f"\n{'='*60}\n--- 步骤 1.7: 预计算所有因子所需的数据列 ---\n{'='*60}")

    all_required_columns = set()

    # 收集所有活跃因子的列需求
    all_active_factors = [f[0] for f in simple_factors_batch
                          ] + complex_factors_batch

    for factor_name in all_active_factors:
        required = FACTOR_REGISTRY[factor_name].get('required_columns', [])
        all_required_columns.update(required)

    # 处理全局行业数据开关
    if LOAD_INDUSTRY_DATA:
        all_required_columns.add('industry')

    sorted_cols = sorted(list(all_required_columns))
    logging.info(f"✅ 本次运行优化后的数据列需求: {sorted_cols}")

    # =====================
    # 2. 计算因子原始值
    # =====================
    all_factors_dfs = {}

    logging.info(f"\n{'='*60}\n--- 步骤 2: 执行因子计算 ---\n{'='*60}")

    # --- 分支 A: 执行简单因子 (Simple Factors) ---
    if not simple_factors_batch:
        logging.info("ℹ️ (无简单因子需要计算)")
    else:
        for factor_name, user_params in simple_factors_batch:
            # 获取配置
            registry_config = FACTOR_REGISTRY.get(factor_name, {})
            registry_params = registry_config.get('params', {})
            required_cols = registry_config.get('required_columns', [])

            # 合并参数 (main 配置覆盖 registry 配置)
            final_params = {**registry_params, **user_params}

            logging.info(f"⚙️ [Simple] 启动计算器: {factor_name}...")

            calculator = FactorCalculator(
                provider_configs=data_manager.provider_configs,
                db_path=BACKTEST_DB_PATH,
                universe=active_universe,
                start_date=START_DATE,
                end_date=END_DATE,
                factor_name=factor_name,
                factor_params=final_params,
                num_threads=FACTOR_CALC_PROCESSES,
                required_columns=required_cols  # 按需加载
            )

            factor_data_df = calculator.calculate_factor()

            if not factor_data_df.empty:
                factor_series = factor_data_df.set_index(
                    'asset', append=True)['factor_value']
                factor_series.name = factor_name
                all_factors_dfs[factor_name] = factor_series.sort_index()
                logging.info(f"  > ✅ 完成: {factor_name}")

    # --- 分支 B: 执行复合因子 (Complex Factors) ---
    if not complex_factors_batch:
        logging.info("ℹ️ (无复合因子需要计算)")
    else:
        logging.info(
            f"⚙️ [Complex] 正在为复合因子加载宽表数据 (Cols: {len(sorted_cols)})...")

        # 加载所有需要的列
        all_data_df = data_manager.get_all_data_for_universe(
            active_universe, required_columns=sorted_cols)

        if all_data_df is None or all_data_df.empty:
            logging.error("❌ 无法加载数据，跳过复合因子计算。")
        else:
            for factor_name in complex_factors_batch:
                if factor_name in COMPLEX_FACTOR_REGISTRY:
                    logging.info(f"⚙️ [Complex] 计算: {factor_name}...")
                    factor_func = COMPLEX_FACTOR_REGISTRY[factor_name]

                    # 复合因子函数直接接收 DataFrame
                    factor_series = factor_func(all_data_df)

                    if factor_series is not None:
                        factor_series.name = factor_name
                        all_factors_dfs[
                            factor_name] = factor_series.sort_index()
                        logging.info(f"  > ✅ 完成: {factor_name}")
                else:
                    logging.warning(
                        f"⚠️ 警告: 因子 {factor_name} 在 COMPLEX_FACTOR_REGISTRY 中未找到。"
                    )

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
        # 单因子通常不需要标准化用于合成，但如果需要统一量纲可以打开下面这行
        # combined_factors_df = combined_factors_df.groupby(level='date').apply(lambda x: STANDARDIZER.standardize(x))
    else:
        logging.info("⚙️ 步骤 3a: 合并因子数据...")
        
        # 【【【重构】】】: 统一的重复索引预处理
        total_dups_found = 0
        for factor_name, factor_series in all_factors_dfs.items():
            if factor_series.index.duplicated().any():
                dup_count = factor_series.index.duplicated().sum()
                total_dups_found += dup_count
                logging.warning(f"⚠️ 因子 {factor_name} 发现 {dup_count} 个重复索引，正在统一处理...")
                
                # 记录重复索引示例（仅记录前5个用于调试）
                dup_indices = factor_series.index[factor_series.index.duplicated()]
                if len(dup_indices) > 0:
                    logging.debug(f"重复索引示例: {dup_indices[:5].tolist()}")
                
                # 去除重复索引，保留最后一个
                original_len = len(factor_series)
                all_factors_dfs[factor_name] = factor_series[~factor_series.index.duplicated(keep='last')]
                logging.info(f"✅ 因子 {factor_name} 已处理重复索引：{original_len} -> {len(all_factors_dfs[factor_name])}")
        
        if total_dups_found > 0:
            logging.info(f"🔧 [预处理] 共处理了 {total_dups_found} 个重复索引，所有因子数据已确保索引唯一性")
        else:
            logging.info("✅ [预处理] 所有因子数据索引唯一性检查通过")
        
        # 检查股票池是否有重复
        original_universe_size = len(active_universe)
        unique_universe = list(set(active_universe))
        if len(unique_universe) != original_universe_size:
            logging.warning(f"⚠️ 股票池发现重复: 原始{original_universe_size}只，去重后{len(unique_universe)}只")
            from collections import Counter
            duplicates = Counter([x for x in active_universe if active_universe.count(x) > 1])
            logging.warning(f"重复的股票代码: {dict(duplicates)}")

        # 【【【优化】】】: 使用更安全的合并方式，避免产生新的重复索引
        try:
            combined_factors_df = pd.concat(all_factors_dfs.values(),
                                            axis=1,
                                            keys=all_factors_dfs.keys())
            if isinstance(combined_factors_df.columns, pd.MultiIndex):
                combined_factors_df.columns = combined_factors_df.columns.droplevel(1)
            
            # 【【【最终检查】】】: 确保合并后的数据没有重复索引
            if combined_factors_df.index.duplicated().any():
                dup_count = combined_factors_df.index.duplicated().sum()
                logging.warning(f"⚠️ 合并后发现 {dup_count} 个重复索引，进行最终处理...")
                combined_factors_df = combined_factors_df[~combined_factors_df.index.duplicated(keep='last')]
                logging.info(f"✅ 合并数据重复索引处理完成")
                
        except Exception as e:
            logging.error(f"❌ 因子合并失败: {e}")
            # 如果合并失败，尝试逐个合并以定位问题
            combined_factors_df = pd.DataFrame()
            for factor_name, factor_series in all_factors_dfs.items():
                try:
                    if combined_factors_df.empty:
                        combined_factors_df = factor_series.to_frame()
                    else:
                        combined_factors_df = combined_factors_df.join(factor_series, how='outer')
                except Exception as inner_e:
                    logging.error(f"❌ 因子 {factor_name} 合并失败: {inner_e}")

        # ======================================================================
        # 【【【关键修正】】】: 在进入策略分支前，统一进行全局截面标准化
        # ======================================================================
        logging.info(
            f"⚙️ 步骤 3b: 执行全局截面标准化 ({STANDARDIZER.__class__.__name__})...")

        combined_factors_df = combined_factors_df.groupby(
            level='date',
            group_keys=False).apply(lambda x: STANDARDIZER.standardize(x))
        logging.info("  > ✅ 所有因子已完成标准化处理。")

        # 核心策略逻辑
        if not STRATEGY_CONFIG.is_rolling():
            # A. 静态策略
            logging.info(
                f"ℹ️ 模式: 静态合成 (策略: {STRATEGY_CONFIG.combiner_class.__name__})")
            combiner = STRATEGY_CONFIG.combiner_class(
                **STRATEGY_CONFIG.combiner_kwargs)

            logging.info("⚙️ 步骤 3c: 因子合成...")
            composite_factor_series = combined_factors_df.groupby(
                level='date').apply(lambda x: combiner.combine(x))
            composite_factor_series.name = 'factor_value'
            final_factor_name = f"Composite_{STRATEGY_NAME}"
        else:
            # B. 动态滚动策略
            logging.info(f"ℹ️ 模式: 动态滚动 (每日权重计算)")
            roller = STRATEGY_CONFIG.create_rolling_calculator(
                forward_return_periods=FORWARD_RETURN_PERIODS,
                factor_names=FACTOR_NAMES)

            if roller is None:
                logging.error("❌ 滚动计算器创建失败。")
                sys.exit(1)

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

        if composite_factor_series is not None:
            combined_factors_df = composite_factor_series.to_frame()
        else:
            combined_factors_df = pd.DataFrame()

    # =====================
    # 4. 生成报告
    # =====================
    if not combined_factors_df.empty:
        # 合并收益率用于最终报告
        # 【【【修复】】】: 彻底处理重复索引问题
        # 使用最安全的方式：直接创建一个新的数据框
        if isinstance(combined_factors_df.index, pd.MultiIndex):
            # 多级索引情况 - 使用级别号而不是名称来避免重复名称问题
            index_names = combined_factors_df.index.names
            
            # 找到 date 和 asset 对应的级别号
            date_level = None
            asset_level = None
            
            for i, name in enumerate(index_names):
                if name == 'date' and date_level is None:
                    date_level = i
                elif name == 'asset' and asset_level is None:
                    asset_level = i
            
            if date_level is not None and asset_level is not None:
                index_data = {
                    'date': combined_factors_df.index.get_level_values(date_level).values,
                    'asset': combined_factors_df.index.get_level_values(asset_level).values
                }
            else:
                # 如果找不到，使用前两个级别
                index_data = {
                    'date': combined_factors_df.index.get_level_values(0).values,
                    'asset': combined_factors_df.index.get_level_values(1).values
                }
        else:
            # 单级索引情况
            if combined_factors_df.index.name == 'date':
                index_data = {
                    'date': combined_factors_df.index.values,
                    'asset': combined_factors_df.index.get_level_values('asset').values if hasattr(combined_factors_df.index, 'get_level_values') else range(len(combined_factors_df))
                }
            else:
                # 假设第一列是 date，第二列是 asset
                cols = combined_factors_df.columns.tolist()
                index_data = {
                    'date': combined_factors_df.iloc[:, 0] if len(cols) > 0 else combined_factors_df.index.values,
                    'asset': combined_factors_df.iloc[:, 1] if len(cols) > 1 else range(len(combined_factors_df))
                }
        
        # 创建新的数据框，包含索引和因子值
        factor_df_for_merge = pd.DataFrame(index_data)
        factor_df_for_merge['factor_value'] = combined_factors_df.iloc[:, -1].values  # 最后一列是因子值
                
        final_factor_data_df = pd.merge(factor_df_for_merge,
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
                benchmark_data=benchmark_df
                if benchmark_df is not None else pd.DataFrame())

            std_name = STANDARDIZER.__class__.__name__

            if "Standardizer" in std_name:
                std_name = std_name.replace("Standardizer", "")

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = os.path.join(
                OUTPUT_DIR,
                f"{timestamp}_report_{final_factor_name}_{std_name}.html")

            logging.info(f"⚙️ 生成 HTML 报告: {output_filename}")
            report_generator.generate_html_report(output_filename)
        else:
            logging.warning("⚠️ 最终因子数据为空，无法生成报告。")

    logging.info(f"\n{'='*60}\n🏁 分析流程执行完毕 🏁\n{'='*60}")
