# main_analyzer.py

import os
import logging
import pandas as pd
from universe_config import UNIVERSE
from data.data_providers import TushareDataProvider, AkshareDataProvider, SQLiteDataProvider
from data.data_manager import DataProviderManager
from factor_analysis.factor_calculator import FactorCalculator
from factor_analysis.factor_report import FactorReport
from logger.logger_config import setup_logging

# 导入标准化器和合成器模块
from core.factor_standardizer import CrossSectionalZScoreStandardizer, NoStandardizer
from core.factor_combiner import EqualWeightCombiner, DynamicSignificanceCombiner

if __name__ == '__main__':

    # 【新增】在这里定义日志文件的存放目录
    LOG_DIR = "logs"

    # 【修改】调用新的日志配置函数，传入目录和文件名前缀
    setup_logging(log_dir=LOG_DIR, log_prefix='factor_analysis')

    logging.info("因子分析程序启动...")

    # =====================
    # 1. 分析参数配置
    # =====================
    # 定义要分析的股票池（universe），这里列出了一些示例A股代码
    # 实际使用时，应从 universe_config.py 导入 UNIVERSE 列表

    # 基准指数（用于衡量相对表现）
    BENCHMARK = '000001'

    # 在这里指定回测专用数据库的路径
    BACKTEST_DB_PATH = './database/quant_data.db'

    # 回测时间范围
    START_DATE = '2016-01-01'
    END_DATE = '2020-12-31'

    # 定义未来收益的预测周期列表（单位：交易日）
    FORWARD_RETURN_PERIODS = [1, 5, 10]

    # 报告存放路径，默认为 factor_reports 文件夹
    OUTPUT_DIR = "factor_reports"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # =====================
    # 2. 数据源配置 (与 main.py 保持一致)
    # =====================
    DATA_PROVIDERS_CONFIG = [
        (SQLiteDataProvider, {
            'db_path': './database/JY_database/sqlite/JY_database.sqlite',
            'table_name': 'JY_t_price_daily'
        }),
    ]

    # =====================
    # 3. 要分析的因子列表
    # =====================
    # 格式为元组列表：每个元素是 (因子名称, 参数字典)
    # 如果列表内有多个因子，程序将自动执行“标准化 -> 合成 -> 评测”流程
    # 如果列表内只有一个因子，程序将直接评测该因子
    FACTORS_TO_ANALYZE = [
        # ('Momentum', {
        #     'period': 20
        # }),
        # ('RSI', {
        #     'rsi_period': 22
        # }),
        ('BollingerBands', {
            'period': 30
        }),
        ('ADXDMI', {
            'period': 14,
            'trend_threshold': 22
        }),
        # ('VolumeSpike', {
        #     'period': 3
        # }),

        # ('Reversal20D', {'period': 40, 'decay': 20.0})
    ]

    # =====================
    # 【新增】4. 标准化器与合成器配置
    # =====================
    # 仅在分析多个因子时生效
    # 选择一个标准化器实例
    STANDARDIZER = CrossSectionalZScoreStandardizer()
    # 选择一个因子合成器实例
    #COMBINER = EqualWeightCombiner()
    COMBINER = DynamicSignificanceCombiner()

    # =====================
    # 5. 初始化数据管理器并准备数据
    # =====================
    all_symbols = list(set(UNIVERSE + [BENCHMARK]))
    data_manager = DataProviderManager(provider_configs=DATA_PROVIDERS_CONFIG,
                                       symbols=all_symbols,
                                       start_date=START_DATE,
                                       end_date=END_DATE,
                                       db_path=BACKTEST_DB_PATH,
                                       num_checker_threads=16,
                                       num_downloader_threads=16,
                                       batch_size=200)
    data_manager.prepare_data_for_universe()

    print(f"正在获取基准 '{BENCHMARK}' 数据用于报告对比...")
    benchmark_df = data_manager.get_dataframe(BENCHMARK)
    if benchmark_df is None or benchmark_df.empty:
        print(f"⚠️ 警告: 未能获取到基准 '{BENCHMARK}' 的数据，报告中将不会包含基准对比。")

    # =====================
    # 【修改】6. 核心分析流程：根据因子数量决定单因子或多因子合成分析
    # =====================

    final_factor_data_df = pd.DataFrame()
    final_factor_name = ""

    # --- 步骤 6a: 计算所有因子的原始值和未来收益 ---
    all_factors_dfs = {}
    future_returns_df = None  # 未来收益只需计算一次

    print(f"\n{'='*60}\n--- 步骤 1: 计算所有指定因子的原始值 ---\n{'='*60}")
    for factor_name, factor_params in FACTORS_TO_ANALYZE:
        print(f"  正在计算因子: {factor_name}...")
        calculator = FactorCalculator(
            data_manager=data_manager,
            universe=UNIVERSE,
            start_date=START_DATE,
            end_date=END_DATE,
            factor_name=factor_name,
            factor_params=factor_params,
            forward_return_periods=FORWARD_RETURN_PERIODS)

        # 这个函数会同时计算因子值和未来N期收益
        factor_data_df = calculator.calculate_factor_and_returns()

        if factor_data_df.empty:
            print(f"❌ 警告: 未能为因子 {factor_name} 生成有效数据，已跳过。")
            continue

        # 将每个因子的 'factor_value' 列提取出来，并重命名
        all_factors_dfs[factor_name] = factor_data_df[[
            'asset', 'factor_value'
        ]].rename(columns={'factor_value': factor_name})

        # 提取未来收益数据，所有因子的未来收益都是一样的，所以只存一次
        if future_returns_df is None:
            return_cols = ['asset'] + [
                f'forward_return_{p}d' for p in FORWARD_RETURN_PERIODS
            ]
            future_returns_df = factor_data_df[return_cols]

    # --- 步骤 6b: 检查因子数量并执行相应流程 ---

    # 【核心逻辑分支】
    if len(all_factors_dfs) > 1:
        # --- 多因子合成路径 ---
        print(f"\n{'='*60}\n--- 检测到多个因子，启动多因子合成流程 ---\n{'='*60}")

        # 步骤 1: 合并所有因子数据为一个宽表
        print("  步骤 2a: 合并所有因子数据...")
        combined_factors_df = all_factors_dfs[list(all_factors_dfs.keys())[0]]
        for i in range(1, len(all_factors_dfs)):
            factor_name = list(all_factors_dfs.keys())[i]
            combined_factors_df = pd.merge(combined_factors_df,
                                           all_factors_dfs[factor_name],
                                           on=['date', 'asset'],
                                           how='inner')

        combined_factors_df.set_index('asset', append=True, inplace=True)

        # 【【核心修改-开始】】
        # 步骤 2: 按日期对因子值进行截面标准化，并显示进度
        print(f"  步骤 2b: 执行截面标准化 ({STANDARDIZER.__class__.__name__})...")

        # 定义一个包裹函数，用于打印进度
        def apply_standardization_with_progress(group):
            # group.name 在这里就是当前分组的索引，也就是日期
            date_str = group.name.strftime('%Y-%m-%d')
            # 使用 \r 实现单行刷新
            print(f"    > 正在处理日期: {date_str}", end='\r')
            # droplevel('date') 是必须的，因为标准化器不期望接收到日期索引
            return STANDARDIZER.standardize(group.droplevel('date'))

        standardized_factors_df = combined_factors_df.groupby(
            level='date').apply(apply_standardization_with_progress)
        print("\n    标准化完成。")  # 换行，结束单行刷新

        # 步骤 3: 对标准化后的因子进行合成，并显示进度
        print(f"  步骤 2c: 执行因子合成 ({COMBINER.__class__.__name__})...")

        # 为合成步骤也定义一个包裹函数
        def apply_combination_with_progress(group):
            date_str = group.name.strftime('%Y-%m-%d')
            print(f"    > 正在处理日期: {date_str}", end='\r')
            return COMBINER.combine(group.droplevel('date'))

        composite_factor_series = standardized_factors_df.groupby(
            level='date').apply(apply_combination_with_progress)
        print("\n    因子合成完成。")  # 换行

        # 【【核心修复】】为Series命名，以便reset_index()时生成正确的列名
        composite_factor_series.name = 'factor_value'

        # 【【核心修改-结束】】

        # 步骤 4: 准备最终用于报告的DataFrame
        print("  步骤 2d: 准备最终报告数据...")
        final_factor_data_df = pd.merge(composite_factor_series.reset_index(),
                                        future_returns_df.reset_index(),
                                        on=['date', 'asset'],
                                        how='inner')
        final_factor_data_df.set_index('date', inplace=True)

        final_factor_name = "CompositeFactor"  # 给合成因子一个名字

    elif len(all_factors_dfs) == 1:
        # --- 单因子路径 ---
        print(f"\n{'='*60}\n--- 检测到单个因子，直接进行评测 ---\n{'='*60}")
        factor_name = list(all_factors_dfs.keys())[0]
        final_factor_name = factor_name

        # 直接合并该单因子的值和未来收益
        single_factor_df = all_factors_dfs[factor_name]
        final_factor_data_df = pd.merge(single_factor_df.reset_index(),
                                        future_returns_df.reset_index(),
                                        on=['date', 'asset'],
                                        how='inner')
        final_factor_data_df.rename(columns={factor_name: 'factor_value'},
                                    inplace=True)
        final_factor_data_df.set_index('date', inplace=True)

    else:
        print("\n--- 未计算出任何有效的因子数据，程序终止。 ---")
        exit()

    # =====================
    # 7. 生成最终的因子分析报告
    # =====================
    if not final_factor_data_df.empty:
        print(
            f"\n{'='*60}\n--- 步骤 3: 为最终因子 '{final_factor_name}' 生成分析报告 ---\n{'='*60}"
        )

        # 数据有效性检查
        final_factor_data_df.dropna(inplace=True)
        if final_factor_data_df.empty:
            print(f"❌ 警告: 最终因子 '{final_factor_name}' 数据在清理后为空，无法生成报告。")
        else:
            print(f"✅ 最终因子数据准备完成，共 {len(final_factor_data_df)} 条记录。")

            # 根据计算结果生成HTML分析报告
            report_generator = FactorReport(
                factor_name=final_factor_name,
                factor_data=final_factor_data_df,
                forward_return_periods=FORWARD_RETURN_PERIODS,
                benchmark_data=benchmark_df)

            # 输出文件路径
            output_filename = os.path.join(OUTPUT_DIR,
                                           f"report_{final_factor_name}.html")
            # 渲染HTML报告并保存到指定路径
            report_generator.generate_html_report(output_filename)
            print(f"✅ 分析报告已成功生成: {output_filename}")

    print("\n--- 所有因子分析流程执行完毕 ---")
