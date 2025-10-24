# main.py
"""
主回测执行文件，串联各个模块，完成策略回测及报告生成全流程。

功能包括：
- 配置回测参数（标的池、基准、时间范围等）
- 使用 Akshare、Tushare 数据提供器获取并准备标的及基准数据
- 初始化多信号策略及各类子模块（信号、止损、决策逻辑）
- 运行 Backtrader 回测引擎 
- 使用 QuantStats 生成回测分析报告

结构清晰，适合用于多因子/多信号策略框架的快速回测和评估。
"""

# 导入各模块
import logging
from universe_config import UNIVERSE
from data.data_manager import DataProviderManager
from data.data_providers import AkshareDataProvider, TushareDataProvider, SQLiteDataProvider
from data.trading_calendars import (TushareTradingCalendar,
                                    AkshareTradingCalendar)
from core.trading_rules import MultiSignalStrategy
from signals.signals import (
    MomentumSignal,
    RSISignal,
    KDJSignal,
    BollingerBandsSignal,
    MovingAverageCrossSignal,
    MACDSignal,
    VolumeSpikeSignal,
    ADXDMISignal,
)
from signals.stop_loss import (ATRTrailingStopLoss, FixedPercentageStopLoss)
from core.factor_standardizer import (
    NoStandardizer,
    CrossSectionalZScoreStandardizer,
    CrossSectionalQuantileStandardizer,
)
from core.decision_logic import FactorRankingLogic
from core.factor_combiner import EqualWeightCombiner
from backtester_analysis.backtester import Backtester
from backtester_analysis.report_generator import ReportGenerator
from logger.logger_config import setup_logging

# 主程序入口
if __name__ == '__main__':

    # 定义日志文件的存放目录
    LOG_DIR = 'logs'
    # 调用新的日志配置函数，传入目录和文件名前缀
    setup_logging(log_dir=LOG_DIR, log_prefix='main_backtest')

    logging.info("回测程序启动...")

    # --- 1a. 数据源配置 ---
    # 【全新】模块化、可插拔的数据源配置
    # 【修改】将 TUSHARE_TOKEN 直接作为参数传入配置字典，不再需要独立的全局变量。

    # 交易日历配置（与数据源共享Token）
    TRADING_CALENDAR_CONFIG = (
        TushareTradingCalendar,
        {
            'token': "d2fee9f337f8944f40988aea2a73c647215579b48e24b94828c1c7b0"
            # 与 TushareDataProvider 共享 token
        }
        # 或者使用 Akshare
        #(AkshareTradingCalendar, {})
    )

    # 数据提供器配置
    DATA_PROVIDERS_CONFIG = [
        # 选项一：本地SQLite数据库数据源
        (
            SQLiteDataProvider,
            {
                # 【【核心配置】】指定你的源数据库文件路径
                'db_path': './database/JY_database/sqlite/JY_database.sqlite',

                # (可选) 如果你的源数据库里的表不叫 'stock_daily_prices'，可以在这里指定
                'table_name': 'JY_t_price_daily'
            }),
        # 选项二：Tushare数据源
        # (
        #     TushareDataProvider,
        #     {
        #         'token':
        #         "d2fee9f337f8944f40988aea2a73c647215579b48e24b94828c1c7b0",  # 将Token直接放在这里
        #         'retries': 1,
        #         'delay': 1
        #     }),
        # 选项三：Akshare数据源
        # (AkshareDataProvider, {
        #     'retries': 1,
        #     'delay': 1
        # })
    ]

    # --- 使用示例 ---
    # 1. 如果只想用 Akshare:
    # DATA_PROVIDERS_CONFIG = [(AkshareDataProvider, {'retries': 3, 'delay': 3})]

    # 2. 如果只想用 Tushare:
    # DATA_PROVIDERS_CONFIG = [(TushareDataProvider, {'token': TUSHARE_TOKEN, 'retries': 3, 'delay': 10})]

    # 3. 如果只想使用数据库中的现有数据，不进行任何网络请求（纯本地模式）:
    # DATA_PROVIDERS_CONFIG = []

    # --- 1b. 回测参数配置 ---
    # 标的池：A股重点企业股票代码 (代码保持不变)
    # 实际使用时，应从 universe_config.py 导入 UNIVERSE 列表

    # 在这里指定回测专用数据库的路径
    BACKTEST_DB_PATH = './database/quant_data.db'

    BENCHMARK = '000001'  # 基准指数（用于衡量相对表现）
    START_DATE = '2021-01-01'  #回测开始日期
    END_DATE = '2023-12-31'  # 回测结束日期

    # --- 2. 数据准备阶段 ---
    #股票池在专门的配置文件 universe_config.py 中定义
    all_symbols = UNIVERSE + [BENCHMARK]  #股票池+基准

    # 初始化统一的数据提供者管理器
    data_provider_manager = DataProviderManager(
        provider_configs=DATA_PROVIDERS_CONFIG,
        symbols=all_symbols,
        start_date=START_DATE,
        end_date=END_DATE,
        db_path=BACKTEST_DB_PATH,
        num_checker_threads=16,  # 自定义检查线程数
        num_downloader_threads=32,  # 自定义下载线程数
        batch_size=200  # 【新参数】每次攒够200个DataFrame再统一写入数据库
    )

    # 调用统一方法准备所有数据（如果配置不为空，此步会进行网络下载）
    data_provider_manager.prepare_data_for_universe()

    # --- 3. 初始化并运行回测逻辑 ---
    # 创建回测实例 (代码保持不变)
    backtester = Backtester(
        strategy_class=MultiSignalStrategy,
        cash=10000000.0,  # 初始资金 1000 万元
        commission=0.0005,  # 手续费（双边共约 0.1%）
        strategy_params={

            # --- 在这里可以轻松选择不同的因子 ---

            # 使用动量因子
            'signal_configs': [
                # (MomentumSignal, {
                #     'period': 10
                # }),  # 动量周期为20天
                # 成交量异常信号
                # (VolumeSpikeSignal, {
                #     'period': 10
                # }),
                (RSISignal, {
                    'rsi_period': 14
                }),  # RSI周期为14天
            ],

            # --- 在这里可以轻松选择不同的止损方法（可以多选，顺序从前到后） ---
            'stop_loss_configs': [
                # ATR止损
                (ATRTrailingStopLoss, {
                    'atr_period': 10,
                    'atr_multiplier': 3.0
                }),
                # 固定百分比止损
                # (FixedPercentageStopLoss, {
                #     'pct': 0.08
                # })
            ],

            # 决策逻辑
            'decision_logic_config': (
                FactorRankingLogic,  #因子排序决策逻辑
                {
                    'top_n': 10,
                    'ranking_asc': False,  # False: 综合分越高越好

                    # --- 在这里可以轻松切换不同的标准化方法 ---
                    # 将不同的因子标准化到0-1之间便于对比
                    # 选项1: 使用截面分位数标准化（默认）
                    'standardizer_config':
                    (CrossSectionalQuantileStandardizer, {}),
                    # 选项2: 使用截面Z-Score标准化 (取消注释即可切换)
                    # ('standardizer_config': (CrossSectionalZScoreStandardizer, {})),
                    # 选项3: 不进行标准化，直接用原始值加总 (取消注释即可切换)
                    # ('standardizer_config': (NoStandardizer, {}),

                    # --- 在这里可以轻松切换不同的因子合成方法 ---
                    # 标准化后的因子如何组合成为总因子并进行最终的排序选股

                    # 选项1: 等权合成
                    'combiner_config': (EqualWeightCombiner, {}),
                }),
            'rebalance_days':
            4,  # 每月约n个交易日进行再平衡调仓
            'printlog':
            True  # 是否在回测中输出交易日志
        })

    # 为每个标的添加数据源时，使用新的 data_provider_manager
    for symbol in UNIVERSE:
        if data_provider_manager.validate_data_quality(symbol):
            data_feed = data_provider_manager.get_bt_feed(symbol)
            if data_feed is not None:
                data_feed._name = symbol
                backtester.cerebro.adddata(data_feed)

    # 执行回测 (代码保持不变)
    backtester.run()

    # --- 4. 回测结果分析与报告生成 ---
    # 如果成功获取回测结果，则进入报告生成环节
    if backtester.results is not None:
        report_gen = ReportGenerator(backtester.results)
        # 【修改】从新的 data_provider_manager 获取基准数据
        benchmark_df = data_provider_manager.get_dataframe(BENCHMARK)
        report_gen.set_benchmark(benchmark_df)
        # 使用 QuantStats 生成专业 HTML 分析报告
        report_gen.generate_quantstats_report(
            filename='QuantStats分析报告.html')  # 输出文件名

    # --- 5. 提示流程完成 ---
    print("--- 所有流程执行完毕 ---")
