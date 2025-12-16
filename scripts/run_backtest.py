"""
回测脚本

运行 Backtrader 事件驱动回测。
"""

import os
import logging

logger = logging.getLogger(__name__)

# 默认配置（仅作为后备，优先使用配置文件）
DEFAULT_CONFIG = {
    'start_date': '2024-01-01',
    'end_date': '2024-12-31',
    'initial_cash': 1000000.0,
    'commission': 0.0003,
    'db_path': './database/quant_data.db',
    'output_dir': 'output/bt_reports',
    'bt_data_dir': './bt_data',
}


def run_backtest(config_loader):
    """
    执行回测

    Args:
        config_loader: 配置加载器实例
    """
    logger.info("=" * 60)
    logger.info("开始回测流程...")
    logger.info("=" * 60)

    # 加载配置
    universe = config_loader.load_universe()
    backtest_config = config_loader.load_backtest()
    data_config = config_loader.load_data()
    providers_config = config_loader.load_providers()

    START_DATE = backtest_config.start_date
    END_DATE = backtest_config.end_date
    INITIAL_CASH = backtest_config.initial_cash
    COMMISSION = backtest_config.commission
    DB_PATH = data_config.database.get('db_path', DEFAULT_CONFIG['db_path'])
    OUTPUT_DIR = backtest_config.output.get('dir', DEFAULT_CONFIG['output_dir'])
    BT_DATA_DIR = DEFAULT_CONFIG['bt_data_dir']

    logger.info(f"股票池大小: {len(universe)}")
    logger.info(f"日期范围: {START_DATE} ~ {END_DATE}")
    logger.info(f"初始资金: {INITIAL_CASH:,.2f}")
    logger.info(f"手续费率: {COMMISSION}")

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BT_DATA_DIR, exist_ok=True)

    # =========================================================================
    # 1. 初始化数据管理器并获取因子数据
    # =========================================================================
    logger.info("\n--- 步骤 1: 初始化数据 ---")

    try:
        from data import DataProviderManager
        from data.providers import SQLiteDataProvider
    except ImportError as e:
        logger.warning(f"导入新数据模块失败: {e}")
        from old_code.data.data_manager import DataProviderManager
        from old_code.data.data_providers import SQLiteDataProvider

    # 从配置构建数据源配置
    DATA_PROVIDERS_CONFIG = []

    # 检查 sqlite 配置
    sqlite_cfg = providers_config.get('sqlite')
    if sqlite_cfg and sqlite_cfg.enabled:
        conn_cfg = sqlite_cfg.config.get('connection', {})
        tables_cfg = sqlite_cfg.config.get('tables', {}).get('daily', {})
        DATA_PROVIDERS_CONFIG.append((
            SQLiteDataProvider,
            {
                'db_path': conn_cfg.get('db_path', './database/JY_database/sqlite/JY_database.sqlite'),
                'table_name': tables_cfg.get('table_name', 'JY_t_price_daily')
            }
        ))

    # 如果没有配置，使用默认值
    if not DATA_PROVIDERS_CONFIG:
        DATA_PROVIDERS_CONFIG = [
            (
                SQLiteDataProvider,
                {
                    'db_path': './database/JY_database/sqlite/JY_database.sqlite',
                    'table_name': 'JY_t_price_daily'
                }
            ),
        ]

    # 获取下载配置
    download_cfg = data_config.download
    NUM_CHECKER_THREADS = download_cfg.get('num_checker_threads', 16)
    NUM_DOWNLOADER_THREADS = download_cfg.get('num_downloader_threads', 16)
    BATCH_SIZE = download_cfg.get('batch_size', 200)

    data_manager = DataProviderManager(
        provider_configs=DATA_PROVIDERS_CONFIG,
        symbols=universe,
        start_date=START_DATE,
        end_date=END_DATE,
        db_path=DB_PATH,
        num_checker_threads=NUM_CHECKER_THREADS,
        num_downloader_threads=NUM_DOWNLOADER_THREADS,
        batch_size=BATCH_SIZE
    )

    # =========================================================================
    # 2. 导出数据供 Backtrader 使用
    # =========================================================================
    logger.info("\n--- 步骤 2: 导出 Backtrader 数据 ---")

    try:
        from backtest.data.exporter import BTDataExporter
    except ImportError:
        try:
            from old_code.bt.data.exporter import BTDataExporter
        except ImportError:
            logger.error("无法导入 BTDataExporter，回测终止")
            return

    exporter = BTDataExporter(data_manager)

    # 简单的等权因子（用于测试）
    import pandas as pd
    logger.info("生成测试用等权因子...")

    # 获取所有数据
    all_data = []
    for symbol in universe[:50]:  # 限制股票数量用于测试
        df = data_manager.get_dataframe(symbol, columns=['close'])
        if df is not None and not df.empty:
            df['asset'] = symbol
            df['factor_value'] = 1.0  # 等权
            all_data.append(df.reset_index())

    if all_data:
        factor_df = pd.concat(all_data, ignore_index=True)
        factor_df.set_index(['date', 'asset'], inplace=True)
        factor_series = factor_df['factor_value']

        logger.info(f"导出数据，股票数: {len(universe[:50])}")
        exporter.export(
            universe=universe[:50],
            start_date=START_DATE,
            end_date=END_DATE,
            factor_series=factor_series
        )
    else:
        logger.error("无法获取数据，回测终止")
        return

    # =========================================================================
    # 3. 运行 Backtrader 回测
    # =========================================================================
    logger.info("\n--- 步骤 3: 运行 Backtrader ---")

    try:
        from old_code.bt.backtest import run_backtest as bt_run_backtest
        cerebro, results = bt_run_backtest()
        logger.info("Backtrader 回测完成")
    except ImportError as e:
        logger.warning(f"导入 Backtrader 模块失败: {e}")
        logger.info("尝试使用内置简化回测...")
        _run_simple_backtest(data_manager, universe[:50], START_DATE, END_DATE, INITIAL_CASH, OUTPUT_DIR)
        return
    except Exception as e:
        logger.error(f"Backtrader 回测失败: {e}", exc_info=True)
        return

    # =========================================================================
    # 4. 生成报告
    # =========================================================================
    logger.info("\n--- 步骤 4: 生成报告 ---")

    try:
        from backtest.reports import ReportGenerator
    except ImportError:
        try:
            from old_code.bt.utils.report_generator import ReportGenerator
        except ImportError:
            logger.warning("无法导入报告生成器")
            ReportGenerator = None

    if ReportGenerator is not None and cerebro is not None:
        report_gen = ReportGenerator(output_dir=OUTPUT_DIR)

        # 获取分析器结果
        analyzers = {}
        if results and len(results) > 0:
            strat = results[0]
            if hasattr(strat, 'analyzers'):
                if hasattr(strat.analyzers, 'sharpe'):
                    analyzers['sharpe'] = strat.analyzers.sharpe.get_analysis()
                if hasattr(strat.analyzers, 'drawdown'):
                    analyzers['drawdown'] = strat.analyzers.drawdown.get_analysis()

        report_path = report_gen.generate(cerebro, strat if results else None, analyzers)
        logger.info(f"报告已生成: {report_path}")

    logger.info("=" * 60)
    logger.info("回测流程执行完毕")
    logger.info("=" * 60)


def _run_simple_backtest(data_manager, universe, start_date, end_date, initial_cash, output_dir):
    """
    简化回测（当 Backtrader 不可用时使用）
    """
    import pandas as pd
    import datetime

    logger.info("运行简化回测...")

    # 收集所有股票的收益率
    returns_list = []
    for symbol in universe:
        df = data_manager.get_dataframe(symbol, columns=['close'])
        if df is not None and not df.empty:
            ret = df['close'].pct_change()
            ret.name = symbol
            returns_list.append(ret)

    if not returns_list:
        logger.error("无法获取数据")
        return

    # 合并收益率
    returns_df = pd.concat(returns_list, axis=1)
    returns_df = returns_df.dropna(how='all')

    # 等权组合收益
    portfolio_returns = returns_df.mean(axis=1)

    # 计算净值
    portfolio_value = initial_cash * (1 + portfolio_returns).cumprod()

    # 计算统计指标
    total_return = (portfolio_value.iloc[-1] / initial_cash - 1) * 100
    sharpe = portfolio_returns.mean() / portfolio_returns.std() * (252 ** 0.5) if portfolio_returns.std() > 0 else 0
    max_drawdown = ((portfolio_value.cummax() - portfolio_value) / portfolio_value.cummax()).max() * 100

    logger.info(f"简化回测结果:")
    logger.info(f"  总收益率: {total_return:.2f}%")
    logger.info(f"  Sharpe Ratio: {sharpe:.3f}")
    logger.info(f"  最大回撤: {max_drawdown:.2f}%")
    logger.info(f"  最终净值: {portfolio_value.iloc[-1]:,.2f}")

    # 生成简单报告
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"{timestamp}_backtrader_report.html")

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>简化回测报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; }}
        h1 {{ color: #333; }}
        .metric {{ display: inline-block; margin: 20px; padding: 20px; background: #f8f9fa; border-radius: 8px; text-align: center; min-width: 150px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .metric-label {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>简化回测报告</h1>
        <p>日期范围: {start_date} ~ {end_date}</p>
        <p>股票数量: {len(universe)}</p>
        <div>
            <div class="metric">
                <div class="metric-value">{total_return:+.2f}%</div>
                <div class="metric-label">总收益率</div>
            </div>
            <div class="metric">
                <div class="metric-value">{sharpe:.3f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric">
                <div class="metric-value">{max_drawdown:.2f}%</div>
                <div class="metric-label">最大回撤</div>
            </div>
            <div class="metric">
                <div class="metric-value">{portfolio_value.iloc[-1]:,.0f}</div>
                <div class="metric-label">最终净值</div>
            </div>
        </div>
    </div>
</body>
</html>
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"简化报告已生成: {report_path}")


if __name__ == '__main__':
    from core.config import load_config
    config_loader = load_config()
    run_backtest(config_loader)
