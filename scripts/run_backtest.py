"""
回测脚本

运行 Backtrader 事件驱动回测。
"""

import os
import logging
import re
import inspect
import pandas as pd

logger = logging.getLogger(__name__)

# 模块级缓存：存储 bt_run_backtest 函数是否支持 generate_report 参数
# 避免每次调用时重复使用 inspect.signature 进行反射检查
_bt_supports_generate_report_cache = None


def _check_bt_supports_generate_report(func):
    """检查函数是否支持 generate_report 参数（带缓存）"""
    global _bt_supports_generate_report_cache
    if _bt_supports_generate_report_cache is None:
        try:
            sig = inspect.signature(func)
            _bt_supports_generate_report_cache = 'generate_report' in sig.parameters
        except Exception:
            _bt_supports_generate_report_cache = False
    return _bt_supports_generate_report_cache

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
    # =========================================================================
    # 0. 动态调整日志等级
    # =========================================================================
    from utils.logger import setup_logging
    logging_config = config_loader.load_logging()
    if logging_config:
        new_level = logging_config.get('level', 'INFO')
        setup_logging(
            log_dir=logging_config.get('log_dir', 'output/logs'),
            log_prefix=logging_config.get('log_prefix', 'backtest'),
            log_level=new_level,
            force=True
        )

    logger.debug("=" * 60)
    logger.debug("开始回测流程...")
    logger.debug("=" * 60)

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

    logger.info(f"最终股票池大小: {len(universe)}")
    logger.info(f"日期范围: {START_DATE} ~ {END_DATE}")
    logger.info(f"初始资金: {INITIAL_CASH:,.2f}")
    logger.info(f"手续费率: {COMMISSION}")

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BT_DATA_DIR, exist_ok=True)

    # =========================================================================
    # 1. 初始化数据管理器并获取因子数据
    # =========================================================================
    logger.debug("\n--- 步骤 1: 初始化数据 ---")

    try:
        from data import DataProviderManager
        from data.providers import SQLiteDataProvider
    except ImportError as e:
        logger.warning(f"导入新数据模块失败: {e}")
        from old_code.data.data_manager import DataProviderManager
        from old_code.data.data_providers import SQLiteDataProvider

    # 从配置构建数据源配置
    DATA_PROVIDERS_CONFIG = []

    # 检查所有启用的数据提供者，寻找 sqlite 类型的提供者
    for provider_name, provider_cfg in providers_config.items():
        if provider_cfg.enabled and provider_name.startswith('sqlite'):
            conn_cfg = provider_cfg.config.get('connection', {})
            tables_cfg = provider_cfg.config.get('tables', {}).get('daily', {})
            
            # 应用完整的配置处理逻辑，包括column_mapping
            processed_kwargs = {
                'db_path': conn_cfg.get('db_path'),
                'table_name': tables_cfg.get('table_name', 'JY_t_price_daily'),
                'column_mapping': tables_cfg.get('column_mapping', {}),
                'date_format': tables_cfg.get('date_format', '%Y-%m-%d')
            }
            
            DATA_PROVIDERS_CONFIG.append((provider_name, SQLiteDataProvider, processed_kwargs))
            logger.info(f"✅ SQLite数据提供者配置已从配置文件加载: {provider_name}, db_path={processed_kwargs['db_path']}")

    # 如果没有配置，抛出异常
    if not DATA_PROVIDERS_CONFIG:
        error_msg = (
            "无法创建SQLite数据提供者。请检查：\n"
            "1. 配置文件 configs/data/providers/sqlite.yaml 是否正确\n"
            "2. SQLite配置是否启用\n"
            "3. 配置格式是否正确"
        )
        logger.critical(error_msg)
        raise RuntimeError(error_msg)

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
    logger.debug("\n--- 步骤 2: 导出 Backtrader 数据 ---")

    try:
        from backtest.data.exporter import BTDataExporter
    except ImportError:
        try:
            from old_code.bt.data.exporter import BTDataExporter
        except ImportError:
            logger.error("无法导入 BTDataExporter，回测终止")
            return

    # 获取因子数据导出配置
    factor_analysis_config = config_loader.load_factor_analysis()
    factor_data_dir = factor_analysis_config.output.get('export_composite_factor', {}).get('dir', 'temp/data_explore/')
    
    # =========================================================================
    # 3. 检查因子数据路径是否存在并进行数据验证
    # =========================================================================
    logger.info(f"\n--- 检查因子数据路径: {factor_data_dir} ---")
    
    if not os.path.exists(factor_data_dir):
        logger.warning(f"⚠️ 因子数据目录不存在: {factor_data_dir}")
        parquet_files = []
    else:
        # 检查目录中是否存在 .parquet 文件
        parquet_files = [f for f in os.listdir(factor_data_dir) if f.endswith('.parquet')]
    
    if not parquet_files:
        logger.warning(f"⚠️ 在 {factor_data_dir} 中未找到任何 .parquet 文件")
        dates_in_files = []
    else:
        # 提取所有文件的日期
        dates_in_files = []
        date_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2})\.parquet$')
        
        for f in parquet_files:
            match = date_pattern.match(f)
            if match:
                try:
                    date_str = match.group(1)
                    date_obj = pd.to_datetime(date_str)
                    dates_in_files.append(date_obj)
                except Exception as e:
                    logger.debug(f"无法解析文件名中的日期 {f}: {e}")
                    continue
            else:
                logger.debug(f"跳过非日期格式文件: {f}")
    
    if dates_in_files:
        min_date = min(dates_in_files)
        max_date = max(dates_in_files)
        logger.info(f"✅ 找到 {len(dates_in_files)} 个有效的因子数据文件")
        logger.info(f"📊 文件日期范围: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
        
        # 检查所需的日期范围
        start_dt = pd.to_datetime(START_DATE)
        end_dt = pd.to_datetime(END_DATE)
        
        if start_dt < min_date:
            logger.warning(f"⚠️ 回测开始日期 {START_DATE} 早于因子数据最早日期 {min_date.strftime('%Y-%m-%d')}")
        if end_dt > max_date:
            logger.warning(f"⚠️ 回测结束日期 {END_DATE} 晚于因子数据最晚日期 {max_date.strftime('%Y-%m-%d')}")
        
        # 检查是否至少有一部分数据是重叠的
        if end_dt < min_date or start_dt > max_date:
            logger.warning(f"⚠️ 回测日期范围 [{START_DATE}, {END_DATE}] 与因子数据日期范围无重叠")
        
    else:
        logger.warning("⚠️ 无法从因子数据目录解析出任何有效日期，将尝试继续（可能导致回测无数据）")

    exporter = BTDataExporter(data_manager, factor_data_dir=factor_data_dir)

    logger.info(f"导出数据，股票数: {len(universe)}")
    exporter.export(
        universe=universe,
        start_date=START_DATE,
        end_date=END_DATE,
        factor_series=None  # 自动从 factor_data_dir 加载
    )

    # =========================================================================
    # 4. 运行 Backtrader 回测
    # =========================================================================
    logger.debug("\n--- 步骤 3: 运行 Backtrader ---")

    try:
        from old_code.bt.backtest import run_backtest as bt_run_backtest

        # 兼容旧版本签名：如果旧函数不支持 generate_report 参数，
        # 则不要在本脚本里再次生成报告，避免"✅ 报告已生成 ..."重复。
        # 使用模块级缓存避免重复的反射开销
        _bt_supports_generate_report = _check_bt_supports_generate_report(bt_run_backtest)
        _skip_report_generation = not _bt_supports_generate_report

        # 移除 tqdm 进度条，恢复最原始的时间步打印
        # 尽量禁用 old_code.bt.backtest 内部报告生成与结束提示（由本脚本统一输出，避免重复）
        if _bt_supports_generate_report:
            cerebro, results = bt_run_backtest(pbar=None, generate_report=False)
        else:
            cerebro, results = bt_run_backtest(pbar=None)

        logger.info("Backtrader 回测完成")

        # 记录结束阶段需要用到的资金信息
        bt_initial_value = getattr(getattr(cerebro, 'broker', None), 'startingcash', None)
        try:
            bt_final_value = cerebro.broker.getvalue() if getattr(cerebro, 'broker', None) is not None else None
        except Exception:
            bt_final_value = None
    except ImportError as e:
        logger.warning(f"导入 Backtrader 模块失败: {e}")
        logger.info("尝试使用内置简化回测...")
        report_path, simple_final_value = _run_simple_backtest(
            data_manager,
            universe[:50],
            START_DATE,
            END_DATE,
            INITIAL_CASH,
            OUTPUT_DIR,
        )
        if report_path:
            logger.info(f"✅ 报告已生成: {report_path}")
        logger.info("✅ 回测流程完成！")
        return
    except Exception as e:
        logger.error(f"Backtrader 回测失败: {e}", exc_info=True)
        return

    # =========================================================================
    # 5. 生成报告
    # =========================================================================
    try:
        from backtest.reports import ReportGenerator
    except ImportError:
        try:
            from old_code.bt.utils.report_generator import ReportGenerator
        except ImportError:
            logger.warning("无法导入报告生成器")
            ReportGenerator = None

    if not _skip_report_generation and ReportGenerator is not None and cerebro is not None:
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
                if hasattr(strat.analyzers, 'trades'):
                    analyzers['trades'] = strat.analyzers.trades.get_analysis()

        # 报告生成器内部会输出一次"✅ 报告已生成: ..."，此处不重复输出
        _ = report_gen.generate(cerebro, strat if results else None, analyzers)

    # =========================================================================
    # 6. 输出回测结束的基本信息
    # =========================================================================
    
    if cerebro is not None:
        try:
            # 获取基本资金信息
            initial_cash = cerebro.broker.startingcash
            final_value = cerebro.broker.getvalue()
            total_return = (final_value / initial_cash - 1) * 100
            absolute_return = final_value - initial_cash
            
            # 获取交易统计信息
            total_trades = 0
            if results and len(results) > 0:
                strat = results[0]
                if hasattr(strat, 'analyzers') and hasattr(strat.analyzers, 'trades'):
                    trades_analysis = strat.analyzers.trades.get_analysis()
                    total_trades = trades_analysis.get('total', {}).get('total', 0)
            
            # 获取风险指标
            sharpe_ratio = "N/A"
            max_drawdown = "N/A"
            
            if analyzers:
                # Sharpe Ratio
                sharpe_analysis = analyzers.get('sharpe', {})
                if sharpe_analysis:
                    sharpe_value = sharpe_analysis.get('sharperatio')
                    if sharpe_value is not None:
                        sharpe_ratio = f"{sharpe_value:.3f}"
                
                # 最大回撤
                drawdown_analysis = analyzers.get('drawdown', {})
                if drawdown_analysis:
                    max_dd = drawdown_analysis.get('max', {}).get('drawdown')
                    if max_dd is not None:
                        max_drawdown = f"{max_dd:.2f}%"
            
            # 输出回测基本信息
            logger.info("=" * 60)
            logger.info("📊 本次回测基本信息")
            logger.info("=" * 60)
            logger.info(f"📅 回测期间: {START_DATE} ~ {END_DATE}")
            logger.info(f"💰 初始资金: {initial_cash:,.2f} 元")
            logger.info(f"💎 最终净值: {final_value:,.2f} 元")
            logger.info(f"📈 绝对收益: {absolute_return:+,.2f} 元")
            logger.info(f"📊 收益率: {total_return:+.2f}%")
            logger.info(f"📉 最大回撤: {max_drawdown}")
            logger.info(f"⚡ Sharpe比率: {sharpe_ratio}")
            logger.info(f"🔄 交易次数: {total_trades}")
            logger.info(f"🎯 股票池大小: {len(universe)} 只")
            logger.info(f"💸 手续费率: {COMMISSION:.4f}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.warning(f"获取回测统计信息时出错: {e}")

    logger.info("✅ 回测流程完成！")


def _run_simple_backtest(data_manager, universe, start_date, end_date, initial_cash, output_dir):
    """
    简化回测（当 Backtrader 不可用时使用）
    """
    import pandas as pd
    import datetime

    logger.debug("运行简化回测...")

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
        return None, None

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

    logger.debug("简化回测结果:")
    logger.debug(f"  总收益率: {total_return:.2f}%")
    logger.debug(f"  Sharpe Ratio: {sharpe:.3f}")
    logger.debug(f"  最大回撤: {max_drawdown:.2f}%")
    logger.debug(f"  最终净值: {portfolio_value.iloc[-1]:,.2f}")

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

    # 简化回测的报告路径由上层统一输出，避免结束阶段重复/冗余
    logger.debug(f"简化报告已生成: {report_path}")
    return report_path, float(portfolio_value.iloc[-1])


if __name__ == '__main__':
    from core.config import load_config
    config_loader = load_config()
    run_backtest(config_loader)
