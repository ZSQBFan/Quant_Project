"""
数据下载脚本

下载和更新股票数据。
配置从 configs/data/ 目录加载。
"""

import logging

logger = logging.getLogger(__name__)


def download_data(config_loader=None):
    """
    执行数据下载

    Args:
        config_loader: 配置加载器实例（可选）
    """
    logger.info("=" * 60)
    logger.info("开始数据下载流程...")
    logger.info("=" * 60)

    # 加载配置
    if config_loader is None:
        from core.config import load_config
        config_loader = load_config()

    # 加载各类配置
    universe = config_loader.load_universe()
    backtest_config = config_loader.load_backtest()
    data_config = config_loader.load_data()
    providers_config = config_loader.load_providers()

    # 获取日期范围（优先使用data配置，否则使用backtest配置）
    START_DATE = data_config.start_date or backtest_config.start_date
    END_DATE = data_config.end_date or backtest_config.end_date

    # 获取数据库配置
    DB_PATH = data_config.database.get('db_path', './database/quant_data.db')

    # 获取下载配置
    download_cfg = data_config.download
    NUM_CHECKER_THREADS = download_cfg.get('num_checker_threads', 16)
    NUM_DOWNLOADER_THREADS = download_cfg.get('num_downloader_threads', 16)
    BATCH_SIZE = download_cfg.get('batch_size', 200)

    logger.info(f"股票池大小: {len(universe)}")
    logger.info(f"日期范围: {START_DATE} ~ {END_DATE}")
    logger.info(f"数据库路径: {DB_PATH}")

    # 初始化数据管理器
    try:
        from data import DataProviderManager
        from data.providers import SQLiteDataProvider
    except ImportError as e:
        logger.warning(f"导入新数据模块失败: {e}")
        logger.info("尝试从 old_code 导入...")
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

    logger.info("初始化 DataProviderManager...")
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

    # 执行数据准备
    logger.info("开始数据准备（检查、下载、写入）...")
    data_manager.prepare_data_for_universe()

    logger.info("=" * 60)
    logger.info("数据下载流程执行完毕")
    logger.info("=" * 60)


if __name__ == '__main__':
    download_data()
