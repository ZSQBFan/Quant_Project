"""
数据下载脚本

下载和更新股票数据。
配置从 configs/data/ 目录加载。
"""

import logging
import pandas as pd

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

    logger.info(f"股票池大小: {len(universe)}")
    logger.info(f"日期范围: {START_DATE} ~ {END_DATE}")
    logger.info(f"目标数据库路径: {DB_PATH}")

    # 初始化目标数据库处理器
    from data.handlers import DatabaseHandler
    target_db_handler = DatabaseHandler(db_path=DB_PATH)

    # 根据 provider_priority 顺序构建数据提供者配置
    for provider_name in data_config.provider_priority:
        provider_cfg = providers_config.get(provider_name)
        if provider_cfg and provider_cfg.enabled:
            logger.info(f"✅ 正在配置数据提供者: {provider_name}")
            
            if provider_name.startswith('sqlite'):
                # SQLite 数据提供者配置
                conn_cfg = provider_cfg.config.get('connection', {})
                tables_cfg = provider_cfg.config.get('tables', {}).get('daily', {})
                
                # 应用完整的配置处理逻辑，包括column_mapping
                processed_kwargs = {
                    'db_path': conn_cfg.get('db_path'),
                    'table_name': tables_cfg.get('table_name', 'stock_daily_prices'),
                    'column_mapping': tables_cfg.get('column_mapping', {}),
                    'date_format': tables_cfg.get('date_format', '%Y-%m-%d')
                }
                
                # 实例化SQLite数据提供者
                from data.providers import SQLiteDataProvider
                sqlite_provider = SQLiteDataProvider(**processed_kwargs)
                logger.info(f"✅ SQLite数据提供者已实例化: {provider_name}, db_path={processed_kwargs['db_path']}")
                
                # 使用全量股票数据获取方法
                logger.info("开始使用全量股票数据获取方法...")
                try:
                    # 获取全股票数据
                    all_data = sqlite_provider.get_all_stock_data(
                        start_date=START_DATE,
                        end_date=END_DATE,
                        required_columns=['open', 'high', 'low', 'close', 'volume', 'turnover', 'pct_change']
                    )
                    
                    if all_data is not None and not all_data.empty:
                        logger.info(f"成功获取全股票数据: {len(all_data)} 条记录")
                        logger.info(f"股票数量: {all_data.index.get_level_values('asset').nunique()}")
                        logger.info(f"日期数量: {all_data.index.get_level_values('date').nunique()}")
                        
                        # 转换数据格式以适应目标数据库
                        # 重置索引以获取列格式的数据
                        df_to_save = all_data.reset_index()
                        
                        # 确保列名符合目标数据库要求
                        column_mapping = {
                            'asset': 'code',
                            'open': 'open',
                            'high': 'high',
                            'low': 'low',
                            'close': 'close',
                            'volume': 'volume',
                            'turnover': 'turnover',
                            'pct_change': 'pct_change'
                        }
                        
                        # 重命名列
                        df_to_save.rename(columns=column_mapping, inplace=True)
                        
                        # 计算额外的技术指标
                        df_to_save['price_change'] = df_to_save['close'] - df_to_save['open']
                        df_to_save['amplitude'] = (df_to_save['high'] - df_to_save['low']) / df_to_save['open'] * 100
                        df_to_save['turnover_rate'] = None  # 暂未计算
                        
                        # 数据去重
                        initial_count = len(df_to_save)
                        df_to_save = df_to_save.drop_duplicates(subset=['code', 'date'], keep='last')
                        final_count = len(df_to_save)
                        if initial_count != final_count:
                            logger.info(f"数据去重: 移除了 {initial_count - final_count} 条重复记录")
                        
                        # 保存到目标数据库
                        logger.info(f"开始保存数据到 {DB_PATH}...")
                        target_db_handler.save_data(df_to_save, 'stock_daily_prices')
                        
                        logger.info("=" * 60)
                        logger.info("数据下载流程执行完毕")
                        logger.info(f"✅ 成功保存 {final_count} 条股票数据")
                        logger.info(f"✅ 涵盖 {df_to_save['code'].nunique()} 只股票")
                        logger.info(f"✅ 时间范围: {START_DATE} ~ {END_DATE}")
                        logger.info("=" * 60)
                        
                    else:
                        logger.warning("未能获取到任何股票数据")
                        return
                        
                except Exception as e:
                    logger.error(f"全量股票数据获取失败: {e}", exc_info=True)
                    return
                    
                # 数据导入完成，退出循环
                break
            else:
                # 其他类型的数据提供者（如 tushare, akshare）
                logger.warning(f"❌ 数据提供者类型 '{provider_name}' 暂未实现")

    # 如果没有配置，抛出异常
    else:
        error_msg = (
            f"无法创建数据提供者。从provider_priority获取的配置: {data_config.provider_priority}\n"
            f"但实际可用的提供者只有: {list(providers_config.keys())}\n"
            "请检查：\n"
            f"1. 数据提供者配置文件是否存在: configs/data/providers/{data_config.provider_priority}.yaml\n"
            "2. 配置中的数据提供者是否启用\n"
            "3. 配置格式是否正确"
        )
        logger.critical(error_msg)
        raise RuntimeError(error_msg)

    # 关闭数据库连接
    target_db_handler.close_connection()


if __name__ == '__main__':
    download_data()
