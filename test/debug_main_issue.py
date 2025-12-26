#!/usr/bin/env python3
"""
模拟主程序中的因子数据结构来调试问题
"""

import os
import sys
import pandas as pd
import logging
from data import DataProviderManager
from data.providers import SQLiteDataProvider
from backtest.data.exporter import BTDataExporter

# 设置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置
from universe_config import UNIVERSE

# 数据源配置
DATA_PROVIDERS_CONFIG = [
    ('sqlite_jy', SQLiteDataProvider, {
        'db_path': './database/JY_database/sqlite/JY_database.sqlite',
        'table_name': 'JY_t_price_daily'
    }),
]

START_DATE = '2024-01-01'
END_DATE = '2024-12-31'
BACKTEST_DB_PATH = './database/quant_data.db'

def test_main_scenario():
    """模拟主程序中的场景"""
    logging.info("=== 模拟主程序场景 ===")
    
    # 初始化数据管理器
    logging.info("初始化DataProviderManager...")
    data_manager = DataProviderManager(
        provider_configs=DATA_PROVIDERS_CONFIG,
        symbols=UNIVERSE[:3],  # 只测试前3个股票
        start_date=START_DATE,
        end_date=END_DATE,
        db_path=BACKTEST_DB_PATH
    )
    
    # 创建一个模拟的composite_factor_series，类似主程序中的结构
    # 模拟多级索引 (date, asset) 的因子数据
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    assets = UNIVERSE[:3]
    
    # 创建MultiIndex
    multi_index = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    
    # 创建模拟因子值
    factor_values = pd.Series(range(len(multi_index)), index=multi_index, name='factor_value')
    
    logging.info(f"模拟因子数据形状: {factor_values.shape}")
    logging.info(f"模拟因子数据索引: {factor_values.index.names}")
    logging.info(f"模拟因子数据样本:")
    logging.info(f"  前3个索引: {list(factor_values.index[:3])}")
    logging.info(f"  第一个索引值: {factor_values.index[0]} (类型: {type(factor_values.index[0])})")
    
    # 测试BTDataExporter
    logging.info("测试BTDataExporter...")
    exporter = BTDataExporter(data_manager)
    
    try:
        result = exporter.export(
            universe=UNIVERSE[:3],
            start_date=START_DATE,
            end_date=END_DATE,
            factor_series=factor_values
        )
        logging.info(f"✅ 导出完成，成功导出 {len(result)} 个文件")
    except Exception as e:
        logging.error(f"❌ 导出错误: {e}")
        import traceback
        logging.error(f"详细错误: {traceback.format_exc()}")

if __name__ == "__main__":
    test_main_scenario()