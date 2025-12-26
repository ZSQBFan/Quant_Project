#!/usr/bin/env python3
"""
调试tuple转datetime问题的简单测试脚本
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

def test_data_export():
    """测试数据导出功能"""
    logging.info("=== 开始调试tuple转datetime问题 ===")
    
    # 初始化数据管理器
    logging.info("初始化DataProviderManager...")
    data_manager = DataProviderManager(
        provider_configs=DATA_PROVIDERS_CONFIG,
        symbols=UNIVERSE[:5],  # 只测试前5个股票
        start_date=START_DATE,
        end_date=END_DATE,
        db_path=BACKTEST_DB_PATH
    )
    
    # 测试单个股票
    test_asset = UNIVERSE[0]
    logging.info(f"测试股票: {test_asset}")
    
    # 获取数据
    logging.info("从DataManager获取数据...")
    prices = data_manager.get_dataframe(
        test_asset,
        columns=['open', 'high', 'low', 'close', 'volume']
    )
    
    if prices is not None and not prices.empty:
        logging.info(f"获取到 {len(prices)} 行数据")
        logging.info(f"数据索引类型: {type(prices.index)}")
        logging.info(f"数据索引名称: {prices.index.names if hasattr(prices.index, 'names') else prices.index.name}")
        logging.info(f"前几行索引: {list(prices.index[:3])}")
        
        # 检查第一个索引值
        if len(prices.index) > 0:
            first_idx = prices.index[0]
            logging.info(f"第一个索引值: {first_idx} (类型: {type(first_idx)})")
            
            if isinstance(first_idx, tuple):
                logging.info(f"tuple内容: {first_idx}")
                logging.info(f"tuple各元素类型: {[type(x) for x in first_idx]}")
        
        # 测试导出
        logging.info("测试BTDataExporter...")
        exporter = BTDataExporter(data_manager)
        
        # 创建一个简单的因子序列
        factor_series = pd.Series([1.0] * len(prices), index=prices.index)
        factor_series.name = 'combined_signal'
        
        try:
            result = exporter._export_single_asset(test_asset, START_DATE, END_DATE, factor_series)
            if result:
                logging.info(f"✅ 导出成功: {result}")
            else:
                logging.info("❌ 导出失败")
        except Exception as e:
            logging.error(f"❌ 导出错误: {e}")
            import traceback
            logging.error(f"详细错误: {traceback.format_exc()}")
    else:
        logging.error("无法获取数据")

if __name__ == "__main__":
    test_data_export()