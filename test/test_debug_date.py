#!/usr/bin/env python3
"""
调试date字段问题的简单脚本
"""

import logging
import sys
import os

# 设置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - [%(levelname)s] - (%(thread)d) - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_data_download():
    """测试数据下载过程中的date字段问题"""
    try:
        # 直接导入数据管理器
        from data.manager import DataProviderManager
        from data.providers import SQLiteDataProvider
        
        # 配置数据提供者
        DATA_PROVIDERS_CONFIG = [
            (
                SQLiteDataProvider,
                {
                    'db_path': './database/JY_database/sqlite/JY_database.sqlite',
                    'table_name': 'JY_t_price_daily'
                }
            ),
        ]
        
        # 测试股票列表（小批量）
        test_symbols = ['000001', '000002']
        
        # 初始化数据管理器
        data_manager = DataProviderManager(
            provider_configs=DATA_PROVIDERS_CONFIG,
            symbols=test_symbols,
            start_date='2024-01-01',
            end_date='2024-01-31',
            db_path='./test_quant_data.db',
            num_checker_threads=1,
            num_downloader_threads=1,
            batch_size=2
        )
        
        # 执行数据准备
        print("开始数据准备...")
        data_manager.prepare_data_for_universe()
        print("数据准备完成")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_data_download()