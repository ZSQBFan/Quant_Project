#!/usr/bin/env python3
"""
调试测试 - 找出 _batch_load_all_data 的问题
"""

import pandas as pd
import numpy as np
import tempfile
import os
import sys
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data.exporter import BTDataExporter

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

class MockDataManager:
    """模拟 DataManager"""
    
    def __init__(self, data_dict):
        self.data_dict = data_dict
    
    def get_all_data_for_universe(self, universe, required_columns=None):
        """批量获取所有股票数据"""
        if not self.data_dict:
            return None
        
        # 方法1: 使用 concat - 这是问题所在
        all_data = pd.concat(self.data_dict.values(), keys=self.data_dict.keys())
        all_data.index.names = ['date', 'asset']
        print(f"[MockDataManager] 合并后的数据:")
        print(f"  - 索引类型: {type(all_data.index)}")
        print(f"  - 索引级别: {all_data.index.names}")
        print(f"  - 数据行数: {len(all_data)}")
        print(f"  - 前5行:\n{all_data.head()}")
        return all_data


def debug_batch_load():
    """调试批量加载"""
    print("=" * 60)
    print("调试测试: _batch_load_all_data")
    print("=" * 60)
    
    # 创建模拟数据
    start_date = '2024-01-01'
    end_date = '2024-01-31'
    
    data_dict = {}
    for i in range(5):
        asset = f'{600000 + i}'
        full_index = pd.date_range(start_date, end_date, freq='B')
        trading_dates = full_index[:10]  # 前10个交易日
        
        asset_data = pd.DataFrame({
            'open': np.random.randn(len(trading_dates)).cumsum() + 100,
            'high': np.random.randn(len(trading_dates)).cumsum() + 102,
            'low': np.random.randn(len(trading_dates)).cumsum() + 98,
            'close': np.random.randn(len(trading_dates)).cumsum() + 100,
            'volume': np.random.randint(10000, 100000, len(trading_dates))
        }, index=trading_dates)
        
        data_dict[asset] = asset_data
    
    universe = list(data_dict.keys())
    print(f"股票列表: {universe}")
    
    # 创建 mock DataManager
    mock_dm = MockDataManager(data_dict)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = BTDataExporter(mock_dm, output_dir=tmpdir)
        
        # 测试 _batch_load_all_data
        print("\n测试 _batch_load_all_data...")
        result_dict = exporter._batch_load_all_data(universe, start_date, end_date)
        print(f"返回的股票数量: {len(result_dict)}")
        print(f"返回的股票: {list(result_dict.keys())}")


if __name__ == '__main__':
    debug_batch_load()
