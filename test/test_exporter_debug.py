#!/usr/bin/env python3
"""
调试测试 - 找出导出失败的原因
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
        
        # 合并所有数据
        all_data = pd.concat(self.data_dict.values(), keys=self.data_dict.keys())
        all_data.index.names = ['date', 'asset']
        return all_data


def debug_test():
    """调试测试"""
    print("=" * 60)
    print("调试测试: 找出导出失败的原因")
    print("=" * 60)
    
    # 创建模拟数据
    start_date = '2024-01-01'
    end_date = '2024-01-31'
    full_index = pd.date_range(start_date, end_date, freq='B')
    
    # 创建一只股票的数据
    trading_dates = full_index[:10]  # 前10个交易日
    asset_data = pd.DataFrame({
        'open': np.random.randn(len(trading_dates)).cumsum() + 100,
        'high': np.random.randn(len(trading_dates)).cumsum() + 102,
        'low': np.random.randn(len(trading_dates)).cumsum() + 98,
        'close': np.random.randn(len(trading_dates)).cumsum() + 100,
        'volume': np.random.randint(10000, 100000, len(trading_dates))
    }, index=trading_dates)
    
    print(f"原始数据索引类型: {type(asset_data.index)}")
    print(f"原始数据索引: {asset_data.index[:3]}")
    
    # 创建因子数据
    factor_data = []
    for date in trading_dates:
        factor_data.append({
            'date': date,
            'asset': '600000',
            'factor_value': np.random.randn()
        })
    factor_df = pd.DataFrame(factor_data)
    factor_df.set_index(['date', 'asset'], inplace=True)
    factor_series = factor_df['factor_value']
    
    print(f"因子数据索引类型: {type(factor_series.index)}")
    
    # 创建 mock DataManager
    mock_dm = MockDataManager({'600000': asset_data})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = BTDataExporter(mock_dm, output_dir=tmpdir)
        
        # 预处理因子数据
        print("\n预处理因子数据...")
        factor_dict = exporter._preprocess_factor_dict(factor_series, full_index)
        print(f"预处理后的因子字典: {list(factor_dict.keys())}")
        
        # 尝试导出
        print("\n尝试导出...")
        try:
            result = exporter._export_single_asset_optimized(
                '600000',
                asset_data.copy(),
                start_date,
                end_date,
                factor_dict.get('600000')
            )
            print(f"导出结果: {result}")
        except Exception as e:
            print(f"导出异常: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    debug_test()
