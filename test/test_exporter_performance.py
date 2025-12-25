#!/usr/bin/env python3
"""
性能测试 - 验证 exporter.py 优化后的性能
"""

import pandas as pd
import numpy as np
import tempfile
import os
import sys
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data.exporter import BTDataExporter


class MockDataManager:
    """模拟 DataManager"""
    
    def __init__(self, data_dict):
        self.data_dict = data_dict
    
    def get_all_data_for_universe(self, universe, required_columns=None):
        """批量获取所有股票数据"""
        if not self.data_dict:
            return None
        
        # 使用正确的索引顺序: (date, asset)
        all_data = []
        for asset, asset_data in self.data_dict.items():
            asset_data_copy = asset_data.copy()
            asset_data_copy['asset'] = asset
            all_data.append(asset_data_copy)
        
        combined_df = pd.concat(all_data)
        combined_df.index.name = 'date'
        combined_df = combined_df.reset_index().set_index(['date', 'asset'])
        return combined_df


def create_mock_data(n_stocks=100, n_days=200):
    """创建模拟数据"""
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    full_index = pd.date_range(start_date, end_date, freq='B')
    
    data_dict = {}
    for i in range(n_stocks):
        asset = f'{600000 + i}'
        # 每只股票有部分日期的数据（从不同起始点开始）
        start_idx = np.random.randint(0, len(full_index) - n_days)
        trading_dates = full_index[start_idx:start_idx + n_days]
        
        asset_data = pd.DataFrame({
            'open': np.random.randn(len(trading_dates)).cumsum() + 100,
            'high': np.random.randn(len(trading_dates)).cumsum() + 102,
            'low': np.random.randn(len(trading_dates)).cumsum() + 98,
            'close': np.random.randn(len(trading_dates)).cumsum() + 100,
            'volume': np.random.randint(10000, 100000, len(trading_dates))
        }, index=trading_dates)
        
        data_dict[asset] = asset_data
    
    return data_dict, full_index


def create_mock_factor_series(data_dict, full_index):
    """创建模拟因子数据（确保没有重复索引）"""
    all_factors = []
    for asset, asset_data in data_dict.items():
        for date in asset_data.index:
            factor_value = np.random.randn()
            all_factors.append({
                'date': pd.Timestamp(date),  # 确保是 Timestamp
                'asset': asset,
                'factor_value': factor_value
            })
    
    factor_df = pd.DataFrame(all_factors)
    # 确保索引唯一
    factor_df = factor_df.drop_duplicates(subset=['date', 'asset'])
    factor_df.set_index(['date', 'asset'], inplace=True)
    return factor_df['factor_value']


def test_performance():
    """性能测试"""
    print("=" * 60)
    print("性能测试: 验证 exporter.py 优化后的性能")
    print("=" * 60)
    
    n_stocks = 500  # 测试500只股票
    print(f"测试规模: {n_stocks} 只股票")
    
    # 创建模拟数据
    print("创建模拟数据...")
    data_dict, full_index = create_mock_data(n_stocks=n_stocks)
    factor_series = create_mock_factor_series(data_dict, full_index)
    print(f"因子数据点: {len(factor_series)}")
    print(f"因子数据索引类型: {type(factor_series.index)}")
    
    mock_dm = MockDataManager(data_dict)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = BTDataExporter(mock_dm, output_dir=tmpdir)
        
        # 预处理因子数据（主线程）
        print("\n预处理因子数据...")
        start_time = time.time()
        factor_dict = exporter._preprocess_factor_dict(factor_series, full_index)
        preprocess_time = time.time() - start_time
        print(f"预处理时间: {preprocess_time:.2f}秒")
        print(f"预处理后的因子字典: {len(factor_dict)} 只股票")
        
        # 导出数据
        universe = list(data_dict.keys())
        print(f"\n开始导出 {len(universe)} 只股票...")
        
        start_time = time.time()
        exported_files = exporter.export(
            universe, 
            '2024-01-01', 
            '2024-12-31',
            factor_series,  # 传入原始 factor_series，内部会预处理
            max_workers=16
        )
        export_time = time.time() - start_time
        
        n_exported = len(exported_files)
        print(f"\n导出结果:")
        print(f"  - 成功导出: {n_exported} 只")
        print(f"  - 总耗时: {export_time:.2f}秒")
        if export_time > 0:
            print(f"  - 平均速度: {n_exported / export_time:.1f} 只/秒")
        else:
            print(f"  - 平均速度: N/A (耗时太短)")
        
        # 计算预期提升
        old_speed = 12.8  # 原始速度
        if export_time > 0:
            new_speed = n_exported / export_time
            speedup = new_speed / old_speed
            
            print(f"\n性能对比:")
            print(f"  - 原始速度: {old_speed} 只/秒")
            print(f"  - 优化后速度: {new_speed:.1f} 只/秒")
            print(f"  - 速度提升: {speedup:.1f}x")
            
            if new_speed >= 50:
                print(f"\n✅ 达到目标: 50 只/秒以上")
            else:
                print(f"\n⚠️ 未达目标: 50 只/秒 (当前: {new_speed:.1f} 只/秒)")
        else:
            print(f"\n⚠️ 导出耗时太短，无法计算速度")
        
        return n_exported


if __name__ == '__main__':
    print("\n开始性能测试\n")
    
    n_exported = test_performance()
    
    print("\n" + "=" * 60)
    if n_exported > 0:
        print("🎉 性能优化成功！导出功能正常！")
        sys.exit(0)
    else:
        print(f"⚠️ 导出失败，未能成功导出任何股票")
        sys.exit(1)
