#!/usr/bin/env python3
"""
测试 exporter.py 修复 - 验证 Length of values does not match length of index 问题已解决
"""

import pandas as pd
import numpy as np
import tempfile
import os
import sys

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
        
        # 合并所有数据
        all_data = pd.concat(self.data_dict.values(), keys=self.data_dict.keys())
        all_data.index.names = ['date', 'asset']
        return all_data


def test_new_stock_reindex_issue():
    """
    测试新股场景：原始数据行数 < full_index 行数
    这测试了修复前的 bug: suspended_mask 在 reindex 前计算导致长度不匹配
    """
    print("=" * 60)
    print("测试: 新股场景 - 原始数据行数 < full_index 行数")
    print("=" * 60)
    
    # 创建模拟数据 - 模拟新股只有部分日期的数据
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    full_index = pd.date_range(start_date, end_date, freq='B')
    
    # 新股只有 100 个交易日的数据（上市较晚）
    trading_dates = full_index[200:300]  # 从第200个交易日开始（假设这是上市日期）
    
    # 创建原始数据（只有100行）
    asset_data = pd.DataFrame({
        'open': np.random.randn(len(trading_dates)).cumsum() + 100,
        'high': np.random.randn(len(trading_dates)).cumsum() + 102,
        'low': np.random.randn(len(trading_dates)).cumsum() + 98,
        'close': np.random.randn(len(trading_dates)).cumsum() + 100,
        'volume': np.random.randint(10000, 100000, len(trading_dates))
    }, index=trading_dates)
    
    print(f"原始数据行数: {len(asset_data)}")
    print(f"full_index 行数: {len(full_index)}")
    print(f"原始数据日期范围: {asset_data.index[0]} ~ {asset_data.index[-1]}")
    
    # 创建 mock DataManager
    mock_dm = MockDataManager({'603381': asset_data})
    
    # 创建导出器
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = BTDataExporter(mock_dm, output_dir=tmpdir)
        
        # 创建空的因子数据（使用新的 numpy 数组 API）
        asset_factors = None  # 无因子数据
        
        # 调用优化的导出方法（新 API：只接受 asset_factors numpy 数组）
        try:
            result = exporter._export_single_asset_optimized(
                '603381', 
                asset_data.copy(),
                start_date, 
                end_date, 
                asset_factors  # 新 API：预处理后的 numpy 数组
            )
            
            if result:
                print(f"✅ 导出成功: {result}")
                
                # 验证导出文件
                exported_df = pd.read_parquet(result)
                print(f"✅ 导出数据行数: {len(exported_df)}")
                print(f"✅ 导出数据日期范围: {exported_df.index[0]} ~ {exported_df.index[-1]}")
                print(f"✅ suspended 列非空: {not exported_df['suspended'].empty}")
                print(f"✅ suspended 列长度匹配: {len(exported_df['suspended']) == len(exported_df)}")
                
                # 检查是否有 NaN
                if exported_df['close'].isna().any():
                    print("⚠️ 导出数据中存在 NaN（可能是填充后仍缺失数据的日期）")
                else:
                    print("✅ 导出数据无 NaN")
                
                return True
            else:
                print("❌ 导出返回 None")
                return False
                
        except ValueError as e:
            if "Length of values" in str(e):
                print(f"❌ 修复失败，仍存在长度不匹配问题: {e}")
                return False
            raise
        except Exception as e:
            print(f"❌ 导出错误: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_normal_stock_reindex():
    """
    测试正常股票场景：原始数据行数 ≈ full_index 行数
    """
    print("\n" + "=" * 60)
    print("测试: 正常股票场景 - 原始数据行数 ≈ full_index 行数")
    print("=" * 60)
    
    start_date = '2024-01-01'
    end_date = '2024-03-31'
    full_index = pd.date_range(start_date, end_date, freq='B')
    
    # 正常股票有大部分日期的数据
    trading_dates = full_index[:200]  # 前200个交易日
    
    asset_data = pd.DataFrame({
        'open': np.random.randn(len(trading_dates)).cumsum() + 100,
        'high': np.random.randn(len(trading_dates)).cumsum() + 102,
        'low': np.random.randn(len(trading_dates)).cumsum() + 98,
        'close': np.random.randn(len(trading_dates)).cumsum() + 100,
        'volume': np.random.randint(10000, 100000, len(trading_dates))
    }, index=trading_dates)
    
    print(f"原始数据行数: {len(asset_data)}")
    print(f"full_index 行数: {len(full_index)}")
    
    mock_dm = MockDataManager({'600000': asset_data})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = BTDataExporter(mock_dm, output_dir=tmpdir)
        
        try:
            result = exporter._export_single_asset_optimized(
                '600000',
                asset_data.copy(),
                start_date,
                end_date,
                None  # 无因子数据
            )
            
            if result:
                print(f"✅ 导出成功: {result}")
                exported_df = pd.read_parquet(result)
                print(f"✅ 导出数据行数: {len(exported_df)}")
                return True
            else:
                print("❌ 导出返回 None")
                return False
                
        except Exception as e:
            print(f"❌ 导出错误: {e}")
            return False


def test_suspended_stock():
    """
    测试停牌股场景：数据中有 NaN（停牌日期）
    """
    print("\n" + "=" * 60)
    print("测试: 停牌股场景 - 数据中有 NaN（停牌日期）")
    print("=" * 60)
    
    start_date = '2024-01-01'
    end_date = '2024-01-31'
    full_index = pd.date_range(start_date, end_date, freq='B')
    
    trading_dates = full_index[:15]  # 前15个交易日有数据
    
    # 创建包含 NaN 的数据（停牌）
    closes = np.random.randn(len(trading_dates)).cumsum() + 100
    closes[5:8] = np.nan  # 停牌3天
    
    asset_data = pd.DataFrame({
        'open': np.random.randn(len(trading_dates)).cumsum() + 100,
        'high': np.random.randn(len(trading_dates)).cumsum() + 102,
        'low': np.random.randn(len(trading_dates)).cumsum() + 98,
        'close': closes,
        'volume': np.random.randint(10000, 100000, len(trading_dates))
    }, index=trading_dates)
    
    print(f"原始数据中的 NaN 数量: {asset_data['close'].isna().sum()}")
    
    mock_dm = MockDataManager({'000001': asset_data})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = BTDataExporter(mock_dm, output_dir=tmpdir)
        
        try:
            result = exporter._export_single_asset_optimized(
                '000001',
                asset_data.copy(),
                start_date,
                end_date,
                None  # 无因子数据
            )
            
            if result:
                print(f"✅ 导出成功: {result}")
                exported_df = pd.read_parquet(result)
                print(f"✅ 导出数据行数: {len(exported_df)}")
                print(f"✅ 停牌标记数量: {exported_df['suspended'].sum()}")
                return True
            else:
                print("❌ 导出返回 None（可能是填充后仍有NaN）")
                return False
                
        except Exception as e:
            print(f"❌ 导出错误: {e}")
            return False


if __name__ == '__main__':
    print("\n开始测试 exporter.py 修复\n")
    
    results = []
    
    # 运行所有测试
    results.append(("新股场景 (长度不匹配修复)", test_new_stock_reindex_issue()))
    results.append(("正常股票场景", test_normal_stock_reindex()))
    results.append(("停牌股场景", test_suspended_stock()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 所有测试通过！修复验证成功！")
        sys.exit(0)
    else:
        print("⚠️ 部分测试失败，需要进一步调查")
        sys.exit(1)
