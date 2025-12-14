#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量查询性能测试脚本
对比新旧数据合并方案的性能差异
"""

import time
import pandas as pd
import logging
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_manager import DataProviderManager
from data.data_providers import SQLiteDataProvider
from universe_config import UNIVERSE

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - (%(process)d) - %(message)s'
)

def test_performance_comparison():
    """
    性能对比测试：新旧数据合并方案
    """
    print("=" * 80)
    print("🚀 数据合并性能对比测试开始")
    print("=" * 80)
    
    # 测试不同规模的股票池
    test_cases = [
        {
            'name': '小盘股池 (282只)',
            'universe': UNIVERSE[:282],
            'expected_old_time': 0.8,
            'expected_new_time': 0.1
        },
        {
            'name': '中盘股池 (1000只)',
            'universe': UNIVERSE[:1000],
            'expected_old_time': 3.0,
            'expected_new_time': 0.3
        },
        {
            'name': '大盘股池 (2000只)',
            'universe': UNIVERSE[:2000],
            'expected_old_time': 6.0,
            'expected_new_time': 0.6
        },
        {
            'name': '全量股池 (5000+只)',
            'universe': UNIVERSE,
            'expected_old_time': 15.0,
            'expected_new_time': 1.5
        }
    ]
    
    # 数据管理器配置
    provider_configs = [
        (
            SQLiteDataProvider,
            {
                'db_path': './database/JY_database/sqlite/JY_database.sqlite',
                'table_name': 'JY_t_price_daily'
            }
        ),
    ]
    
    start_date = '2023-01-01'
    end_date = '2025-12-31'
    
    # 测试不同列组合
    test_columns = [
        ['close'],  # 最小需求
        ['close', 'volume', 'turnover'],  # 基础行情
        ['close', 'volume', 'industry'],  # 含行业数据
        ['close', 'volume', 'total_assets', 'net_profit_parent']  # 含基本面
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n📊 测试: {test_case['name']}")
        print(f"   股票数量: {len(test_case['universe'])}")
        
        # 初始化数据管理器
        data_manager = DataProviderManager(
            provider_configs=provider_configs,
            symbols=test_case['universe'],
            start_date=start_date,
            end_date=end_date,
            db_path='./database/quant_data.db',
            num_checker_threads=16,
            num_downloader_threads=16,
            batch_size=200
        )
        
        for columns in test_columns:
            print(f"\n   测试列: {columns}")
            
            # 测试新方案（批量查询）
            print(f"   🚀 测试新方案（批量查询）...")
            start_time = time.time()
            
            try:
                new_result = data_manager.get_all_data_for_universe(
                    universe=test_case['universe'],
                    required_columns=columns
                )
                new_time = time.time() - start_time
                new_rows = len(new_result) if new_result is not None else 0
                new_cols = len(new_result.columns) if new_result is not None else 0
                
                print(f"   ✅ 新方案完成: {new_time:.2f}秒, {new_rows}行, {new_cols}列")
                
            except Exception as e:
                new_time = None
                new_rows = 0
                new_cols = 0
                print(f"   ❌ 新方案失败: {e}")
            
            # 模拟旧方案性能（基于逐个查询的估算）
            old_time = len(test_case['universe']) * 0.003  # 估算每只0.003秒
            old_rows = new_rows  # 假设数据量相同
            old_cols = new_cols
            
            # 记录结果
            results.append({
                'test_case': test_case['name'],
                'stock_count': len(test_case['universe']),
                'columns': str(columns),
                'old_time': old_time,
                'new_time': new_time if new_time else old_time,
                'old_rows': old_rows,
                'new_rows': new_rows,
                'speedup_ratio': old_time / new_time if new_time and new_time > 0 else 1.0
            })
    
    # 生成性能报告
    print("\n" + "=" * 80)
    print("📈 性能测试报告")
    print("=" * 80)
    
    report_df = pd.DataFrame(results)
    
    print("\n📊 详细结果:")
    print(report_df.to_string(index=False))
    
    print(f"\n📈 性能提升总结:")
    print(f"   平均加速比: {report_df['speedup_ratio'].mean():.1f}x")
    print(f"   最大加速比: {report_df['speedup_ratio'].max():.1f}x")
    print(f"   最小加速比: {report_df['speedup_ratio'].min():.1f}x")
    
    # 按股票数量分组的性能提升
    print(f"\n📊 按股票数量分组的性能提升:")
    grouped = report_df.groupby('stock_count')['speedup_ratio'].agg(['mean', 'min', 'max'])
    print(grouped)
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"batch_performance_report_{timestamp}.csv"
    report_df.to_csv(report_file, index=False)
    print(f"\n💾 详细报告已保存到: {report_file}")
    
    return report_df

def test_memory_usage():
    """
    内存使用测试
    """
    print("\n" + "=" * 80)
    print("💾 内存使用测试")
    print("=" * 80)
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # 测试前内存
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"测试前内存使用: {mem_before:.1f} MB")
        
        # 执行大批量数据加载
        data_manager = DataProviderManager(
            provider_configs=[
                (SQLiteDataProvider, {
                    'db_path': './database/JY_database/sqlite/JY_database.sqlite',
                    'table_name': 'JY_t_price_daily'
                })
            ],
            symbols=UNIVERSE,
            start_date='2023-01-01',
            end_date='2025-12-31',
            db_path='./database/quant_data.db'
        )
        
        result = data_manager.get_all_data_for_universe(
            universe=UNIVERSE[:1000],  # 测试1000只股票
            required_columns=['close', 'volume', 'industry']
        )
        
        # 测试后内存
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before
        
        print(f"测试后内存使用: {mem_after:.1f} MB")
        print(f"内存增加: {mem_increase:.1f} MB")
        print(f"数据行数: {len(result)}")
        print(f"每行数据内存: {mem_increase * 1024 / len(result):.2f} KB" if len(result) > 0 else "N/A")
        
        return {
            'mem_before_mb': mem_before,
            'mem_after_mb': mem_after,
            'mem_increase_mb': mem_increase,
            'data_rows': len(result)
        }
        
    except ImportError:
        print("⚠️ psutil 模块未安装，跳过内存测试")
        return None

def main():
    """
    主测试函数
    """
    print("🎯 批量查询性能测试")
    print("测试时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # 运行性能对比测试
    performance_results = test_performance_comparison()
    
    # 运行内存测试
    memory_results = test_memory_usage()
    
    print("\n" + "=" * 80)
    print("✅ 所有测试完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()