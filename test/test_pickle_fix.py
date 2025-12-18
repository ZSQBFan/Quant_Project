#!/usr/bin/env python3
"""
测试Pickle序列化修复是否有效
"""

import pickle
import sys
import os

# 添加项目路径
sys.path.insert(0, '/Volumes/叽叽叽/Code/quant_project_3.12_macmini')

from data.handlers.database import DatabaseHandler
from data.manager import DataProviderManager

def test_database_handler_pickle():
    """测试DatabaseHandler的pickle序列化"""
    print("测试 DatabaseHandler pickle 序列化...")
    
    try:
        # 创建DatabaseHandler实例
        db_handler = DatabaseHandler(':memory:')  # 使用内存数据库进行测试
        
        # 尝试pickle序列化
        pickled = pickle.dumps(db_handler)
        print(f"✅ DatabaseHandler pickle 成功，序列化大小: {len(pickled)} 字节")
        
        # 尝试反序列化
        unpickled = pickle.loads(pickled)
        print("✅ DatabaseHandler unpickle 成功")
        
        # 验证基本功能
        assert unpickled.db_path == db_handler.db_path
        print("✅ 基本属性验证通过")
        
        return True
    except Exception as e:
        print(f"❌ DatabaseHandler pickle 测试失败: {e}")
        return False

def test_data_provider_manager_pickle():
    """测试DataProviderManager的pickle序列化"""
    print("\n测试 DataProviderManager pickle 序列化...")
    
    try:
        # 创建DataProviderManager实例
        data_manager = DataProviderManager(
            provider_configs=[],
            symbols=['000001', '000002'],
            start_date='2024-01-01',
            end_date='2024-12-31'
        )
        
        # 尝试pickle序列化
        pickled = pickle.dumps(data_manager)
        print(f"✅ DataProviderManager pickle 成功，序列化大小: {len(pickled)} 字节")
        
        # 尝试反序列化
        unpickled = pickle.loads(pickled)
        print("✅ DataProviderManager unpickle 成功")
        
        # 验证基本属性
        assert unpickled.symbols == data_manager.symbols
        assert unpickled.start_date == data_manager.start_date
        print("✅ 基本属性验证通过")
        
        return True
    except Exception as e:
        print(f"❌ DataProviderManager pickle 测试失败: {e}")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("开始测试 Pickle 序列化修复")
    print("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    # 测试DatabaseHandler
    if test_database_handler_pickle():
        success_count += 1
    
    # 测试DataProviderManager  
    if test_data_provider_manager_pickle():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {success_count}/{total_tests} 通过")
    if success_count == total_tests:
        print("🎉 所有pickle序列化测试通过！")
        sys.exit(0)
    else:
        print("⚠️  部分测试失败，需要进一步调试")
        sys.exit(1)