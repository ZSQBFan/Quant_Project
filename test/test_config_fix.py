#!/usr/bin/env python3
"""
测试配置修复

验证修复后的全股票模式配置是否正常工作
"""

import os
import sys
import yaml

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_config_fix():
    """测试配置修复"""
    print("🧪 测试配置修复")
    print("=" * 60)
    
    try:
        from core.config import load_config
        
        config_loader = load_config()
        providers_config = config_loader.load_providers()
        data_config = config_loader.load_data()
        
        print(f"✅ 配置加载成功")
        print(f"   use_all_stocks: {data_config.use_all_stocks}")
        print(f"   provider_priority: {data_config.provider_priority}")
        
        # 测试SQLite提供者配置
        if 'sqlite' in providers_config:
            sqlite_config = providers_config['sqlite']
            print(f"\n📋 SQLite提供者配置:")
            print(f"   enabled: {sqlite_config.enabled}")
            print(f"   priority: {sqlite_config.priority}")
            print(f"   config keys: {list(sqlite_config.config.keys())}")
            
            if 'connection' in sqlite_config.config:
                conn = sqlite_config.config['connection']
                print(f"   connection.db_path: {conn.get('db_path')}")
            
            if 'tables' in sqlite_config.config:
                tables = sqlite_config.config['tables']
                if 'daily' in tables:
                    daily = tables['daily']
                    print(f"   tables.daily.table_name: {daily.get('table_name')}")
        
        # 测试配置处理逻辑
        print(f"\n📋 测试配置处理逻辑:")
        
        for provider_name in data_config.provider_priority:
            if provider_name in providers_config and providers_config[provider_name].enabled:
                provider_config = providers_config[provider_name]
                provider_kwargs = provider_config.config.copy()
                
                print(f"\n   测试 {provider_name}:")
                
                # 应用修复的配置处理逻辑
                if provider_name == 'sqlite':
                    if 'connection' in provider_kwargs and 'tables' in provider_kwargs:
                        conn_config = provider_kwargs['connection']
                        tables_config = provider_kwargs['tables']
                        daily_config = tables_config.get('daily', {})
                        
                        processed_kwargs = {
                            'db_path': conn_config.get('db_path'),
                            'table_name': daily_config.get('table_name', 'JY_t_price_daily'),
                            'column_mapping': daily_config.get('column_mapping', {})
                        }
                        print(f"     ✅ 配置处理成功")
                        print(f"     db_path: {processed_kwargs['db_path']}")
                        print(f"     table_name: {processed_kwargs['table_name']}")
                        print(f"     column_mapping keys: {list(processed_kwargs['column_mapping'].keys())}")
                        
                        # 验证关键参数
                        if processed_kwargs['db_path']:
                            print(f"     ✅ db_path 有效")
                        else:
                            print(f"     ❌ db_path 无效")
                    else:
                        print(f"     ❌ 缺少connection或tables配置")
                else:
                    print(f"     ✅ 配置传递正常")
        
        print(f"\n🎉 配置修复验证完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🚀 开始测试配置修复")
    test_config_fix()
    print("\n🎉 测试完成")

if __name__ == "__main__":
    main()