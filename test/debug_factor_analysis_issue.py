#!/usr/bin/env python3
"""
调试 run_factor_analysis.py 中的 SQLiteDataProvider 问题
重现"全股票模式"下的错误
"""

import os
import sys
import logging
import traceback

# 设置调试日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_full_stock_mode():
    """测试全股票模式下的SQLiteDataProvider问题"""
    logger.info("🔍 重现全股票模式问题...")
    
    try:
        # 模拟 run_factor_analysis.py 中的配置加载逻辑
        import yaml
        
        # 读取配置
        providers_config_path = "configs/data/providers/sqlite.yaml"
        if not os.path.exists(providers_config_path):
            logger.error(f"❌ 配置文件不存在: {providers_config_path}")
            return False
            
        with open(providers_config_path, 'r', encoding='utf-8') as f:
            provider_config = yaml.safe_load(f)
        
        logger.info(f"✅ 加载配置: {provider_config}")
        
        # 模拟 run_factor_analysis.py 中的配置处理逻辑
        from data.providers import SQLiteDataProvider
        
        # 这是 run_factor_analysis.py 中的配置处理逻辑（第126-138行）
        provider_kwargs = provider_config.copy()
        
        logger.info("处理前的provider_kwargs:")
        logger.info(f"  {provider_kwargs}")
        
        if 'connection' in provider_kwargs and 'tables' in provider_kwargs:
            conn_config = provider_kwargs['connection']
            tables_config = provider_kwargs['tables']
            daily_config = tables_config.get('daily', {})
            
            processed_kwargs = {
                'db_path': conn_config.get('db_path'),
                'table_name': daily_config.get('table_name', 'JY_t_price_daily'),
                'column_mapping': daily_config.get('column_mapping', {})
            }
            
            logger.info("处理后的processed_kwargs:")
            logger.info(f"  {processed_kwargs}")
            
            # 检查 db_path
            if not processed_kwargs.get('db_path'):
                logger.error("❌ 关键错误: db_path为空!")
                return False
            else:
                logger.info(f"✅ db_path存在: {processed_kwargs['db_path']}")
            
            # 测试配置映射
            logger.info("检查列名映射:")
            logger.info(f"  配置的映射: {processed_kwargs['column_mapping']}")
            
            # 尝试创建提供者
            logger.info("尝试创建SQLiteDataProvider实例...")
            provider_instance = SQLiteDataProvider(**processed_kwargs)
            logger.info("✅ SQLiteDataProvider创建成功!")
            
            # 模拟全股票模式下的 get_all_symbols 调用
            target_date = "2024-01-01"  # START_DATE
            
            logger.info(f"模拟全股票模式调用 get_all_symbols('{target_date}')...")
            
            try:
                all_symbols = provider_instance.get_all_symbols(target_date)
                logger.info(f"✅ get_all_symbols成功! 获取到 {len(all_symbols)} 只股票")
                if len(all_symbols) > 0:
                    logger.info(f"   示例股票代码: {all_symbols[:5]}")
                return True
            except Exception as e:
                logger.error(f"❌ get_all_symbols失败: {e}")
                logger.error("错误详情:")
                traceback.print_exc()
                return False
        else:
            logger.error("❌ 配置格式不正确，缺少connection或tables")
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        logger.error("详细堆栈信息:")
        traceback.print_exc()
        return False

def test_data_manager_duplicate_call():
    """测试DataProviderManager中可能出现的重复调用"""
    logger.info("\n🔍 测试DataProviderManager中的重复调用...")
    
    try:
        from data.manager import DataProviderManager
        from data.providers import SQLiteDataProvider
        
        # 模拟 run_factor_analysis.py 中的DATA_PROVIDERS_CONFIG（第214-222行）
        DATA_PROVIDERS_CONFIG = [
            (
                SQLiteDataProvider,
                {
                    'db_path': './database/JY_database/sqlite/JY_database.sqlite',
                    'table_name': 'JY_t_price_daily'
                }
            ),
        ]
        
        logger.info("DataProviderManager配置:")
        for i, (cls, kwargs) in enumerate(DATA_PROVIDERS_CONFIG):
            logger.info(f"  提供者 {i+1}: {cls.__name__}")
            logger.info(f"    参数: {kwargs}")
        
        # 创建DataProviderManager
        data_manager = DataProviderManager(
            provider_configs=DATA_PROVIDERS_CONFIG,
            symbols=['000001', '000002'],
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        logger.info("✅ DataProviderManager创建成功!")
        
        # 模拟DataProviderManager._get_provider的调用
        logger.info("模拟第一次_get_provider调用...")
        provider1 = data_manager._get_provider('sqlite')
        
        if provider1:
            logger.info("✅ 第一次调用成功")
            logger.info(f"   数据库路径: {getattr(provider1, 'source_db_path', '未设置')}")
            
            # 测试get_all_symbols
            symbols1 = provider1.get_all_symbols("2024-01-01")
            logger.info(f"   第一次get_all_symbols: {len(symbols1)} 只股票")
        
        # 模拟可能的第二次调用（重复错误的原因）
        logger.info("模拟第二次_get_provider调用...")
        provider2 = data_manager._get_provider('sqlite')
        
        if provider2:
            logger.info("✅ 第二次调用成功")
            logger.info(f"   数据库路径: {getattr(provider2, 'source_db_path', '未设置')}")
            
            # 检查是否是同一个实例
            if provider1 is provider2:
                logger.info("✅ 确认是同一个实例（符合预期）")
            else:
                logger.warning("⚠️ 不同的实例（可能是问题）")
            
            # 再次测试get_all_symbols
            symbols2 = provider2.get_all_symbols("2024-01-01")
            logger.info(f"   第二次get_all_symbols: {len(symbols2)} 只股票")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        logger.error("详细堆栈信息:")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始调试 run_factor_analysis.py 中的SQLiteDataProvider问题")
    logger.info("=" * 70)
    
    tests = [
        ("全股票模式问题重现", test_full_stock_mode),
        ("DataProviderManager重复调用", test_data_manager_duplicate_call),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 执行测试: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ 测试 {test_name} 发生异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    logger.info("\n" + "=" * 70)
    logger.info("📊 调试结果汇总:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 测试通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 所有测试通过！问题可能已经解决。")
    else:
        logger.warning(f"⚠️ {total-passed} 个测试失败，发现了具体问题。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)