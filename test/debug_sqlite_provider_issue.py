#!/usr/bin/env python3
"""
调试 SQLiteDataProvider db_path 参数传递问题
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

def test_sqlite_provider_config():
    """测试SQLite数据提供者配置"""
    logger.info("🔍 开始调试SQLiteDataProvider配置问题...")
    
    try:
        # 1. 测试直接创建SQLiteDataProvider
        logger.info("\n=== 测试1: 直接创建SQLiteDataProvider ===")
        from data.providers import SQLiteDataProvider
        
        # 模拟配置文件结构（从sqlite.yaml读取）
        sqlite_config = {
            'enabled': True,
            'priority': 1,
            'config': {
                'connection': {
                    'db_path': "./database/JY_database/sqlite/JY_database.sqlite"
                },
                'tables': {
                    'daily': {
                        'table_name': "JY_t_price_daily",
                        'column_mapping': {
                            'trade_date': 'date',
                            'stock_code': 'symbol',
                            'open_price': 'open',
                            'high_price': 'high',
                            'low_price': 'low',
                            'close_price': 'close',
                            'volume': 'volume',
                            'amount': 'amount',
                            'adj_factor': 'adj_factor'
                        }
                    }
                }
            }
        }
        
        # 应用与run_factor_analysis.py相同的配置处理逻辑
        provider_kwargs = sqlite_config['config'].copy()
        
        logger.info(f"原始provider_kwargs: {provider_kwargs}")
        
        # SQLite配置处理逻辑
        if 'connection' in provider_kwargs and 'tables' in provider_kwargs:
            conn_config = provider_kwargs['connection']
            tables_config = provider_kwargs['tables']
            daily_config = tables_config.get('daily', {})
            
            processed_kwargs = {
                'db_path': conn_config.get('db_path'),
                'table_name': daily_config.get('table_name', 'JY_t_price_daily'),
                'column_mapping': daily_config.get('column_mapping', {})
            }
            
            logger.info(f"处理后的kwargs: {processed_kwargs}")
            
            # 检查关键参数
            if not processed_kwargs.get('db_path'):
                logger.error("❌ 关键错误: db_path为空!")
                return False
            else:
                logger.info(f"✅ db_path存在: {processed_kwargs['db_path']}")
                
            # 尝试创建SQLiteDataProvider
            logger.info("尝试创建SQLiteDataProvider实例...")
            provider = SQLiteDataProvider(**processed_kwargs)
            logger.info("✅ SQLiteDataProvider创建成功!")
            
            # 测试get_all_symbols
            logger.info("测试get_all_symbols方法...")
            symbols = provider.get_all_symbols("2024-01-01")
            logger.info(f"✅ get_all_symbols成功，获取到 {len(symbols)} 只股票")
            if len(symbols) > 0:
                logger.info(f"   示例股票代码: {symbols[:5]}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        logger.error("详细堆栈信息:")
        traceback.print_exc()
        return False

def test_data_manager_initialization():
    """测试DataProviderManager初始化"""
    logger.info("\n=== 测试2: DataProviderManager初始化 ===")
    
    try:
        from data.manager import DataProviderManager
        from data.providers import SQLiteDataProvider
        
        # 使用与run_factor_analysis.py相同的配置
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
        
        # 测试_get_provider方法
        data_manager = DataProviderManager(
            provider_configs=DATA_PROVIDERS_CONFIG,
            symbols=['000001', '000002'],
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        logger.info("✅ DataProviderManager创建成功!")
        
        # 测试_get_provider方法
        logger.info("测试_get_provider方法...")
        provider = data_manager._get_provider('sqlite')
        
        if provider:
            logger.info("✅ _get_provider('sqlite')成功!")
            logger.info(f"   提供者类型: {type(provider)}")
            logger.info(f"   数据库路径: {getattr(provider, 'source_db_path', '未设置')}")
            logger.info(f"   表名: {getattr(provider, 'table_name', '未设置')}")
            
            # 测试get_all_symbols
            symbols = provider.get_all_symbols("2024-01-01")
            logger.info(f"✅ get_all_symbols成功，获取到 {len(symbols)} 只股票")
            return True
        else:
            logger.error("❌ _get_provider('sqlite')返回None")
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        logger.error("详细堆栈信息:")
        traceback.print_exc()
        return False

def test_config_loading():
    """测试配置文件加载"""
    logger.info("\n=== 测试3: 配置文件加载 ===")
    
    try:
        # 直接读取配置文件
        config_path = "configs/data/providers/sqlite.yaml"
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"✅ 配置文件读取成功: {config_path}")
            logger.info(f"   connection.db_path: {config.get('connection', {}).get('db_path')}")
            logger.info(f"   tables.daily.table_name: {config.get('tables', {}).get('daily', {}).get('table_name')}")
            
            return True
        else:
            logger.error(f"❌ 配置文件不存在: {config_path}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 配置文件加载失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始调试SQLiteDataProvider db_path参数传递问题")
    logger.info("=" * 60)
    
    tests = [
        ("配置文件加载", test_config_loading),
        ("SQLite提供者配置", test_sqlite_provider_config),
        ("DataProviderManager初始化", test_data_manager_initialization),
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
    logger.info("\n" + "=" * 60)
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
        logger.info("🎉 所有测试通过！配置问题已解决。")
    else:
        logger.warning(f"⚠️ {total-passed} 个测试失败，发现配置问题。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)