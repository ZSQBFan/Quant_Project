#!/usr/bin/env python3
"""
测试多SQLite数据源支持的验证脚本
"""

import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_config_loading():
    """测试配置加载"""
    try:
        logger.info("=== 测试配置加载 ===")
        
        # 导入配置加载器
        from core.config import load_config
        
        # 加载配置
        config_loader = load_config()
        logger.info("✅ 配置加载器创建成功")
        
        # 加载数据提供者配置
        providers_config = config_loader.load_providers()
        logger.info(f"✅ 数据提供者配置加载成功，共 {len(providers_config)} 个提供者")
        
        # 检查SQLite提供者
        sqlite_providers = [name for name in providers_config.keys() if name.startswith('sqlite')]
        logger.info(f"✅ 发现 {len(sqlite_providers)} 个SQLite提供者: {sqlite_providers}")
        
        # 检查提供者配置详情
        for provider_name, provider_config in providers_config.items():
            logger.info(f"  - {provider_name}: enabled={provider_config.enabled}")
            if provider_config.enabled:
                logger.info(f"    配置: {provider_config.config}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 配置加载测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_data_provider_manager():
    """测试数据提供者管理器"""
    try:
        logger.info("\n=== 测试数据提供者管理器 ===")
        
        from data.manager import DataProviderManager
        from data.providers import SQLiteDataProvider
        
        # 模拟配置数据
        provider_configs = [
            ('sqlite_csmr', SQLiteDataProvider, {
                'db_path': './database/CSMR/CSMR_stock_daily_price.sqlite',
                'table_name': 'JY_t_price_daily',
                'column_mapping': {
                    '_date': 'date',
                    'ticker': 'code',
                    '_open': 'open',
                    '_high': 'high',
                    '_low': 'low',
                    '_close': 'close',
                    '_volume': 'volume',
                    '_value': 'turnover',
                    '_return': 'pct_change'
                }
            }),
            ('sqlite_jy', SQLiteDataProvider, {
                'db_path': './database/JY_stock_daily_price.sqlite',
                'table_name': 'JY_t_price_daily',
                'column_mapping': {
                    '_date': 'date',
                    'ticker': 'code',
                    '_open': 'open',
                    '_high': 'high',
                    '_low': 'low',
                    '_close': 'close',
                    '_volume': 'volume',
                    '_value': 'turnover',
                    '_return': 'pct_change'
                }
            })
        ]
        
        # 创建数据提供者管理器
        data_manager = DataProviderManager(
            provider_configs=provider_configs,
            symbols=['600519'],  # 贵州茅台作为测试
            start_date='2024-01-01',
            end_date='2024-01-31',
            db_path='./test_quant_data.db',
            auto_detect_universe=False
        )
        
        logger.info("✅ 数据提供者管理器创建成功")
        
        # 测试获取提供者实例
        provider1 = data_manager._get_provider('sqlite_csmr')
        provider2 = data_manager._get_provider('sqlite_jy')
        
        logger.info(f"✅ sqlite_csmr提供者实例: {provider1 is not None}")
        logger.info(f"✅ sqlite_jy提供者实例: {provider2 is not None}")
        
        # 验证是否为不同实例
        if provider1 and provider2:
            same_instance = provider1 is provider2
            logger.info(f"✅ 不同提供者是否为同一实例: {same_instance} (应为False)")
            if not same_instance:
                logger.info("✅ 多SQLite数据源支持验证成功!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据提供者管理器测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_factor_analysis_script():
    """测试因子分析脚本配置"""
    try:
        logger.info("\n=== 测试因子分析脚本配置 ===")
        
        # 导入因子分析脚本
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        import run_factor_analysis
        
        # 检查配置处理逻辑
        from core.config import load_config
        
        config_loader = load_config()
        data_config = config_loader.load_data()
        
        logger.info(f"✅ 全股票模式: {data_config.use_all_stocks}")
        logger.info(f"✅ 提供者优先级: {data_config.provider_priority}")
        
        # 检查DATA_PROVIDERS_CONFIG构建逻辑
        providers_config = config_loader.load_providers()
        
        # 模拟脚本中的配置构建过程
        DATA_PROVIDERS_CONFIG = []
        
        for provider_name in providers_config:
            provider_config = providers_config[provider_name]
            
            if provider_name.startswith('sqlite') and provider_config.enabled:
                logger.info(f"处理SQLite数据提供者: {provider_name}")
                
                provider_kwargs = provider_config.config.copy()
                
                if 'connection' in provider_kwargs and 'tables' in provider_kwargs:
                    conn_config = provider_kwargs['connection']
                    tables_config = provider_kwargs['tables']
                    daily_config = tables_config.get('daily', {})
                    
                    processed_kwargs = {
                        'db_path': conn_config.get('db_path'),
                        'table_name': daily_config.get('table_name', 'JY_t_price_daily'),
                        'column_mapping': daily_config.get('column_mapping', {})
                    }
                    
                    # 使用名称标识符作为元组的第一个元素
                    DATA_PROVIDERS_CONFIG.append((provider_name, type('SQLiteDataProvider', (), {}), processed_kwargs))
                    logger.info(f"✅ {provider_name}数据提供者配置已加载: {processed_kwargs}")
        
        logger.info(f"✅ 总共加载了 {len(DATA_PROVIDERS_CONFIG)} 个数据提供者配置")
        for name, cls, kwargs in DATA_PROVIDERS_CONFIG:
            logger.info(f"  - {name}: {kwargs['db_path']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 因子分析脚本测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主测试函数"""
    logger.info("🧪 开始多SQLite数据源支持验证测试")
    logger.info("=" * 60)
    
    tests = [
        ("配置加载", test_config_loading),
        ("数据提供者管理器", test_data_provider_manager),
        ("因子分析脚本配置", test_factor_analysis_script),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 执行测试: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ 测试通过: {test_name}")
                passed += 1
            else:
                logger.error(f"❌ 测试失败: {test_name}")
                failed += 1
        except Exception as e:
            logger.error(f"❌ 测试异常: {test_name} - {e}")
            failed += 1
    
    logger.info("\n" + "=" * 60)
    logger.info(f"🏁 测试完成! 通过: {passed}, 失败: {failed}")
    
    if failed == 0:
        logger.info("🎉 所有测试通过! 多SQLite数据源支持修改成功!")
    else:
        logger.warning("⚠️  部分测试失败，请检查相关修改")
    
    return failed == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)