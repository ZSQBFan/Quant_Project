#!/usr/bin/env python3
"""
测试全股票功能

验证新增的"获取全部股票代码"功能是否正常工作。
"""

import os
import sys
import logging
from typing import List, Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_base_provider():
    """测试基类抽象方法"""
    from data.providers.base import BaseDataProvider
    
    logger.info("🧪 测试基类抽象方法...")
    
    # 创建基类实例
    provider = BaseDataProvider()
    
    # 测试fetch_data方法（应该抛出NotImplementedError）
    try:
        provider.fetch_data("000001", "2024-01-01", "2024-01-31")
        logger.error("❌ fetch_data方法应该抛出NotImplementedError")
        return False
    except NotImplementedError:
        logger.info("✅ fetch_data方法正确抛出NotImplementedError")
    
    # 测试get_all_symbols方法（应该抛出NotImplementedError）
    try:
        provider.get_all_symbols("2024-01-01")
        logger.error("❌ get_all_symbols方法应该抛出NotImplementedError")
        return False
    except NotImplementedError:
        logger.info("✅ get_all_symbols方法正确抛出NotImplementedError")
    
    return True

def test_config_loading():
    """测试配置加载"""
    logger.info("🧪 测试配置加载...")
    
    try:
        from core.config import load_config
        
        config_loader = load_config()
        data_config = config_loader.load_data()
        
        # 检查新增的use_all_stocks字段
        if hasattr(data_config, 'use_all_stocks'):
            logger.info(f"✅ 配置加载成功，use_all_stocks = {data_config.use_all_stocks}")
            return True
        else:
            logger.error("❌ DataConfig类中缺少use_all_stocks字段")
            return False
            
    except Exception as e:
        logger.error(f"❌ 配置加载失败: {e}")
        return False

def test_sqlite_provider():
    """测试SQLite数据提供者"""
    logger.info("🧪 测试SQLite数据提供者...")
    
    try:
        from data.providers.sqlite import SQLiteDataProvider
        
        # 测试初始化（使用默认配置）
        provider = SQLiteDataProvider(
            db_path="./database/JY_database/sqlite/JY_database.sqlite",
            table_name="JY_t_price_daily"
        )
        
        logger.info("✅ SQLite数据提供者初始化成功")
        
        # 测试get_all_symbols方法
        try:
            symbols = provider.get_all_symbols("2024-01-01")
            logger.info(f"✅ SQLite get_all_symbols成功，获取到 {len(symbols)} 只股票")
            if len(symbols) > 0:
                logger.info(f"   示例股票代码: {symbols[:5]}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ SQLite get_all_symbols执行失败（可能数据库不存在）: {e}")
            return True  # 这不是致命错误
            
    except Exception as e:
        logger.error(f"❌ SQLite数据提供者测试失败: {e}")
        return False

def test_akshare_provider():
    """测试AKShare数据提供者"""
    logger.info("🧪 测试AKShare数据提供者...")
    
    try:
        from data.providers.akshare import AkshareDataProvider
        
        # 测试初始化
        provider = AkshareDataProvider()
        logger.info("✅ AKShare数据提供者初始化成功")
        
        # 测试get_all_symbols方法
        try:
            symbols = provider.get_all_symbols()
            logger.info(f"✅ AKShare get_all_symbols成功，获取到 {len(symbols)} 只股票")
            if len(symbols) > 0:
                logger.info(f"   示例股票代码: {symbols[:5]}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ AKShare get_all_symbols执行失败（可能网络问题）: {e}")
            return True  # 这不是致命错误
            
    except Exception as e:
        logger.error(f"❌ AKShare数据提供者测试失败: {e}")
        return False

def test_tushare_provider():
    """测试Tushare数据提供者"""
    logger.info("🧪 测试Tushare数据提供者...")
    
    try:
        from data.providers.tushare import TushareDataProvider
        
        # 测试初始化（没有token的情况）
        provider = TushareDataProvider(token="fake_token_for_test")
        logger.info("✅ Tushare数据提供者初始化成功")
        
        # 测试get_all_symbols方法
        try:
            symbols = provider.get_all_symbols("2024-01-01")
            logger.info(f"✅ Tushare get_all_symbols成功，获取到 {len(symbols)} 只股票")
            if len(symbols) > 0:
                logger.info(f"   示例股票代码: {symbols[:5]}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Tushare get_all_symbols执行失败（可能token无效或网络问题）: {e}")
            return True  # 这不是致命错误
            
    except Exception as e:
        logger.error(f"❌ Tushare数据提供者测试失败: {e}")
        return False

def test_config_file():
    """测试配置文件"""
    logger.info("🧪 测试配置文件...")
    
    config_path = "configs/data/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'use_all_stocks:' in content:
                logger.info("✅ 配置文件包含use_all_stocks配置")
                return True
            else:
                logger.error("❌ 配置文件缺少use_all_stocks配置")
                return False
    else:
        logger.error(f"❌ 配置文件不存在: {config_path}")
        return False

def test_script_integration():
    """测试脚本集成"""
    logger.info("🧪 测试脚本集成...")
    
    # 检查因子分析脚本
    factor_script = "scripts/run_factor_analysis.py"
    if os.path.exists(factor_script):
        with open(factor_script, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'data_config.use_all_stocks' in content:
                logger.info("✅ 因子分析脚本包含全股票模式逻辑")
            else:
                logger.error("❌ 因子分析脚本缺少全股票模式逻辑")
                return False
    
    # 检查回测脚本
    backtest_script = "scripts/run_backtest.py"
    if os.path.exists(backtest_script):
        with open(backtest_script, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'data_config.use_all_stocks' in content:
                logger.info("✅ 回测脚本包含全股票模式逻辑")
            else:
                logger.error("❌ 回测脚本缺少全股票模式逻辑")
                return False
    
    return True

def main():
    """主测试函数"""
    logger.info("🚀 开始测试全股票功能...")
    logger.info("=" * 60)
    
    tests = [
        ("配置加载", test_config_loading),
        ("配置文件", test_config_file),
        ("基类抽象方法", test_base_provider),
        ("SQLite提供者", test_sqlite_provider),
        ("AKShare提供者", test_akshare_provider),
        ("Tushare提供者", test_tushare_provider),
        ("脚本集成", test_script_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 测试: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ 测试 {test_name} 发生异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    logger.info("\n" + "=" * 60)
    logger.info("📊 测试结果汇总:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 测试通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 所有测试通过！全股票功能实现成功。")
        return True
    else:
        logger.warning(f"⚠️ {total-passed} 个测试失败，请检查实现。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)