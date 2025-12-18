#!/usr/bin/env python3
"""
测试修复后的所有脚本兼容性
"""

import os
import sys
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_script_config_loading():
    """测试各脚本的配置加载功能"""
    logger.info("🔍 测试修复后脚本的配置加载...")
    
    scripts_to_test = [
        ('run_factor_analysis.py', '因子分析脚本'),
        ('download_data.py', '数据下载脚本'),
        ('run_backtest.py', '回测脚本')
    ]
    
    results = []
    
    for script_name, display_name in scripts_to_test:
        logger.info(f"\n📋 测试 {display_name} ({script_name})...")
        
        try:
            # 检查脚本文件是否存在
            script_path = f"scripts/{script_name}"
            if not os.path.exists(script_path):
                logger.error(f"❌ 脚本文件不存在: {script_path}")
                results.append((display_name, False))
                continue
            
            # 检查配置相关代码
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否包含修复后的配置处理逻辑
            checks = [
                ('load_providers', '加载提供者配置'),
                ('column_mapping', '列名映射配置'),
                ('sqlite.*config', 'SQLite配置处理'),
                ('connection.*db_path', '数据库路径配置')
            ]
            
            all_passed = True
            for pattern, desc in checks:
                import re
                if re.search(pattern, content):
                    logger.info(f"  ✅ {desc}: 存在")
                else:
                    logger.warning(f"  ⚠️ {desc}: 未找到")
                    all_passed = False
            
            # 检查是否移除了硬编码
            hardcoded_patterns = [
                "'db_path': './database/JY_database/sqlite/JY_database.sqlite'",
                "'table_name': 'JY_t_price_daily'"
            ]
            
            has_hardcoded = False
            for pattern in hardcoded_patterns:
                if pattern in content:
                    # 检查是否在默认配置中（在try-except中作为后备）
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if pattern in line:
                            # 检查上下文
                            context = '\n'.join(lines[max(0, i-3):i+4])
                            if 'try:' in context or 'except' in context or 'default' in context.lower():
                                logger.info(f"  ✅ 硬编码配置在后备逻辑中: 合理")
                            else:
                                logger.warning(f"  ⚠️ 发现硬编码配置: {pattern}")
                                has_hardcoded = True
            
            if not has_hardcoded:
                logger.info(f"  ✅ 无硬编码配置: 通过")
            
            result = all_passed and not has_hardcoded
            results.append((display_name, result))
            
            if result:
                logger.info(f"  ✅ {display_name} 配置兼容性: 通过")
            else:
                logger.warning(f"  ⚠️ {display_name} 配置兼容性: 有问题")
                
        except Exception as e:
            logger.error(f"  ❌ 测试 {display_name} 失败: {e}")
            results.append((display_name, False))
    
    return results

def test_sqlite_yaml_syntax():
    """测试sqlite.yaml配置文件语法"""
    logger.info("\n🔍 测试sqlite.yaml配置文件...")
    
    try:
        import yaml
        
        config_path = "configs/data/providers/sqlite.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查关键配置项
        required_keys = [
            'enabled', 'priority', 'connection', 'tables'
        ]
        
        for key in required_keys:
            if key in config:
                logger.info(f"  ✅ {key}: 存在")
            else:
                logger.error(f"  ❌ {key}: 缺失")
                return False
        
        # 检查connection配置
        connection = config.get('connection', {})
        if 'db_path' in connection:
            logger.info(f"  ✅ connection.db_path: {connection['db_path']}")
        else:
            logger.error(f"  ❌ connection.db_path: 缺失")
            return False
        
        # 检查tables配置
        tables = config.get('tables', {})
        if 'daily' in tables:
            daily = tables['daily']
            if 'column_mapping' in daily:
                column_mapping = daily['column_mapping']
                logger.info(f"  ✅ column_mapping: {len(column_mapping)} 个映射")
                
                # 检查关键映射
                required_mappings = ['ticker', '_date', '_open', '_close', '_volume']
                for mapping in required_mappings:
                    if mapping in column_mapping:
                        logger.info(f"    ✅ {mapping} -> {column_mapping[mapping]}")
                    else:
                        logger.warning(f"    ⚠️ {mapping}: 缺失")
            else:
                logger.error(f"  ❌ daily.column_mapping: 缺失")
                return False
        else:
            logger.error(f"  ❌ tables.daily: 缺失")
            return False
        
        logger.info("✅ sqlite.yaml 配置文件语法检查: 通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ sqlite.yaml 配置文件检查失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始测试修复后的脚本兼容性")
    logger.info("=" * 60)
    
    # 测试配置文件
    yaml_ok = test_sqlite_yaml_syntax()
    
    # 测试脚本配置加载
    script_results = test_script_config_loading()
    
    # 汇总结果
    logger.info("\n" + "=" * 60)
    logger.info("📊 兼容性测试结果汇总:")
    
    total_passed = 0
    total_tests = 1 + len(script_results)  # yaml + scripts
    
    if yaml_ok:
        total_passed += 1
        logger.info("  SQLite配置 yaml: ✅ 通过")
    else:
        logger.info("  SQLite配置 yaml: ❌ 失败")
    
    for script_name, result in script_results:
        if result:
            total_passed += 1
            logger.info(f"  {script_name}: ✅ 通过")
        else:
            logger.info(f"  {script_name}: ❌ 失败")
    
    logger.info(f"\n🎯 兼容性测试通过率: {total_passed}/{total_tests} ({total_passed/total_tests*100:.1f}%)")
    
    if total_passed == total_tests:
        logger.info("🎉 所有兼容性测试通过！修复成功。")
        return True
    else:
        logger.warning(f"⚠️ {total_tests-total_passed} 个测试失败，需要进一步检查。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)