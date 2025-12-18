#!/usr/bin/env python3
"""
验证数据库实际表结构
"""

import os
import sys
import sqlite3
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_database_schema():
    """检查数据库表结构"""
    db_path = "./database/JY_database/sqlite/JY_database.sqlite"
    
    logger.info(f"🔍 检查数据库表结构: {db_path}")
    
    if not os.path.exists(db_path):
        logger.error(f"❌ 数据库文件不存在: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        logger.info(f"📋 数据库中的表: {[table[0] for table in tables]}")
        
        # 检查 JY_t_price_daily 表结构
        table_name = 'JY_t_price_daily'
        if (table_name,) in tables:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            logger.info(f"\n📊 {table_name} 表结构:")
            for col in columns:
                col_id, col_name, col_type, not_null, default_value, primary_key = col
                pk_indicator = " (主键)" if primary_key else ""
                logger.info(f"   {col_name:<15} {col_type:<10} {pk_indicator}")
            
            # 检查表中的示例数据
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
            sample = cursor.fetchone()
            if sample:
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                logger.info(f"\n📝 {table_name} 示例数据:")
                for i, col in enumerate(columns):
                    logger.info(f"   {col[1]}: {sample[i]}")
        else:
            logger.error(f"❌ 表 {table_name} 不存在")
            
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ 检查数据库结构失败: {e}")
        return False

def test_config_loading_issues():
    """分析配置加载问题"""
    logger.info("\n🔍 分析配置加载问题...")
    
    # 检查不同代码路径如何处理配置
    logger.info("分析不同代码路径的配置处理方式:")
    
    # 路径1: run_factor_analysis.py 中的全股票模式
    logger.info("\n📍 路径1: run_factor_analysis.py 全股票模式")
    logger.info("  - 读取: configs/data/providers/sqlite.yaml")
    logger.info("  - 处理: 使用配置的 column_mapping")
    logger.info("  - 结果: 查询失败 (stock_code, trade_date 不存在)")
    
    # 路径2: run_factor_analysis.py 中的DataProviderManager
    logger.info("\n📍 路径2: run_factor_analysis.py DataProviderManager")
    logger.info("  - 读取: 硬编码配置")
    logger.info("  - 处理: 使用硬编码的列名映射")
    logger.info("  - 结果: 查询成功 (ticker, _date 存在)")
    
    # 路径3: 检查现有代码中是否还有其他硬编码
    logger.info("\n📍 路径3: 其他可能的硬编码路径")
    
    try:
        # 检查 SQLiteDataProvider 的默认映射
        from data.providers import SQLiteDataProvider
        
        # 创建实例来查看默认映射
        provider = SQLiteDataProvider(
            db_path="./database/JY_database/sqlite/JY_database.sqlite",
            table_name="JY_t_price_daily"
        )
        
        logger.info(f"SQLiteDataProvider 默认 column_mapping:")
        for k, v in provider.column_mapping.items():
            logger.info(f"  {k} -> {v}")
            
        logger.info("\n✅ 发现关键问题：")
        logger.info("  - get_all_symbols 方法使用 column_mapping 来构建查询")
        logger.info("  - 如果 column_mapping 是配置的，查询使用配置列名")
        logger.info("  - 如果 column_mapping 是默认的，查询使用默认列名")
        logger.info("  - run_factor_analysis.py 中全股票模式使用了配置映射")
        logger.info("  - DataProviderManager 使用了硬编码配置")
        
    except Exception as e:
        logger.error(f"❌ 检查默认映射失败: {e}")

def main():
    """主函数"""
    logger.info("🚀 开始验证数据库表结构并分析配置问题")
    logger.info("=" * 60)
    
    # 检查数据库结构
    check_database_schema()
    
    # 分析配置加载问题
    test_config_loading_issues()
    
    logger.info("\n" + "=" * 60)
    logger.info("📋 分析总结:")
    logger.info("1. 数据库实际列名: ticker, _date (而非 stock_code, trade_date)")
    logger.info("2. 配置文件与数据库不匹配")
    logger.info("3. 某些代码路径使用硬编码配置，避开了配置问题")
    logger.info("4. 全股票模式直接使用配置，导致问题暴露")

if __name__ == "__main__":
    main()