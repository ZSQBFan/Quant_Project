#!/usr/bin/env python3
"""
数据完整性检查脚本
检查2021-02-01之前的数据分布，特别是因子计算需要的数据
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# 数据库路径
DB_PATH = "./database/CSMR/CSMR_stock_daily_price.sqlite"
TABLE_NAME = "daily_price"

def check_database_exists():
    """检查数据库文件是否存在"""
    if not os.path.exists(DB_PATH):
        print(f"❌ 数据库文件不存在: {DB_PATH}")
        return False
    print(f"✅ 数据库文件存在: {DB_PATH}")
    return True

def connect_database():
    """连接数据库"""
    try:
        conn = sqlite3.connect(DB_PATH)
        print(f"✅ 成功连接到数据库")
        return conn
    except Exception as e:
        print(f"❌ 连接数据库失败: {e}")
        return None

def get_table_structure(conn):
    """获取表结构"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
    columns = cursor.fetchall()
    print(f"\n📊 表 {TABLE_NAME} 的结构:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    return columns

def check_date_range_data(conn, start_date, end_date):
    """检查指定日期范围内的数据"""
    cursor = conn.cursor()
    
    # 检查交易日期范围
    query_dates = """
    SELECT 
        MIN(TradingDate) as min_date,
        MAX(TradingDate) as max_date,
        COUNT(DISTINCT TradingDate) as total_trading_days,
        COUNT(*) as total_records
    FROM daily_price 
    WHERE TradingDate >= ? AND TradingDate <= ?
    """
    
    cursor.execute(query_dates, (start_date, end_date))
    result = cursor.fetchone()
    
    print(f"\n📅 {start_date} 到 {end_date} 期间的数据统计:")
    print(f"  - 最早交易日期: {result[0]}")
    print(f"  - 最晚交易日期: {result[1]}")
    print(f"  - 总交易天数: {result[2]}")
    print(f"  - 总记录数: {result[3]}")
    
    return result

def check_monthly_data_distribution(conn, start_date, end_date):
    """按月检查数据分布"""
    cursor = conn.cursor()
    
    query_monthly = """
    SELECT 
        substr(TradingDate, 1, 7) as month,
        COUNT(DISTINCT TradingDate) as trading_days,
        COUNT(*) as total_records,
        COUNT(DISTINCT Symbol) as unique_symbols
    FROM daily_price 
    WHERE TradingDate >= ? AND TradingDate <= ?
    GROUP BY substr(TradingDate, 1, 7)
    ORDER BY month
    """
    
    cursor.execute(query_monthly, (start_date, end_date))
    results = cursor.fetchall()
    
    print(f"\n📊 按月数据分布:")
    print(f"{'月份':<10} {'交易天数':<8} {'总记录数':<8} {'股票数量':<8}")
    print("-" * 35)
    for row in results:
        print(f"{row[0]:<10} {row[1]:<8} {row[2]:<8} {row[3]:<8}")
    
    return results

def check_sample_stocks_data(conn, start_date, end_date):
    """检查样本股票的数据情况"""
    cursor = conn.cursor()
    
    # 获取一些常见股票代码
    query_symbols = """
    SELECT Symbol, COUNT(*) as record_count
    FROM daily_price 
    WHERE TradingDate >= ? AND TradingDate <= ?
    GROUP BY Symbol
    ORDER BY record_count DESC
    LIMIT 10
    """
    
    cursor.execute(query_symbols, (start_date, end_date))
    symbols = cursor.fetchall()
    
    print(f"\n🔍 样本股票数据检查 (前10个):")
    print(f"{'股票代码':<10} {'记录数':<8}")
    print("-" * 20)
    for symbol in symbols:
        print(f"{symbol[0]:<10} {symbol[1]:<8}")
    
    # 检查几个具体股票在2020年12月的数据
    if symbols:
        sample_symbols = [s[0] for s in symbols[:3]]
        print(f"\n📋 样本股票 {sample_symbols} 在2020年12月的数据详情:")
        
        for symbol in sample_symbols:
            query_detail = """
            SELECT 
                TradingDate,
                ClosePrice,
                Volume,
                CASE WHEN ClosePrice IS NULL THEN 1 ELSE 0 END as missing_price,
                CASE WHEN Volume IS NULL THEN 1 ELSE 0 END as missing_volume
            FROM daily_price 
            WHERE Symbol = ? AND TradingDate >= '2020-12-01' AND TradingDate <= '2020-12-31'
            ORDER BY TradingDate
            """
            
            cursor.execute(query_detail, (symbol,))
            detail_results = cursor.fetchall()
            
            missing_price = sum([r[3] for r in detail_results])
            missing_volume = sum([r[4] for r in detail_results])
            total_days = len(detail_results)
            
            print(f"  {symbol}:")
            print(f"    - 2020年12月交易天数: {total_days}")
            print(f"    - 缺失收盘价: {missing_price} 天")
            print(f"    - 缺失成交量: {missing_volume} 天")
            
            if detail_results:
                print(f"    - 最早数据: {detail_results[0][0]}")
                print(f"    - 最晚数据: {detail_results[-1][0]}")

def check_factor_calculation_readiness(conn, start_date, end_date):
    """检查因子计算所需的数据准备情况"""
    print(f"\n🧮 因子计算数据准备检查:")
    
    cursor = conn.cursor()
    
    # 检查价格数据的完整性
    query_completeness = """
    SELECT 
        TradingDate,
        COUNT(*) as total_stocks,
        COUNT(ClosePrice) as stocks_with_price,
        COUNT(Volume) as stocks_with_volume,
        ROUND(COUNT(ClosePrice) * 100.0 / COUNT(*), 2) as price_completeness,
        ROUND(COUNT(Volume) * 100.0 / COUNT(*), 2) as volume_completeness
    FROM daily_price 
    WHERE TradingDate >= ? AND TradingDate <= ?
    GROUP BY TradingDate
    ORDER BY TradingDate
    LIMIT 20
    """
    
    cursor.execute(query_completeness, (start_date, end_date))
    completeness_results = cursor.fetchall()
    
    print(f"\n📊 前20个交易日的数据完整性:")
    print(f"{'交易日期':<12} {'总股票':<6} {'有价格':<6} {'有成交量':<8} {'价格完整率':<8} {'成交量完整率':<10}")
    print("-" * 65)
    
    for row in completeness_results:
        print(f"{row[0]:<12} {row[1]:<6} {row[2]:<6} {row[3]:<8} {row[4]:<8}% {row[5]:<10}%")

def main():
    """主函数"""
    print("🔍 开始数据完整性检查...")
    
    # 检查数据库文件
    if not check_database_exists():
        return
    
    # 连接数据库
    conn = connect_database()
    if not conn:
        return
    
    try:
        # 获取表结构
        get_table_structure(conn)
        
        # 定义检查日期范围
        start_date = "2020-12-01"
        end_date = "2021-02-01"
        
        # 检查整体数据范围
        check_date_range_data(conn, start_date, end_date)
        
        # 按月检查数据分布
        check_monthly_data_distribution(conn, start_date, end_date)
        
        # 检查样本股票
        check_sample_stocks_data(conn, start_date, end_date)
        
        # 检查因子计算准备情况
        check_factor_calculation_readiness(conn, start_date, end_date)
        
        print(f"\n✅ 数据完整性检查完成")
        
    except Exception as e:
        print(f"❌ 检查过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    main()