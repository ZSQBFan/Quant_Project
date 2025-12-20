#!/usr/bin/env python3
"""
滚动窗口数据充足性分析脚本
分析RollingICIRCalculator的60天滚动窗口是否需要更多历史数据
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 数据库路径
DB_PATH = "./database/CSMR/CSMR_stock_daily_price.sqlite"
TABLE_NAME = "daily_price"

def analyze_rolling_window_requirements():
    """分析滚动窗口的数据需求"""
    print("🔍 分析滚动窗口数据需求...")
    
    # RollingICIR配置参数
    rolling_window_days = 60
    analysis_start_date = "2021-01-01"
    
    # 计算需要的历史数据开始日期
    required_start_date = "2020-11-01"  # 提前约60天
    
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # 检查从2020-11-01开始的数据可用性
        query = """
        SELECT 
            MIN(TradingDate) as earliest_date,
            MAX(TradingDate) as latest_date,
            COUNT(DISTINCT TradingDate) as total_trading_days,
            COUNT(DISTINCT Symbol) as total_symbols
        FROM daily_price 
        WHERE TradingDate >= ?
        """
        
        cursor = conn.cursor()
        cursor.execute(query, (required_start_date,))
        result = cursor.fetchone()
        
        print(f"\n📊 滚动窗口数据分析 ({required_start_date} 之后):")
        print(f"  - 最早交易日期: {result[0]}")
        print(f"  - 最晚交易日期: {result[1]}")
        print(f"  - 总交易天数: {result[2]}")
        print(f"  - 总股票数: {result[3]}")
        
        # 检查2021年1月开始的数据是否足够支持60天窗口
        print(f"\n🎯 2021年1月开始分析的数据充足性:")
        
        # 统计每个月的交易天数
        query_monthly = """
        SELECT 
            substr(TradingDate, 1, 7) as month,
            COUNT(DISTINCT TradingDate) as trading_days,
            COUNT(*) as total_records
        FROM daily_price 
        WHERE TradingDate >= ?
        GROUP BY substr(TradingDate, 1, 7)
        ORDER BY month
        """
        
        cursor.execute(query_monthly, (required_start_date,))
        monthly_results = cursor.fetchall()
        
        print(f"{'月份':<10} {'交易天数':<8} {'总记录数':<8}")
        print("-" * 28)
        for row in monthly_results:
            print(f"{row[0]:<10} {row[1]:<8} {row[2]:<8}")
        
        # 计算到2021年1月1日时有多少历史数据
        query_history_at_start = """
        SELECT 
            COUNT(DISTINCT TradingDate) as trading_days_before_2021_01_01
        FROM daily_price 
        WHERE TradingDate < '2021-01-01'
        """
        
        cursor.execute(query_history_at_start)
        history_days = cursor.fetchone()[0]
        
        print(f"\n📈 2021年1月1日前的累积交易天数: {history_days} 天")
        
        if history_days >= 60:
            print(f"✅ 足够支持60天滚动窗口 (需要: 60天, 实际: {history_days}天)")
        else:
            print(f"❌ 不够支持60天滚动窗口 (需要: 60天, 实际: {history_days}天)")
            print(f"   建议调整分析开始日期或滚动窗口大小")
        
        # 检查特定因子的数据需求
        print(f"\n🧮 特定因子数据需求分析:")
        check_factor_data_requirements(cursor)
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

def check_factor_data_requirements(cursor):
    """检查特定因子的数据需求"""
    # 根据配置文件，因子列表包括：
    # Momentum, Reversal20D, IndNeu_SalesGrowth, IndNeu_Momentum, IndNeu_Reversal20D 等
    
    # Momentum和Reversal20D需要20天的历史数据
    # IndNeu_SalesGrowth等基本面因子可能需要更长的历史数据
    
    print(f"\n  📊 Momentum因子 (需要20天历史):")
    check_factor_availability(cursor, "2021-01-01", 20, "动量因子")
    
    print(f"\n  📊 Reversal20D因子 (需要20天历史):")
    check_factor_availability(cursor, "2021-01-01", 20, "反转因子")
    
    print(f"\n  📊 行业中性因子 (需要更长历史):")
    check_factor_availability(cursor, "2021-01-01", 60, "行业中性因子")
    
    # 检查2020年11月和12月的数据质量
    print(f"\n  📋 数据质量检查:")
    check_data_quality(cursor, "2020-11-01", "2020-12-31")
    print(f"\n  📋 数据质量检查:")
    check_data_quality(cursor, "2020-12-01", "2020-12-31")

def check_factor_availability(cursor, start_date, required_days, factor_name):
    """检查因子数据的可用性"""
    query = """
    SELECT 
        COUNT(DISTINCT TradingDate) as available_days
    FROM daily_price 
    WHERE TradingDate >= date(?, '-' || ? || ' days')
    """
    
    cursor.execute(query, (start_date, required_days))
    available_days = cursor.fetchone()[0]
    
    if available_days >= required_days:
        print(f"    ✅ {factor_name}: 充足 ({available_days}/{required_days} 天)")
    else:
        print(f"    ❌ {factor_name}: 不足 ({available_days}/{required_days} 天)")

def check_data_quality(cursor, start_date, end_date):
    """检查数据质量"""
    query = """
    SELECT 
        COUNT(*) as total_records,
        COUNT(CASE WHEN ClosePrice IS NULL OR ClosePrice = '' THEN 1 END) as missing_prices,
        COUNT(CASE WHEN Volume IS NULL OR Volume = '' THEN 1 END) as missing_volumes,
        ROUND(COUNT(CASE WHEN ClosePrice IS NOT NULL AND ClosePrice != '' THEN 1 END) * 100.0 / COUNT(*), 2) as price_completeness
    FROM daily_price 
    WHERE TradingDate >= ? AND TradingDate <= ?
    """
    
    cursor.execute(query, (start_date, end_date))
    result = cursor.fetchone()
    
    print(f"    {start_date} 到 {end_date}:")
    print(f"      - 总记录: {result[0]:,}")
    print(f"      - 缺失价格: {result[1]:,}")
    print(f"      - 缺失成交量: {result[2]:,}")
    print(f"      - 价格完整率: {result[3]}%")

def analyze_ic_calculation_requirements():
    """分析IC计算的具体要求"""
    print(f"\n🧮 IC计算数据要求分析:")
    print(f"  RollingICIRCalculator需要:")
    print(f"    1. 至少10个有效数据点计算IC")
    print(f"    2. 60天滚动窗口的历史数据")
    print(f"    3. 各因子的历史数据和对应的forward_return列")
    
    # 检查是否有forward_return数据
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 检查是否有因子数据表或计算好的forward_return列
    print(f"\n  📊 检查因子计算相关的数据结构:")
    
    # 查看表中的实际列
    cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]
    
    factor_related_columns = [col for col in column_names if any(keyword in col.lower() 
                             for keyword in ['factor', 'return', 'momentum', 'reversal'])]
    
    print(f"    表 {TABLE_NAME} 中的因子相关列:")
    if factor_related_columns:
        for col in factor_related_columns:
            print(f"      - {col}")
    else:
        print(f"      - 未找到因子相关列 (这可能是问题所在)")
        print(f"      - 实际列名: {column_names}")
    
    conn.close()

def main():
    """主函数"""
    print("🔍 开始滚动窗口数据充足性分析...")
    
    # 分析滚动窗口需求
    analyze_rolling_window_requirements()
    
    # 分析IC计算要求
    analyze_ic_calculation_requirements()
    
    print(f"\n✅ 分析完成")

if __name__ == "__main__":
    main()