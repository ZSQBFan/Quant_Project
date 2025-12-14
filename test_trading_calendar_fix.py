#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试交易日历修复的脚本
"""

import logging
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_manager import DataProviderManager
from data.trading_calendars import TushareTradingCalendar, AkshareTradingCalendar
from data.data_providers import AkshareDataProvider

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_trading_calendar():
    """测试交易日历功能"""
    print("=" * 60)
    print("测试交易日历修复")
    print("=" * 60)
    
    # 测试1: AkshareTradingCalendar (不需要token)
    print("\n1. 测试 AkshareTradingCalendar...")
    try:
        akshare_calendar = AkshareTradingCalendar()
        trading_days = akshare_calendar.get_trading_days("2024-01-01", "2024-01-10")
        print(f"✅ AkshareTradingCalendar 成功获取 {len(trading_days)} 个交易日")
        print(f"   交易日示例: {trading_days[:3]}")
    except Exception as e:
        print(f"❌ AkshareTradingCalendar 失败: {e}")
    
    # 测试2: TushareTradingCalendar (需要token)
    print("\n2. 测试 TushareTradingCalendar...")
    try:
        # 尝试从环境变量获取token
        tushare_token = os.getenv('TUSHARE_TOKEN', 'test_token')
        tushare_calendar = TushareTradingCalendar(token=tushare_token)
        trading_days = tushare_calendar.get_trading_days("2024-01-01", "2024-01-10")
        print(f"✅ TushareTradingCalendar 成功获取 {len(trading_days)} 个交易日")
        print(f"   交易日示例: {trading_days[:3]}")
    except Exception as e:
        print(f"❌ TushareTradingCalendar 失败: {e}")
        print("   (这可能是由于无效的token导致的，是正常的)")
    
    # 测试3: DataProviderManager 初始化
    print("\n3. 测试 DataProviderManager 初始化...")
    try:
        # 创建一个简单的配置
        provider_configs = [
            (AkshareDataProvider, {'adjust': 'hfq'}),
        ]
        
        manager = DataProviderManager(
            provider_configs=provider_configs,
            symbols=['000001', '000002'],
            start_date='2024-01-01',
            end_date='2024-01-05',
            db_path='test_quant_data.db'
        )
        
        print("✅ DataProviderManager 初始化成功")
        print(f"   交易日历提供者类型: {type(manager.calendar_provider).__name__}")
        
        # 测试获取交易日历
        trading_days = manager.calendar_provider.get_trading_days("2024-01-01", "2024-01-10")
        print(f"   获取到 {len(trading_days)} 个交易日")
        
    except Exception as e:
        print(f"❌ DataProviderManager 初始化失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_trading_calendar()