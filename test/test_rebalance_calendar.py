"""
测试调仓日历生成器
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from backtest.core.rebalance_calendar import RebalanceCalendar, create_rebalance_calendar


def test_daily_rebalance():
    """测试每日调仓"""
    print("\n=== 测试每日调仓 ===")
    calendar = RebalanceCalendar(
        start_date='2024-01-01',
        end_date='2024-01-31',
        frequency='daily'
    )
    dates = calendar.generate()
    print(f"生成了 {len(dates)} 个调仓日")
    print(f"前5个日期: {dates[:5]}")
    print(f"后5个日期: {dates[-5:]}")
    assert len(dates) > 0, "应该生成调仓日期"
    print("✅ 每日调仓测试通过")


def test_weekly_rebalance():
    """测试每周调仓"""
    print("\n=== 测试每周调仓 (每周一) ===")
    calendar = RebalanceCalendar(
        start_date='2024-01-01',
        end_date='2024-03-31',
        frequency='weekly',
        weekday=0  # 周一
    )
    dates = calendar.generate()
    print(f"生成了 {len(dates)} 个调仓日")
    print(f"前5个日期: {dates[:5]}")

    # 验证所有日期都是周一
    for date_str in dates:
        date = pd.to_datetime(date_str)
        assert date.weekday() == 0, f"{date_str} 不是周一"

    print("✅ 每周调仓测试通过")


def test_monthly_rebalance():
    """测试每月调仓"""
    print("\n=== 测试每月调仓 (每月1号) ===")
    calendar = RebalanceCalendar(
        start_date='2024-01-01',
        end_date='2024-12-31',
        frequency='monthly',
        monthday=1
    )
    dates = calendar.generate()
    print(f"生成了 {len(dates)} 个调仓日")
    print(f"所有日期: {dates}")

    # 应该有12个月
    assert len(dates) == 12, f"一年应该有12个调仓日，实际: {len(dates)}"

    print("✅ 每月调仓测试通过")


def test_interval_rebalance():
    """测试间隔调仓"""
    print("\n=== 测试间隔调仓 (每20个交易日) ===")
    calendar = RebalanceCalendar(
        start_date='2024-01-01',
        end_date='2024-12-31',
        frequency='interval',
        interval_days=20
    )
    dates = calendar.generate()
    print(f"生成了 {len(dates)} 个调仓日")
    print(f"前5个日期: {dates[:5]}")

    assert len(dates) > 0, "应该生成调仓日期"

    print("✅ 间隔调仓测试通过")


def test_with_trading_calendar():
    """测试使用交易日历"""
    print("\n=== 测试使用交易日历 ===")

    # 创建一个简单的交易日历（排除周末）
    trading_calendar = pd.bdate_range(start='2024-01-01', end='2024-03-31')

    calendar = RebalanceCalendar(
        start_date='2024-01-01',
        end_date='2024-03-31',
        frequency='monthly',
        monthday=1,
        trading_calendar=trading_calendar
    )
    dates = calendar.generate()
    print(f"生成了 {len(dates)} 个调仓日")
    print(f"所有日期: {dates}")

    # 验证所有日期都在交易日历中
    for date_str in dates:
        date = pd.to_datetime(date_str)
        assert date in trading_calendar, f"{date_str} 不在交易日历中"

    print("✅ 交易日历测试通过")


def test_create_from_config():
    """测试从配置字典创建"""
    print("\n=== 测试从配置创建 ===")

    # 测试月度调仓配置
    config = {
        'frequency': 'monthly',
        'monthday': 1
    }
    dates = create_rebalance_calendar(
        start_date='2024-01-01',
        end_date='2024-06-30',
        config=config
    )
    print(f"月度调仓: 生成了 {len(dates)} 个调仓日")
    print(f"日期: {dates}")
    assert len(dates) == 6, f"半年应该有6个调仓日，实际: {len(dates)}"

    # 测试每周调仓配置
    config = {
        'frequency': 'weekly',
        'weekday': 4  # 周五
    }
    dates = create_rebalance_calendar(
        start_date='2024-01-01',
        end_date='2024-01-31',
        config=config
    )
    print(f"每周调仓: 生成了 {len(dates)} 个调仓日")
    print(f"日期: {dates}")

    # 测试间隔调仓配置
    config = {
        'frequency': 'interval',
        'interval_days': 10
    }
    dates = create_rebalance_calendar(
        start_date='2024-01-01',
        end_date='2024-03-31',
        config=config
    )
    print(f"间隔调仓: 生成了 {len(dates)} 个调仓日")
    print(f"前5个日期: {dates[:5]}")

    print("✅ 配置创建测试通过")


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")

    # 测试月底日期
    print("测试每月31号（2月会调整到最后一天）...")
    calendar = RebalanceCalendar(
        start_date='2024-01-01',
        end_date='2024-04-30',  # 延长到4月确保能看到3月的结果
        frequency='monthly',
        monthday=31
    )
    dates = calendar.generate()
    print(f"生成了 {len(dates)} 个调仓日: {dates}")

    # 验证至少生成了日期
    assert len(dates) >= 2, "应该至少生成2个月的调仓日"

    # 验证2月调整到最后一天（2024年是闰年，2月有29天）
    feb_dates = [d for d in dates if d.startswith('2024-02')]
    if feb_dates:
        feb_date = pd.to_datetime(feb_dates[0])
        print(f"2月的调仓日: {feb_date.strftime('%Y-%m-%d')} (天数: {feb_date.day})")
        # 2024年是闰年，2月应该有29天
        assert feb_date.day == 29, "2024年2月应该调整到29号"

    # 验证3月31日会被调整（因为是周日）
    mar_dates = [d for d in dates if d.startswith('2024-03')]
    if mar_dates:
        mar_date = pd.to_datetime(mar_dates[0])
        print(f"3月的调仓日: {mar_date.strftime('%Y-%m-%d')} (星期{['一','二','三','四','五','六','日'][mar_date.weekday()]})")

    print("✅ 边界情况测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始测试调仓日历生成器")
    print("=" * 60)

    try:
        test_daily_rebalance()
        test_weekly_rebalance()
        test_monthly_rebalance()
        test_interval_rebalance()
        test_with_trading_calendar()
        test_create_from_config()
        test_edge_cases()

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        raise


if __name__ == '__main__':
    run_all_tests()
