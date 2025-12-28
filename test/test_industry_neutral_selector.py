"""
测试行业中性化选股器

验证 IndustryNeutralSelector 的基本功能
"""

import sys
import os
import math
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.pipeline.selectors import IndustryNeutralSelector


class MockData:
    """模拟 Backtrader 的数据对象"""

    def __init__(self, name, signal, industry, suspended=False):
        self._name = name
        self._signal = signal
        self._industry = industry
        self._suspended = suspended

    def __len__(self):
        return 1

    @property
    def combined_signal(self):
        return MockLine(self._signal)

    @property
    def industry(self):
        return MockLine(self._industry)

    @property
    def suspended(self):
        return MockLine(self._suspended)


class MockLine:
    """模拟 Backtrader 的 Line 对象"""

    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        return self.value


def test_equal_strategy():
    """测试等权分配策略"""
    print("=" * 60)
    print("测试 1: 等权分配策略")
    print("=" * 60)

    # 创建模拟数据：3个行业，每个行业4只股票
    mock_datas = [
        # 金融行业
        MockData("600000", 0.95, "金融"),
        MockData("600036", 0.90, "金融"),
        MockData("601318", 0.85, "金融"),
        MockData("601398", 0.80, "金融"),

        # 消费行业
        MockData("600519", 0.88, "消费"),
        MockData("000858", 0.83, "消费"),
        MockData("600887", 0.78, "消费"),
        MockData("002304", 0.73, "消费"),

        # 科技行业
        MockData("000063", 0.92, "科技"),
        MockData("002415", 0.87, "科技"),
        MockData("300750", 0.82, "科技"),
        MockData("688981", 0.77, "科技"),
    ]

    # 创建选股器：总共选10只，等权分配
    selector = IndustryNeutralSelector(
        total_stocks=10,
        strategy='equal',
        stocks_per_industry=3
    )

    # 执行选股
    selected = selector.select(mock_datas)

    print(f"\n总候选股票: {len(mock_datas)}")
    print(f"选中股票数: {len(selected)}")
    print(f"\n选中的股票:")

    # 统计行业分布
    industry_count = {}
    for stock in selected:
        industry = stock.industry[0]
        print(f"  {stock._name}: 信号={stock.combined_signal[0]:.2f}, 行业={industry}")
        industry_count[industry] = industry_count.get(industry, 0) + 1

    print(f"\n行业分布: {industry_count}")

    # 验证
    assert len(selected) <= 10, "选中股票数应该不超过10只"
    assert len(industry_count) == 3, "应该覆盖3个行业"
    print("✅ 测试通过: 等权分配策略")


def test_proportional_strategy():
    """测试比例分配策略"""
    print("\n" + "=" * 60)
    print("测试 2: 比例分配策略")
    print("=" * 60)

    # 创建模拟数据：3个行业，股票数量不同
    mock_datas = [
        # 金融行业 (6只)
        MockData("600000", 0.95, "金融"),
        MockData("600036", 0.90, "金融"),
        MockData("601318", 0.85, "金融"),
        MockData("601398", 0.80, "金融"),
        MockData("601988", 0.75, "金融"),
        MockData("601288", 0.70, "金融"),

        # 消费行业 (3只)
        MockData("600519", 0.88, "消费"),
        MockData("000858", 0.83, "消费"),
        MockData("600887", 0.78, "消费"),

        # 科技行业 (3只)
        MockData("000063", 0.92, "科技"),
        MockData("002415", 0.87, "科技"),
        MockData("300750", 0.82, "科技"),
    ]

    # 创建选股器：总共选10只，按比例分配
    selector = IndustryNeutralSelector(
        total_stocks=10,
        strategy='proportional',
        min_stocks_per_industry=1
    )

    # 执行选股
    selected = selector.select(mock_datas)

    print(f"\n总候选股票: {len(mock_datas)}")
    print(f"选中股票数: {len(selected)}")
    print(f"\n选中的股票:")

    # 统计行业分布
    industry_count = {}
    for stock in selected:
        industry = stock.industry[0]
        print(f"  {stock._name}: 信号={stock.combined_signal[0]:.2f}, 行业={industry}")
        industry_count[industry] = industry_count.get(industry, 0) + 1

    print(f"\n行业分布: {industry_count}")
    print(f"金融行业占比: {industry_count.get('金融', 0)}/{len(selected)}")

    # 验证
    assert len(selected) <= 10, "选中股票数应该不超过10只"
    assert industry_count.get('金融', 0) >= industry_count.get('消费', 0), \
        "金融行业股票数应该多于消费行业（按比例分配）"
    print("✅ 测试通过: 比例分配策略")


def test_top_industries_strategy():
    """测试头部行业策略"""
    print("\n" + "=" * 60)
    print("测试 3: 头部行业策略")
    print("=" * 60)

    # 创建模拟数据：4个行业，信号强度不同
    mock_datas = [
        # 科技行业 (平均信号最强)
        MockData("000063", 0.95, "科技"),
        MockData("002415", 0.92, "科技"),
        MockData("300750", 0.88, "科技"),

        # 消费行业 (平均信号第二)
        MockData("600519", 0.85, "消费"),
        MockData("000858", 0.82, "消费"),
        MockData("600887", 0.78, "消费"),

        # 金融行业 (平均信号第三)
        MockData("600000", 0.75, "金融"),
        MockData("600036", 0.72, "金融"),
        MockData("601318", 0.68, "金融"),

        # 医药行业 (平均信号最弱)
        MockData("600276", 0.65, "医药"),
        MockData("000538", 0.62, "医药"),
        MockData("002422", 0.58, "医药"),
    ]

    # 创建选股器：总共选6只，只在前2个行业中选
    selector = IndustryNeutralSelector(
        total_stocks=6,
        strategy='top_industries',
        top_industries=2
    )

    # 执行选股
    selected = selector.select(mock_datas)

    print(f"\n总候选股票: {len(mock_datas)}")
    print(f"选中股票数: {len(selected)}")
    print(f"\n选中的股票:")

    # 统计行业分布
    industry_count = {}
    for stock in selected:
        industry = stock.industry[0]
        print(f"  {stock._name}: 信号={stock.combined_signal[0]:.2f}, 行业={industry}")
        industry_count[industry] = industry_count.get(industry, 0) + 1

    print(f"\n行业分布: {industry_count}")

    # 验证
    assert len(selected) <= 6, "选中股票数应该不超过6只"
    assert len(industry_count) <= 2, "应该只覆盖最多2个行业"
    assert "医药" not in industry_count, "不应该选择医药行业（信号最弱）"
    print("✅ 测试通过: 头部行业策略")


def test_filter_suspended_and_nan():
    """测试过滤停牌和NaN信号"""
    print("\n" + "=" * 60)
    print("测试 4: 过滤停牌和无效信号")
    print("=" * 60)

    # 创建模拟数据：包含停牌和NaN信号
    mock_datas = [
        # 正常股票
        MockData("600000", 0.95, "金融"),
        MockData("600036", 0.90, "金融"),

        # 停牌股票
        MockData("600519", 0.88, "消费", suspended=True),

        # 信号为NaN
        MockData("000063", float('nan'), "科技"),

        # 行业信息缺失
        MockData("002415", 0.87, None),

        # 正常股票
        MockData("601318", 0.85, "金融"),
        MockData("000858", 0.83, "消费"),
    ]

    # 创建选股器
    selector = IndustryNeutralSelector(
        total_stocks=10,
        strategy='equal'
    )

    # 执行选股
    selected = selector.select(mock_datas)

    print(f"\n总候选股票: {len(mock_datas)}")
    print(f"选中股票数: {len(selected)}")
    print(f"\n选中的股票:")

    for stock in selected:
        print(f"  {stock._name}: 信号={stock.combined_signal[0]:.2f}, 行业={stock.industry[0]}")

    # 验证
    selected_names = [s._name for s in selected]
    assert "600519" not in selected_names, "停牌股票不应该被选中"
    assert "000063" not in selected_names, "NaN信号股票不应该被选中"
    assert "002415" not in selected_names, "无行业信息股票不应该被选中"
    print("✅ 测试通过: 过滤停牌和无效信号")


if __name__ == "__main__":
    test_equal_strategy()
    test_proportional_strategy()
    test_top_industries_strategy()
    test_filter_suspended_and_nan()

    print("\n" + "=" * 60)
    print("🎉 所有测试通过！")
    print("=" * 60)
