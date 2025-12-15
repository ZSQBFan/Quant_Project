# test_selectors.py
"""
选股器模块测试文件
"""

import math
import sys
import os

# 添加项目路径到Python路径
sys.path.append('/Volumes/叽叽叽/Code/quant_project_3.12_macmini')

from bt.pipeline.selectors import TopNSelector, SelectorBase


class MockData:
    """模拟股票数据对象"""
    def __init__(self, name, signal_value, suspended=False, has_data=True):
        self._name = name
        self.suspended = [suspended] if has_data else []
        self.combined_signal = [signal_value] if has_data else []
        self._has_data = has_data
    
    def __len__(self):
        return 1 if self._has_data else 0


def test_top_n_selector():
    """测试 TopNSelector 的基本功能"""
    print("🧪 测试 TopNSelector 基本功能")
    
    # 创建测试数据
    test_data = [
        MockData("股票A", 0.8, suspended=False),     # 有效数据，信号强
        MockData("股票B", 0.6, suspended=False),     # 有效数据，信号中等  
        MockData("股票C", 0.9, suspended=False),     # 有效数据，信号最强
        MockData("股票D", math.nan, suspended=False), # 信号为NaN
        MockData("股票E", 0.7, suspended=True),      # 停牌
        MockData("股票F", 0.5, suspended=False),     # 有效数据，信号弱
        MockData("无数据股票", 0.4, has_data=False), # 无数据
    ]
    
    # 创建选股器，选择前3只
    selector = TopNSelector(top_n=3, name="测试选股器")
    
    # 执行选股
    selected = selector.select(test_data)
    
    print(f"候选股票数: {len(test_data)}")
    print(f"选中股票数: {len(selected)}")
    
    # 验证结果
    assert len(selected) == 3, f"期望选中3只股票，实际选中{len(selected)}只"
    
    # 验证排序（信号强度降序）
    signals = [stock.combined_signal[0] for stock in selected]
    assert signals == sorted(signals, reverse=True), "股票未按信号强度降序排序"
    
    print("✅ 基本功能测试通过")
    
    # 打印选中的股票
    print("\n选中的股票:")
    for i, stock in enumerate(selected, 1):
        signal = stock.combined_signal[0]
        print(f"  {i}. {stock._name} (信号: {signal:.2f})")


def test_edge_cases():
    """测试边界情况"""
    print("\n🧪 测试边界情况")
    
    # 测试空列表
    selector = TopNSelector(top_n=5)
    result = selector.select([])
    assert len(result) == 0, "空列表应该返回空结果"
    print("✅ 空列表测试通过")
    
    # 测试所有股票都被过滤
    all_filtered_data = [
        MockData("停牌股票1", 0.9, suspended=True),
        MockData("停牌股票2", 0.8, suspended=True),
        MockData("NaN股票", math.nan, suspended=False),
        MockData("无数据股票", 0.7, has_data=False),
    ]
    
    result = selector.select(all_filtered_data)
    assert len(result) == 0, "所有股票被过滤时应返回空结果"
    print("✅ 全过滤情况测试通过")
    
    # 测试股票数量少于top_n
    limited_data = [
        MockData("股票1", 0.9),
        MockData("股票2", 0.8),
    ]
    
    result = selector.select(limited_data)
    assert len(result) == 2, "股票数量少于top_n时应返回所有有效股票"
    print("✅ 股票数量不足测试通过")


if __name__ == "__main__":
    print("🚀 开始测试选股器模块")
    
    try:
        test_top_n_selector()
        test_edge_cases()
        print("\n🎉 所有测试通过！选股器模块实现正确。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()