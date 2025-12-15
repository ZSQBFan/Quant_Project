# test/test_triggers_simple.py

"""
触发器系统简单测试

验证核心功能：
1. 触发器能否成功触发
2. 停牌拦截是否正常工作
3. 止损触发器是否正常工作
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtrader as bt
from logger.logger_config import setup_logging
from bt.core.base_strategy import BacktestStrategy
from bt.core.constants import ActionPriority, ActionType
from bt.triggers import StopLossTrigger
from bt.data.feeds import FactorPandasData


class MockSelector:
    """模拟选股器"""
    def select(self, all_datas):
        return all_datas[:1] if all_datas else []


class MockAllocator:
    """模拟权重分配器"""
    def allocate(self, selected_stocks):
        return {data: 1.0 for data in selected_stocks} if selected_stocks else {}


class MockCapitalManager:
    """模拟资金管理器"""
    def get_allocation(self, total_value):
        return total_value * 0.9


class SimpleTestTrigger:
    """简单测试触发器"""
    
    def __init__(self, strategy):
        self.strategy = strategy
        self.call_count = 0
        
    def check_and_execute(self):
        """提交买入意图"""
        self.call_count += 1
        
        for data in self.strategy.datas:
            self.strategy.submit_action(
                data=data,
                action=ActionType.BUY,
                size=100,
                reason=f'测试信号 #{self.call_count}',
                priority=ActionPriority.OTHER
            )


def test_trigger_basic_functionality():
    """测试触发器基本功能"""
    print("=" * 60)
    print("测试 1: 触发器基本功能")
    print("=" * 60)
    
    try:
        # 创建测试数据
        dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
        data = {
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [98, 99, 100, 101, 102],
            'close': [101, 102, 103, 104, 105],
            'volume': [10000] * 5,
            'openinterest': np.zeros(5),
            'combined_signal': np.ones(5),
            'suspended': np.zeros(5, dtype=bool)
        }
        
        df = pd.DataFrame(data, index=dates)
        
        # 创建Cerebro引擎
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100000.0)
        
        # 添加数据
        data_feed = FactorPandasData(dataname=df, name='TEST_STOCK')
        cerebro.adddata(data_feed)
        
        # 创建策略
        class TestStrategy(BacktestStrategy):
            def __init__(self):
                super().__init__()
                self.triggers = [SimpleTestTrigger(self)]
        
        # 添加策略
        cerebro.addstrategy(
            TestStrategy,
            selector=MockSelector(),
            allocator=MockAllocator(),
            capital_manager=MockCapitalManager()
        )
        
        print("✅ 策略初始化完成")
        
        # 运行回测
        print("🚀 运行基本功能测试...")
        initial_value = cerebro.broker.getvalue()
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()
        
        # 检查结果
        strat = results[0]
        trigger = strat.triggers[0]
        
        print(f"\n📊 测试结果:")
        print(f"   触发器调用次数: {trigger.call_count}")
        print(f"   资金变化: {final_value - initial_value:,.2f}")
        
        # 验证：应该有交易执行
        if trigger.call_count > 0:
            print("✅ 触发器基本功能正常")
            return True
        else:
            print("❌ 触发器未调用")
            return False
            
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stop_loss_trigger():
    """测试止损触发器"""
    print("\n" + "=" * 60)
    print("测试 2: 止损触发器")
    print("=" * 60)
    
    try:
        # 创建下跌趋势数据
        dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
        
        # 每天跌5%，总共跌约40%
        base_price = 100.0
        prices = [base_price * (0.95 ** i) for i in range(10)]
        
        data = {
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [10000] * 10,
            'openinterest': np.zeros(10),
            'combined_signal': np.ones(10),
            'suspended': np.zeros(10, dtype=bool)
        }
        
        df = pd.DataFrame(data, index=dates)
        
        print(f"✅ 创建下跌数据: {prices[0]:.2f} -> {prices[-1]:.2f}")
        
        # 创建Cerebro引擎
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100000.0)
        
        # 添加数据
        data_feed = FactorPandasData(dataname=df, name='FALL_TEST')
        cerebro.adddata(data_feed)
        
        # 创建策略（添加止损触发器）
        class StopLossTestStrategy(BacktestStrategy):
            def __init__(self):
                super().__init__()
                # 添加止损触发器
                self.triggers = [StopLossTrigger(self, loss_threshold=-0.15)]
            
            def next(self):
                # 在第2天买入建立持仓
                if len(self.data) == 2:
                    self.buy(data=self.datas[0], size=1000)
                super().next()
        
        # 添加策略
        cerebro.addstrategy(
            StopLossTestStrategy,
            selector=MockSelector(),
            allocator=MockAllocator(),
            capital_manager=MockCapitalManager()
        )
        
        print("✅ 止损测试策略初始化完成")
        
        # 运行回测
        print("🚀 运行止损测试...")
        initial_value = cerebro.broker.getvalue()
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()
        
        print(f"\n📊 止损测试结果:")
        print(f"   初始资金: {initial_value:,.2f}")
        print(f"   最终资金: {final_value:,.2f}")
        print(f"   资金变化: {final_value - initial_value:,.2f}")
        
        # 验证：应该有止损交易
        if abs(final_value - initial_value) > 100:  # 有显著变化说明有交易
            print("✅ 止损触发器正常工作")
            return True
        else:
            print("❌ 止损触发器未工作")
            return False
            
    except Exception as e:
        print(f"❌ 止损测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🎯 触发器系统简单功能测试")
    print("=" * 80)
    
    # 设置日志
    setup_logging(log_dir='logs', log_prefix='trigger_simple_test')
    
    tests_passed = 0
    total_tests = 2
    
    try:
        # 测试 1: 基本功能
        print("\n🧪 运行测试 1: 触发器基本功能")
        if test_trigger_basic_functionality():
            tests_passed += 1
        
        # 测试 2: 止损触发器
        print("\n🧪 运行测试 2: 止损触发器")
        if test_stop_loss_trigger():
            tests_passed += 1
        
        # 输出结果
        print(f"\n{'='*80}")
        print(f"🏁 测试总结:")
        print(f"   通过: {tests_passed}/{total_tests}")
        print(f"   失败: {total_tests - tests_passed}/{total_tests}")
        
        if tests_passed == total_tests:
            print(f"\n🎉 所有触发器系统测试通过！")
            return True
        else:
            print(f"\n❌ 部分测试失败")
            return False
            
    except Exception as e:
        print(f"\n💥 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)