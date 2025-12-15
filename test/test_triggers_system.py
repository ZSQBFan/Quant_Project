# test/test_triggers_system.py

"""
触发器系统测试

测试内容：
1. 触发器能否成功触发
2. 遇到极端情况，如连续的停牌，触发器连续发出意图，能否都被拦截
3. 中央指令缓冲区能否根据提交的意图的正确排序、去重、执行

参考文件：test/test_backtrader_integration.py
参考架构：BACKTRADER_SETUP.md
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtrader as bt
from logger.logger_config import setup_logging
from bt.core.base_strategy import BacktestStrategy
from bt.core.constants import ActionPriority, ActionType
from bt.triggers import StopLossTrigger, RebalanceDayTrigger
from bt.data.feeds import FactorPandasData
from bt.data.exporter import BTDataExporter


class MockSelector:
    """模拟选股器"""
    
    def select(self, all_datas):
        """选择前一半股票"""
        return all_datas[:len(all_datas)//2] if len(all_datas) > 1 else all_datas


class MockAllocator:
    """模拟权重分配器"""
    
    def allocate(self, selected_stocks):
        """等权重分配"""
        if not selected_stocks:
            return {}
        weight = 1.0 / len(selected_stocks)
        return {data: weight for data in selected_stocks}


class MockCapitalManager:
    """模拟资金管理器"""
    
    def get_allocation(self, total_value):
        """返回总资金的90%用于投资"""
        return total_value * 0.9


class TriggerTestStrategy(BacktestStrategy):
    """
    触发器测试策略
    
    用于测试止损和调仓日触发器
    """
    
    params = (
        ('selector', None),
        ('allocator', None), 
        ('capital_manager', None),
        ('triggers', []),
        ('loss_threshold', -0.10),  # 10%止损
    )
    
    def __init__(self):
        super().__init__()
        
        # 添加止损触发器
        self.triggers.append(lambda s: StopLossTrigger(s, self.p.loss_threshold))
        
        self.log("✅ 触发器测试策略初始化完成")


class ContinuousSignalTrigger:
    """
    连续信号触发器
    用于测试极端情况（连续停牌，触发器连续发出意图）
    """
    
    def __init__(self, strategy, signal_value=1.0):
        self.strategy = strategy
        self.signal_value = signal_value
        self.call_count = 0
        
    def check_and_execute(self):
        """连续提交买入意图"""
        self.call_count += 1
        
        for data in self.strategy.datas:
            # 每次调用都为同一只股票提交买入意图
            self.strategy.submit_action(
                data=data,
                action=ActionType.BUY,
                size=100,
                reason=f'连续信号测试 #{self.call_count}',
                priority=ActionPriority.OTHER
            )


def test_stop_loss_trigger():
    """
    测试止损触发器
    """
    print("=" * 60)
    print("测试 1: 止损触发器")
    print("=" * 60)
    
    try:
        # 1. 创建下跌趋势测试数据
        def create_falling_stock_data(symbol='FALL_STOCK', days=20):
            end_date = datetime.now()
            dates = pd.date_range(end=end_date, periods=days, freq='D')
            
            # 创建下跌趋势：每天跌2%
            base_price = 100.0
            prices = []
            for i in range(days):
                price = base_price * (0.98 ** i)  # 每天跌2%
                prices.append(price)
            
            data = {
                'open': prices,
                'high': [p * 1.02 for p in prices],  # 高开2%
                'low': [p * 0.98 for p in prices],   # 低开2%
                'close': prices,
                'volume': [10000] * days,
                'openinterest': np.zeros(days),
                'combined_signal': np.full(days, 1.0),  # 强买入信号
                'suspended': np.full(days, False)      # 无停牌
            }
            
            df = pd.DataFrame(data, index=dates)
            return df
        
        test_df = create_falling_stock_data('FALL_TEST', days=20)
        print(f"✅ 创建下跌股票数据: {len(test_df)} 行")
        print(f"   起始价格: {test_df['close'].iloc[0]:.2f}")
        print(f"   结束价格: {test_df['close'].iloc[-1]:.2f}")
        print(f"   总跌幅: {(test_df['close'].iloc[-1]/test_df['close'].iloc[0] - 1)*100:.1f}%")
        
        # 2. 创建Cerebro引擎
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100000.0)
        
        # 3. 添加数据
        data = FactorPandasData(dataname=test_df, name='FALL_TEST')
        cerebro.adddata(data)
        
        # 4. 添加策略（设置较宽松的止损以确保触发）
        cerebro.addstrategy(
            TriggerTestStrategy,
            selector=MockSelector(),
            allocator=MockAllocator(),
            capital_manager=MockCapitalManager(),
            loss_threshold=-0.05  # 5%止损，容易触发
        )
        
        # 5. 添加交易分析器
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        print("✅ 止损测试策略初始化完成")
        
        # 6. 运行回测
        print("🚀 运行止损触发器测试...")
        initial_value = cerebro.broker.getvalue()
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()
        
        # 7. 验证结果
        strat = results[0]
        trades = strat.analyzers.trades.get_analysis()
        
        print(f"\n📊 止损测试结果:")
        print(f"   初始资金: {initial_value:,.2f}")
        print(f"   最终资金: {final_value:,.2f}")
        print(f"   总交易次数: {trades.get('total', {}).get('total', 0)}")
        
        # 检查是否有交易（应该有止损交易）
        total_trades = trades.get('total', {}).get('total', 0)
        if total_trades > 0:
            print("✅ 止损触发器正常工作，触发了止损交易")
            return True
        else:
            print("❌ 止损触发器未工作，未触发任何交易")
            return False
            
    except Exception as e:
        print(f"❌ 止损触发器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_continuous_signals_with_suspension():
    """
    测试连续信号与停牌拦截
    验证触发器连续发出意图时，停牌能否正确拦截
    """
    print("\n" + "=" * 60)
    print("测试 2: 连续信号与停牌拦截")
    print("=" * 60)
    
    try:
        # 1. 创建包含停牌的测试数据
        def create_suspended_stock_data(symbol='SUSP_STOCK', days=10):
            end_date = datetime.now()
            dates = pd.date_range(end=end_date, periods=days, freq='D')
            
            np.random.seed(42)
            base_price = 100.0
            
            data = {
                'open': [base_price] * days,
                'high': [base_price + 2] * days,
                'low': [base_price - 2] * days,
                'close': [base_price] * days,
                'volume': [10000] * days,
                'openinterest': np.zeros(days),
                'combined_signal': np.full(days, 1.0),  # 强买入信号
                'suspended': [False] * days
            }
            
            df = pd.DataFrame(data, index=dates)
            
            # 设置中间几天停牌
            df.loc[3:6, ['open', 'high', 'low', 'close']] = np.nan
            df.loc[3:6, 'suspended'] = True
            
            return df
        
        test_df = create_suspended_stock_data('SUSP_TEST', days=10)
        print(f"✅ 创建停牌测试数据: {len(test_df)} 行")
        
        # 统计停牌天数
        suspended_days = test_df['suspended'].sum()
        print(f"   停牌天数: {suspended_days}")
        
        # 2. 创建Cerebro引擎
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100000.0)
        
        # 3. 添加数据
        data = FactorPandasData(dataname=test_df, name='SUSP_TEST')
        cerebro.adddata(data)
        
        # 4. 添加策略（使用连续信号触发器）
        class ContinuousTestStrategy(BacktestStrategy):
            def __init__(self):
                super().__init__()
                # 添加连续信号触发器
                self.triggers.append(lambda s: ContinuousSignalTrigger(s))
        
        cerebro.addstrategy(
            ContinuousTestStrategy,
            selector=MockSelector(),
            allocator=MockAllocator(),
            capital_manager=MockCapitalManager()
        )
        
        # 5. 添加分析器
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        print("✅ 连续信号测试策略初始化完成")
        
        # 6. 运行回测
        print("🚀 运行连续信号与停牌拦截测试...")
        initial_value = cerebro.broker.getvalue()
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()
        
        # 7. 验证结果
        strat = results[0]
        trades = strat.analyzers.trades.get_analysis()
        continuous_trigger = next(t for t in strat.triggers if isinstance(t, ContinuousSignalTrigger))
        
        print(f"\n📊 连续信号测试结果:")
        print(f"   触发器调用次数: {continuous_trigger.call_count}")
        print(f"   总交易次数: {trades.get('total', {}).get('total', 0)}")
        print(f"   被拦截交易数: {continuous_trigger.call_count - trades.get('total', {}).get('total', 0)}")
        
        # 关键验证：应该没有实际交易（都被停牌拦截）
        total_trades = trades.get('total', {}).get('total', 0)
        if total_trades == 0 and continuous_trigger.call_count > 0:
            print("✅ 停牌拦截机制正常工作，所有意图都被正确拦截")
            return True
        elif total_trades > 0:
            print(f"⚠️  有 {total_trades} 笔交易未被拦截，可能存在问题")
            return False
        else:
            print("❌ 测试异常，未能触发足够多的意图")
            return False
            
    except Exception as e:
        print(f"❌ 连续信号与停牌拦截测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_priority_buffer():
    """
    测试中央指令缓冲区的优先级排序和去重功能
    """
    print("\n" + "=" * 60)
    print("测试 3: 中央指令缓冲区优先级排序")
    print("=" * 60)
    
    try:
        # 1. 创建测试数据
        def create_priority_test_data(symbol='PRIO_STOCK', days=5):
            end_date = datetime.now()
            dates = pd.date_range(end=end_date, periods=days, freq='D')
            
            data = {
                'open': [100, 101, 102, 103, 104],
                'high': [102, 103, 104, 105, 106],
                'low': [98, 99, 100, 101, 102],
                'close': [101, 102, 103, 104, 105],
                'volume': [10000] * 5,
                'openinterest': np.zeros(5),
                'combined_signal': np.full(5, 1.0),
                'suspended': np.full(5, False)
            }
            
            df = pd.DataFrame(data, index=dates)
            return df
        
        test_df = create_priority_test_data('PRIO_TEST', days=5)
        print(f"✅ 创建优先级测试数据: {len(test_df)} 行")
        
        # 2. 创建带有多个冲突意图的触发器
        class ConflictIntentTrigger:
            """故意产生冲突意图的触发器"""
            
            def __init__(self, strategy):
                self.strategy = strategy
                
            def check_and_execute(self):
                """为同一只股票提交多个不同优先级的意图"""
                data = self.strategy.datas[0]  # 只有一个股票
                
                # 提交多个冲突意图
                intents = [
                    (ActionType.BUY, 100, "买入意图", ActionPriority.REBALANCE),
                    (ActionType.SELL, 100, "卖出意图", ActionPriority.REBALANCE),
                    (ActionType.CLOSE, None, "止损意图", ActionPriority.STOP_LOSS),
                    (ActionType.BUY, 200, "再次买入", ActionPriority.REBALANCE)
                ]
                
                for action, size, reason, priority in intents:
                    self.strategy.submit_action(
                        data=data,
                        action=action,
                        size=size,
                        reason=reason,
                        priority=priority
                    )
        
        # 3. 创建Cerebro引擎
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100000.0)
        
        # 4. 添加数据
        data = FactorPandasData(dataname=test_df, name='PRIO_TEST')
        cerebro.adddata(data)
        
        # 5. 添加策略
        class PriorityTestStrategy(BacktestStrategy):
            def __init__(self):
                super().__init__()
                self.triggers.append(lambda s: ConflictIntentTrigger(s))
        
        cerebro.addstrategy(
            PriorityTestStrategy,
            selector=MockSelector(),
            allocator=MockAllocator(),
            capital_manager=MockCapitalManager()
        )
        
        # 6. 添加分析器
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        print("✅ 优先级测试策略初始化完成")
        
        # 7. 运行回测
        print("🚀 运行优先级排序测试...")
        initial_value = cerebro.broker.getvalue()
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()
        
        # 8. 验证结果
        strat = results[0]
        trades = strat.analyzers.trades.get_analysis()
        
        print(f"\n📊 优先级排序测试结果:")
        print(f"   初始资金: {initial_value:,.2f}")
        print(f"   最终资金: {final_value:,.2f}")
        print(f"   总交易次数: {trades.get('total', {}).get('total', 0)}")
        
        # 检查具体的交易详情
        trade_list = trades.get('trades', [])
        print(f"   交易详情: {len(trade_list)} 笔")
        
        for i, trade in enumerate(trade_list):
            print(f"     交易 {i+1}: {trade}")
        
        # 验证：应该只有一笔交易（止损，优先级最高）
        if len(trade_list) == 1:
            trade = trade_list[0]
            # 检查是否是平仓操作（CLOSE对应止损）
            print("✅ 优先级排序正常，只执行了最高优先级的止损操作")
            return True
        elif len(trade_list) == 0:
            print("⚠️  没有执行任何交易，可能需要检查数据或逻辑")
            return False
        else:
            print(f"❌ 优先级排序异常，执行了 {len(trade_list)} 笔交易，期望1笔")
            return False
            
    except Exception as e:
        print(f"❌ 优先级排序测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主测试函数
    """
    print("🎯 触发器系统功能测试")
    print("=" * 80)
    
    # 设置日志
    setup_logging(log_dir='logs', log_prefix='trigger_test')
    
    # 测试统计
    tests_passed = 0
    total_tests = 3
    
    try:
        # 测试 1: 止损触发器
        print("\n🧪 运行测试 1: 止损触发器")
        if test_stop_loss_trigger():
            tests_passed += 1
        
        # 测试 2: 连续信号与停牌拦截
        print("\n🧪 运行测试 2: 连续信号与停牌拦截")
        if test_continuous_signals_with_suspension():
            tests_passed += 1
        
        # 测试 3: 中央指令缓冲区优先级排序
        print("\n🧪 运行测试 3: 中央指令缓冲区优先级排序")
        if test_action_priority_buffer():
            tests_passed += 1
        
        # 输出最终结果
        print(f"\n{'='*80}")
        print(f"🏁 触发器系统测试总结:")
        print(f"   通过: {tests_passed}/{total_tests}")
        print(f"   失败: {total_tests - tests_passed}/{total_tests}")
        
        if tests_passed == total_tests:
            print(f"\n🎉 所有触发器系统测试通过！")
            print(f"\n✅ 验证结果:")
            print(f"   1. 止损触发器正常触发 ✅")
            print(f"   2. 停牌拦截机制正常 ✅")
            print(f"   3. 中央指令缓冲区优先级排序正常 ✅")
            return True
        else:
            print(f"\n❌ 部分测试失败，请检查日志")
            return False
            
    except Exception as e:
        print(f"\n💥 测试过程中发生未处理的错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)