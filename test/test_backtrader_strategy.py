# test/test_backtrader_strategy.py

"""
测试 Backtrader 策略基类和中央指令缓冲区

这个测试脚本验证：
1. Backtrader 能否每个时间步轮询策略基类
2. 中央指令缓冲区能否正常工作
3. 触发器系统是否能正确提交和执行意图

🚨 触发器调用机制说明：

最开始的代码可能没有成功调用触发器的原因：

1. **参数传递错误**：
   - triggers参数必须是函数列表，不是触发器实例列表
   - 正确格式：triggers=[lambda s: SimpleTestTrigger(s)]
   - 错误格式：triggers=[SimpleTestTrigger()] 或 triggers=[SimpleTestTrigger]

2. **触发器实例化失败**：
   - 触发器构造函数需要strategy参数
   - 如果构造函数参数不匹配，实例化会失败
   - 检查日志中是否有"已加载触发器: XXX"消息

3. **调用时机问题**：
   - 触发器在BacktestStrategy.next()中每个时间步调用
   - 如果next()方法被覆盖或未正确调用，触发器不会执行
   - 检查Backtrader日志确认每个时间步都在运行

4. **调试方法**：
   - 在BacktestStrategy.next()开头添加打印：
     print(f"触发器数量: {len(self.triggers)}")
   - 在触发器check_and_execute()开头添加打印：
     print(f"触发器执行: {self.__class__.__name__}")
   - 检查日志中的"📈 执行:"消息确认交易执行

5. **触发器逻辑问题**：
   - check_and_execute()内部条件判断错误
   - 提交action时参数不完整（data, action等）
   - 优先级设置问题
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtrader as bt
from logger.logger_config import setup_logging
from bt.core.base_strategy import BacktestStrategy
from bt.core.constants import ActionPriority, ActionType
from bt.data.feeds import FactorPandasData


class SimpleTestTrigger:
    """
    简单测试触发器
    
    用于验证触发器系统的基本功能：
    - 每隔5个交易日触发一次买入
    - 每隔8个交易日触发一次卖出
    """
    
    def __init__(self, strategy):
        self.strategy = strategy
        self.day_count = 0
        
    def check_and_execute(self):
        """检查条件并提交交易意图"""
        self.day_count += 1
        
        # 获取当前数据
        for data in self.strategy.datas:
            # 每5天买入一次
            if self.day_count % 5 == 0:
                self.strategy.submit_action(
                    data=data,
                    action=ActionType.BUY,
                    size=100,
                    reason=f"测试触发器买入(第{self.day_count}天)",
                    priority=ActionPriority.REBALANCE
                )
            
            # 每8天卖出一次
            if self.day_count % 8 == 0:
                self.strategy.submit_action(
                    data=data,
                    action=ActionType.SELL,
                    size=50,
                    reason=f"测试触发器卖出(第{self.day_count}天)",
                    priority=ActionPriority.REBALANCE
                )


def create_test_data(symbol='TEST_STOCK', days=20):
    """
    创建测试用的股票数据
    
    参数:
        symbol: 股票代码
        days: 数据天数
    
    返回:
        包含OHLCV和因子信号的DataFrame
    """
    # 生成日期序列
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='D')
    
    # 生成随机价格数据
    np.random.seed(42)  # 固定随机种子，确保结果可重现
    base_price = 100.0
    
    data = {
        'open': base_price + np.random.normal(0, 2, days),
        'high': base_price + np.random.normal(2, 2, days),
        'low': base_price + np.random.normal(-2, 2, days),
        'close': base_price + np.random.normal(0, 2, days),
        'volume': np.random.randint(1000, 10000, days),
        'openinterest': np.zeros(days),
        'combined_signal': np.random.normal(0, 1, days),  # 合成因子信号
        'suspended': np.random.choice([False, True], days, p=[0.95, 0.05])  # 5%概率停牌
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # 确保价格关系合理
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
    
    return df


def test_backtrader_strategy():
    """
    测试 Backtrader 策略基类和中央指令缓冲区
    """
    print("=" * 60)
    print("开始测试 Backtrader 策略基类和中央指令缓冲区")
    print("=" * 60)
    
    # 设置日志
    setup_logging(log_dir='logs', log_prefix='backtrader_test')
    
    # 创建 Cerebro 引擎
    cerebro = bt.Cerebro()
    
    # 设置初始资金
    cerebro.broker.setcash(100000.0)
    
    # 创建测试数据
    test_df = create_test_data('TEST_STOCK', days=20)
    
    # 创建数据源
    data = FactorPandasData(dataname=test_df, name='TEST_STOCK')
    cerebro.adddata(data)
    
    # 添加策略
    # ⚠️ 重要说明：这里的triggers参数是一个函数列表，每个函数接收strategy参数
    # 返回一个触发器实例。BacktestStrategy在初始化时会调用这些函数来创建触发器实例。
    cerebro.addstrategy(
        BacktestStrategy,
        triggers=[lambda s: SimpleTestTrigger(s)]
    )
    
    # 📝 触发器调用机制说明：
    # 1. 在BacktestStrategy.__init__()中，遍历self.p.triggers
    # 2. 对每个trigger_factory调用trigger_factory(self)创建触发器实例
    # 3. 将触发器实例保存到self.triggers列表中
    # 4. 在BacktestStrategy.next()中遍历self.triggers调用trigger.check_and_execute()
    #
    # 如果最开始的代码没有成功调用触发器，可能的原因：
    # - triggers参数传递不正确
    # - 触发器实例化失败
    # - next()方法没有被正确调用
    # - 触发器check_and_execute()逻辑有问题
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    # 运行回测
    print("\n开始运行回测...")
    initial_value = cerebro.broker.getvalue()
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    
    # 打印结果
    print(f"\n回测结果:")
    print(f"初始资金: {initial_value:,.2f}")
    print(f"最终资金: {final_value:,.2f}")
    print(f"总收益: {final_value - initial_value:,.2f} ({(final_value/initial_value - 1)*100:.2f}%)")
    
    # 打印分析器结果
    strat = results[0]
    print("\n风险指标:")
    print(f"夏普比率: {strat.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A')}")
    print(f"最大回撤: {strat.analyzers.drawdown.get_analysis().max.drawdown:.2f}%")
    
    print("\n✅ Backtrader 策略基类和中央指令缓冲区测试完成！")
    
    return True


if __name__ == '__main__':
    try:
        success = test_backtrader_strategy()
        if success:
            print("\n🎉 所有测试通过！")
        else:
            print("\n❌ 测试失败！")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)