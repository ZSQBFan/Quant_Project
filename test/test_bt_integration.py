"""
测试Backtrader与报告生成器的集成
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import backtrader as bt
from backtest.data.feeds import FactorPandasData
from backtest.reports.report_generator import ReportGenerator
from datetime import datetime

class SimpleStrategy(bt.Strategy):
    def __init__(self):
        self.order = None

    def next(self):
        # 简单策略：什么都不做，只是让回测运行
        pass

def test_backtrader_integration():
    """测试Backtrader与报告生成器的完整集成"""
    print("=" * 60)
    print("测试Backtrader与报告生成器集成")
    print("=" * 60)

    # 1. 创建Cerebro引擎
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(1000000)
    cerebro.broker.setcommission(commission=0.001)

    # 2. 准备测试数据
    dates = pd.date_range('2024-01-01', periods=100, freq='B')
    np.random.seed(42)

    base_price = 100.0
    price_changes = np.random.randn(100).cumsum() * 0.5

    test_df = pd.DataFrame({
        'open': base_price + price_changes,
        'high': base_price + price_changes + np.abs(np.random.randn(100)) * 0.5,
        'low': base_price + price_changes - np.abs(np.random.randn(100)) * 0.5,
        'close': base_price + price_changes + np.random.randn(100) * 0.2,
        'volume': np.random.randint(1000, 10000, 100),
        'openinterest': 0,
        'combined_signal': np.random.randn(100),  # 确保没有NaN
        'suspended': False,
        'industry': 1  # 使用整数而不是字符串
    }, index=dates)

    print(f"\n创建测试数据: {len(test_df)} 行")
    print(f"日期范围: {test_df.index[0]} 到 {test_df.index[-1]}")

    # 3. 添加数据到Cerebro
    data = FactorPandasData(dataname=test_df, name='TEST_STOCK')
    cerebro.adddata(data)

    # 4. 添加策略
    cerebro.addstrategy(SimpleStrategy)

    # 5. 添加观察者（这很重要！）
    cerebro.addobserver(bt.observers.Value)

    # 6. 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print("\n运行Backtrader回测...")
    initial_value = cerebro.broker.getvalue()
    print(f"初始资金: {initial_value:,.2f}")

    # 7. 运行回测
    results = cerebro.run()

    final_value = cerebro.broker.getvalue()
    print(f"最终净值: {final_value:,.2f}")
    print(f"收益: {((final_value/initial_value - 1) * 100):.2f}%")

    # 8. 检查结果
    if not results:
        print("❌ 回测未返回结果")
        return False

    strat = results[0]
    print(f"\n检查策略对象...")
    print(f"  - 有observers? {hasattr(strat, 'observers')}")

    if hasattr(strat, 'observers'):
        print(f"  - 有value observer? {hasattr(strat.observers, 'value')}")
        if hasattr(strat.observers, 'value'):
            v_list = list(strat.observers.value.value)
            print(f"  - 净值点数: {len(v_list)}")
            if len(v_list) > 0:
                print(f"  - 净值范围: {min(v_list):.2f} ~ {max(v_list):.2f}")

            # 检查日期数组
            if len(strat.datas) > 0:
                d_array = strat.datas[0].datetime.array
                print(f"  - 日期数组长度: {len(d_array)}")

    # 9. 生成报告
    print(f"\n生成报告...")
    output_path = 'output/bt_reports/test_bt_integration.html'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    report_gen = ReportGenerator()
    try:
        report_gen.generate(
            strategy_results=results,
            output_path=output_path,
            title="Backtrader集成测试报告"
        )
        print(f"✅ 报告已生成: {output_path}")

        # 10. 验证报告内容
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()

            checks = [
                ('生成时间', 'N/A' not in content.split('生成时间')[1].split('</p>')[0] if '生成时间' in content else False),
                ('无nan值', 'nan%' not in content),
                ('有净值数据', '核心表现指标' in content),
            ]

            print("\n验证结果:")
            for name, passed in checks:
                status = "✅" if passed else "❌"
                print(f"  {status} {name}")

            return all(passed for _, passed in checks)

    except Exception as e:
        print(f"❌ 生成报告失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_backtrader_integration()
    sys.exit(0 if success else 1)
