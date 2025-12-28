"""
换手率分析器测试

测试验证：
1. 分析器能正确追踪买入和卖出金额
2. 换手率计算公式正确
3. 年化换手率计算正确
4. 输出清晰的测试结果供人工验证
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import backtrader as bt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.analyzers.turnover import TurnoverAnalyzer


class SimpleTradingStrategy(bt.Strategy):
    """
    简单交易策略 - 用于测试换手率分析器
    
    策略规则：
    - 第1天：买入 1000 股
    - 第5天：卖出 500 股
    - 第10天：买入 500 股
    - 第15天：全部卖出
    """
    
    params = (
        ('buy_size', 1000),
        ('sell_ratio', 0.5),
    )
    
    def __init__(self):
        """初始化策略"""
        self.order_count = 0
        self.trade_log = []
        
    def next(self):
        """每周期执行"""
        current_date = self.datetime.date(0)
        current_bar = len(self)
        
        # 订单执行日志
        for order in self.broker.orders:
            if order.status == order.Completed:
                if order.isbuy():
                    self.trade_log.append({
                        'date': current_date,
                        'action': 'BUY',
                        'size': order.executed.size,
                        'price': order.executed.price,
                        'value': order.executed.value,
                        'commission': order.executed.comm
                    })
                else:
                    self.trade_log.append({
                        'date': current_date,
                        'action': 'SELL',
                        'size': order.executed.size,
                        'price': order.executed.price,
                        'value': order.executed.value,
                        'commission': order.executed.comm
                    })
        
        # 交易规则
        if current_bar == 1:  # 第1天
            # 初始买入
            self.order_count += 1
            self.buy(size=self.params.buy_size)
            
        elif current_bar == 5:  # 第5天
            # 卖出部分持仓
            current_size = self.getposition().size
            sell_size = int(current_size * self.params.sell_ratio)
            if sell_size > 0:
                self.order_count += 1
                self.sell(size=sell_size)
                
        elif current_bar == 10:  # 第10天
            # 再次买入
            self.order_count += 1
            self.buy(size=500)
            
        elif current_bar == 15:  # 第15天
            # 全部清仓
            current_size = self.getposition().size
            if current_size > 0:
                self.order_count += 1
                self.sell(size=current_size)


def create_test_data(days=20, initial_price=100.0):
    """
    创建测试数据
    
    Args:
        days: 数据天数
        initial_price: 初始价格
    
    Returns:
        DataFrame: 测试数据
    """
    start_date = datetime(2024, 1, 1)
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    
    np.random.seed(42)
    
    # 生成价格数据（轻微上涨趋势）
    price_changes = np.random.normal(0.1, 1.0, days)  # 每天平均上涨0.1%
    prices = initial_price * np.cumprod(1 + price_changes / 100)
    
    data = {
        'open': prices * (1 + np.random.normal(0, 0.005, days)),
        'high': prices * (1 + np.random.normal(0.01, 0.005, days)),
        'low': prices * (1 + np.random.normal(-0.01, 0.005, days)),
        'close': prices,
        'volume': np.random.randint(100000, 500000, days),
        'openinterest': np.zeros(days)
    }
    
    df = pd.DataFrame(data, index=dates)
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
    
    return df


def test_turnover_analyzer():
    """
    测试换手率分析器
    """
    print("=" * 70)
    print("🧪 换手率分析器测试")
    print("=" * 70)
    
    all_passed = True
    
    # 创建测试数据
    print("\n📊 创建测试数据...")
    test_df = create_test_data(days=20, initial_price=100.0)
    print(f"   数据天数: {len(test_df)}")
    print(f"   价格范围: {test_df['close'].min():.2f} - {test_df['close'].max():.2f}")
    
    # 创建 Cerebro 引擎
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)  # 初始资金10万
    
    # 添加数据
    data = bt.feeds.PandasData(dataname=test_df, name='TEST_STOCK')
    cerebro.adddata(data)
    
    # 添加策略
    cerebro.addstrategy(SimpleTradingStrategy)
    
    # 添加换手率分析器
    cerebro.addanalyzer(TurnoverAnalyzer, _name='turnover')
    
    # 运行回测
    print("\n🚀 运行回测...")
    initial_value = cerebro.broker.getvalue()
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    
    # 获取分析结果
    strat = results[0]
    turnover_analysis = strat.analyzers.turnover.get_analysis()
    
    # ==================== 测试结果验证 ====================
    print("\n" + "=" * 70)
    print("📈 回测结果")
    print("=" * 70)
    print(f"   初始资金: {initial_value:,.2f}")
    print(f"   最终资金: {final_value:,.2f}")
    print(f"   总收益: {final_value - initial_value:,.2f} ({(final_value/initial_value - 1)*100:.2f}%)")
    
    # ==================== 测试用例 1: 验证返回数据结构 ====================
    print("\n" + "-" * 70)
    print("🧪 测试用例 1: 验证返回数据结构")
    print("-" * 70)
    
    required_keys = ['total_turnover', 'avg_turnover', 'annual_turnover', 
                     'turnover_list', 'period_details', 'total_buy', 'total_sell']
    
    keys_check = all(key in turnover_analysis for key in required_keys)
    print(f"   必要键检查: {'✅ 通过' if keys_check else '❌ 失败'}")
    
    if keys_check:
        print(f"   total_turnover: {turnover_analysis['total_turnover']:.6f}")
        print(f"   avg_turnover: {turnover_analysis['avg_turnover']:.6f}")
        print(f"   annual_turnover: {turnover_analysis['annual_turnover']:.6f}")
        print(f"   turnover_list 长度: {len(turnover_analysis['turnover_list'])}")
        print(f"   period_details 长度: {len(turnover_analysis['period_details'])}")
        print(f"   total_buy: {turnover_analysis['total_buy']:,.2f}")
        print(f"   total_sell: {turnover_analysis['total_sell']:,.2f}")
    else:
        all_passed = False
        missing_keys = [key for key in required_keys if key not in turnover_analysis]
        print(f"   缺失的键: {missing_keys}")
    
    # ==================== 测试用例 2: 验证换手率范围 ====================
    print("\n" + "-" * 70)
    print("🧪 测试用例 2: 验证换手率范围 (0 ~ 1)")
    print("-" * 70)
    
    turnover_list = turnover_analysis['turnover_list']
    valid_range = all(0.0 <= t <= 1.0 for t in turnover_list)
    print(f"   换手率范围检查: {'✅ 通过' if valid_range else '❌ 失败'}")
    
    if not valid_range:
        invalid_values = [t for t in turnover_list if t < 0.0 or t > 1.0]
        print(f"   异常值: {invalid_values}")
        all_passed = False
    
    if turnover_list:
        print(f"   最大换手率: {max(turnover_list):.6f}")
        print(f"   最小换手率: {min(turnover_list):.6f}")
    
    # ==================== 测试用例 3: 验证买入/卖出金额追踪 ====================
    print("\n" + "-" * 70)
    print("🧪 测试用例 3: 验证买入/卖出金额追踪")
    print("-" * 70)
    
    total_buy = turnover_analysis['total_buy']
    total_sell = turnover_analysis['total_sell']
    
    print(f"   总买入金额: {total_buy:,.2f}")
    print(f"   总卖出金额: {total_sell:,.2f}")
    
    # 验证有交易发生
    has_trades = total_buy > 0 or total_sell > 0
    print(f"   交易发生检查: {'✅ 有交易' if has_trades else '❌ 无交易'}")
    
    if not has_trades:
        all_passed = False
    
    # ==================== 测试用例 4: 验证换手率计算公式 ====================
    print("\n" + "-" * 70)
    print("🧪 测试用例 4: 验证换手率计算公式")
    print("   公式: period_turnover = (buy_value + sell_value) / (2 × portfolio_value)")
    print("-" * 70)
    
    # 手动计算验证
    period_details = turnover_analysis['period_details']
    manual_calc_passed = True
    
    for detail in period_details:
        if detail['portfolio_value'] > 0:
            expected_turnover = (detail['buy_value'] + detail['sell_value']) / (2 * detail['portfolio_value'])
            actual_turnover = detail['turnover']
            
            if abs(expected_turnover - actual_turnover) > 0.0001:
                print(f"   ❌ 日期 {detail['date'].strftime('%Y-%m-%d')}: "
                      f"期望={expected_turnover:.6f}, 实际={actual_turnover:.6f}")
                manual_calc_passed = False
    
    if manual_calc_passed:
        print(f"   公式验证: ✅ 通过")
    else:
        all_passed = False
    
    # ==================== 测试用例 5: 验证年化换手率计算 ====================
    print("\n" + "-" * 70)
    print("🧪 测试用例 5: 验证年化换手率计算")
    print("   公式: annual_turnover = avg_turnover × 252")
    print("-" * 70)
    
    avg_turnover = turnover_analysis['avg_turnover']
    annual_turnover = turnover_analysis['annual_turnover']
    
    expected_annual = avg_turnover * 252
    annual_calc_passed = abs(annual_turnover - expected_annual) < 0.0001
    
    print(f"   平均换手率: {avg_turnover:.6f}")
    print(f"   期望年化换手率: {expected_annual:.6f}")
    print(f"   实际年化换手率: {annual_turnover:.6f}")
    print(f"   年化公式验证: {'✅ 通过' if annual_calc_passed else '❌ 失败'}")
    
    if not annual_calc_passed:
        all_passed = False
    
    # ==================== 测试用例 6: 验证总换手率 ====================
    print("\n" + "-" * 70)
    print("🧪 测试用例 6: 验证总换手率 (所有期换手率之和)")
    print("-" * 70)
    
    turnover_list = turnover_analysis['turnover_list']
    expected_total = sum(turnover_list)
    actual_total = turnover_analysis['total_turnover']
    
    total_calc_passed = abs(expected_total - actual_total) < 0.0001
    
    print(f"   turnover_list 总和: {expected_total:.6f}")
    print(f"   total_turnover: {actual_total:.6f}")
    print(f"   总换手率验证: {'✅ 通过' if total_calc_passed else '❌ 失败'}")
    
    if not total_calc_passed:
        all_passed = False
    
    # ==================== 详细换手率列表 ====================
    print("\n" + "=" * 70)
    print("📊 每期换手率详情")
    print("=" * 70)
    print(f"{'日期':<12} {'换手率':<12} {'买入金额':<12} {'卖出金额':<12} {'组合价值':<15}")
    print("-" * 70)
    
    for detail in turnover_analysis['period_details']:
        print(f"{detail['date'].strftime('%Y-%m-%d'):<12} "
              f"{detail['turnover']:<12.6f} "
              f"{detail['buy_value']:<12,.2f} "
              f"{detail['sell_value']:<12,.2f} "
              f"{detail['portfolio_value']:<15,.2f}")
    
    # ==================== 交易日志 ====================
    print("\n" + "=" * 70)
    print("📝 交易执行日志")
    print("=" * 70)
    
    if strat.trade_log:
        print(f"{'日期':<12} {'操作':<6} {'数量':<8} {'价格':<10} {'金额':<15} {'佣金':<10}")
        print("-" * 70)
        
        for trade in strat.trade_log:
            print(f"{trade['date'].strftime('%Y-%m-%d'):<12} "
                  f"{trade['action']:<6} "
                  f"{trade['size']:<8} "
                  f"{trade['price']:<10.2f} "
                  f"{trade['value']:<15,.2f} "
                  f"{trade['commission']:<10,.2f}")
    else:
        print("   无交易记录")
    
    # ==================== 测试总结 ====================
    print("\n" + "=" * 70)
    print("🏁 测试总结")
    print("=" * 70)
    
    print(f"\n📈 换手率指标:")
    print(f"   - 平均单期换手率: {turnover_analysis['avg_turnover']:.6f} ({turnover_analysis['avg_turnover']*100:.4f}%)")
    print(f"   - 年化换手率: {turnover_analysis['annual_turnover']:.6f} ({turnover_analysis['annual_turnover']*100:.2f}%)")
    print(f"   - 总换手率: {turnover_analysis['total_turnover']:.6f} ({turnover_analysis['total_turnover']*100:.2f}%)")
    print(f"   - 交易天数: {len(turnover_analysis['turnover_list'])} 天")
    
    print(f"\n💰 交易统计:")
    print(f"   - 总买入金额: {turnover_analysis['total_buy']:,.2f}")
    print(f"   - 总卖出金额: {turnover_analysis['total_sell']:,.2f}")
    print(f"   - 总交易次数: {len(strat.trade_log)} 次")
    
    if all_passed:
        print(f"\n✅ 所有测试用例通过！")
        return True
    else:
        print(f"\n❌ 部分测试用例失败，请检查上述输出")
        return False


def test_edge_cases():
    """
    测试边界情况
    """
    print("\n" + "=" * 70)
    print("🧪 边界情况测试")
    print("=" * 70)
    
    all_passed = True
    
    # ==================== 测试用例: 空数据 ====================
    print("\n📊 测试用例: 无交易情况")
    print("-" * 50)
    
    class NoTradeStrategy(bt.Strategy):
        """无交易策略"""
        def next(self):
            pass  # 不进行任何交易
    
    # 创建测试数据
    test_df = create_test_data(days=10)
    
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    
    data = bt.feeds.PandasData(dataname=test_df, name='TEST_STOCK')
    cerebro.adddata(data)
    
    cerebro.addstrategy(NoTradeStrategy)
    cerebro.addanalyzer(TurnoverAnalyzer, _name='turnover')
    
    results = cerebro.run()
    strat = results[0]
    turnover_analysis = strat.analyzers.turnover.get_analysis()
    
    # 验证空交易情况
    if turnover_analysis['avg_turnover'] == 0.0 and turnover_analysis['total_turnover'] == 0.0:
        print("   ✅ 无交易时换手率为0")
    else:
        print("   ❌ 无交易时换手率应该为0")
        all_passed = False
    
    return all_passed


def main():
    """
    主测试函数
    """
    print("\n" + "🎯" + "=" * 68 + "🎯")
    print("        换手率分析器 TurnoverAnalyzer 单元测试")
    print("🎯" + "=" * 68 + "🎯\n")
    
    # 运行主要测试
    main_result = test_turnover_analyzer()
    
    # 运行边界情况测试
    edge_result = test_edge_cases()
    
    # 最终总结
    print("\n" + "=" * 70)
    print("🏁 最终测试结果")
    print("=" * 70)
    
    if main_result and edge_result:
        print("\n✅ 所有测试通过！换手率分析器功能正常。")
        return True
    else:
        print("\n❌ 部分测试失败，请检查上述输出。")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
