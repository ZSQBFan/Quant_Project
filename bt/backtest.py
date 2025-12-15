# bt/backtest.py

"""
Backtrader 回测主程序

这是整个回测系统的入口点，负责：
1. 读取配置
2. 初始化 Cerebro 引擎
3. 加载数据
4. 组装策略组件
5. 运行回测
6. 生成报告
"""

import os
import sys
import yaml
import json
import logging
import pandas as pd
import backtrader as bt

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bt.data.feeds import FactorPandasData
from bt.core.base_strategy import BacktestStrategy
from bt.triggers.stop_loss import StopLossTrigger
from bt.triggers.rebalance_day import RebalanceDayTrigger
from bt.pipeline.selectors import TopNSelector
from bt.pipeline.allocators import EqualWeightAllocator
from bt.pipeline.capital import FullPositionManager
from bt.utils.report_generator import ReportGenerator


def run_backtest(config_path: str = 'config/strategy_main.yaml') -> tuple:
    """
    运行回测主程序
    
    参数:
        config_path: 策略配置文件路径
    
    返回:
        (cerebro, results) 元组
    """
    print("=" * 60)
    print("🚀 Backtrader MVP 回测系统启动")
    print("=" * 60)
    
    # ========================================
    # 1. 读取配置
    # ========================================
    print("\n[1/6] 📖 读取配置...")
    
    with open(config_path, encoding='utf-8') as f:
        strat_conf = yaml.safe_load(f)
    
    with open('config/broker.json', encoding='utf-8') as f:
        broker_conf = json.load(f)
    
    with open('config/trading_days.json', encoding='utf-8') as f:
        trading_days = json.load(f)['dates']
    
    print(f"  策略名称: {strat_conf['strategy']['name']}")
    print(f"  初始资金: {broker_conf['initial_cash']:,}")
    print(f"  佣金率: {broker_conf['commission']['rate']:.4f}")
    print(f"  调仓日数量: {len(trading_days)}")
    print(f"  选股数量: {strat_conf['pipeline']['selector']['params']['n']}")
    
    # 获取止损阈值（从配置读取，如果没有则使用默认值）
    stop_loss_threshold = strat_conf['strategy'].get('stop_loss', -0.10)
    print(f"  止损阈值: {stop_loss_threshold:.0%}")
    
    # ========================================
    # 2. 初始化 Cerebro 引擎
    # ========================================
    print("\n[2/6] ⚙️ 初始化回测引擎...")
    
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(broker_conf['initial_cash'])
    cerebro.broker.setcommission(commission=broker_conf['commission']['rate'])
    
    print(f"  ✅ Cerebro 引擎已初始化")
    
    # ========================================
    # 3. 加载数据
    # ========================================
    print("\n[3/6] 📊 加载数据...")
    
    data_dir = './bt/data_export/'
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"数据目录不存在: {data_dir}\n"
            f"请先运行数据导出流程。"
        )
    
    stock_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    if not stock_files:
        raise FileNotFoundError(f"数据目录为空: {data_dir}")
    
    print(f"  找到 {len(stock_files)} 个股票数据文件")
    
    loaded_count = 0
    for s_file in stock_files:
        try:
            df = pd.read_parquet(os.path.join(data_dir, s_file))
            
            # 确保索引是 DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            ticker = s_file.replace('.parquet', '')
            data = FactorPandasData(dataname=df, name=ticker)
            cerebro.adddata(data)
            loaded_count += 1
            
        except Exception as e:
            logging.warning(f"  ⚠️ 加载 {s_file} 失败: {e}")
    
    print(f"  ✅ 成功加载 {loaded_count} 只股票")
    
    # ========================================
    # 4. 组装策略组件
    # ========================================
    print("\n[4/6] 🔧 组装策略组件...")
    
    # Pipeline 组件
    selector = TopNSelector(top_n=strat_conf['pipeline']['selector']['params']['n'])
    allocator = EqualWeightAllocator()
    capital_manager = FullPositionManager(utilization_ratio=0.95)
    
    # 触发器工厂函数（延迟实例化）
    triggers = [
        lambda s: StopLossTrigger(s, loss_threshold=stop_loss_threshold),
        lambda s: RebalanceDayTrigger(s, trading_days_list=trading_days)
    ]
    
    # 添加策略
    cerebro.addstrategy(
        BacktestStrategy,
        selector=selector,
        allocator=allocator,
        capital_manager=capital_manager,
        triggers=triggers
    )
    
    print(f"  ✅ 选股器: TopNSelector (N={strat_conf['pipeline']['selector']['params']['n']})")
    print(f"  ✅ 分配器: EqualWeightAllocator")
    print(f"  ✅ 资金管理: FullPositionManager (95%)")
    print(f"  ✅ 触发器: StopLoss, RebalanceDay")
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # ========================================
    # 5. 运行回测
    # ========================================
    print("\n[5/6] 🏃 运行回测...")
    print("  (这可能需要一些时间...)")
    
    results = cerebro.run()
    strat = results[0]
    
    print("  ✅ 回测完成")
    
    # ========================================
    # 6. 生成报告
    # ========================================
    print("\n[6/6] 📝 生成报告...")
    
    # 提取分析器结果
    analyzers = {
        'sharpe': strat.analyzers.sharpe.get_analysis(),
        'drawdown': strat.analyzers.drawdown.get_analysis(),
        'trades': strat.analyzers.trades.get_analysis()
    }
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("📊 回测结果摘要")
    print("=" * 60)
    
    initial_cash = broker_conf['initial_cash']
    final_value = cerebro.broker.getvalue()
    total_return = (final_value / initial_cash - 1) * 100
    
    print(f"  初始资金: {initial_cash:>15,.2f}")
    print(f"  最终净值: {final_value:>15,.2f}")
    print(f"  总收益率: {total_return:>14.2f}%")
    
    sharpe = analyzers['sharpe'].get('sharperatio')
    if sharpe is not None:
        print(f"  Sharpe Ratio: {sharpe:>11.3f}")
    
    max_dd = analyzers['drawdown'].get('max', {}).get('drawdown')
    if max_dd is not None:
        print(f"  最大回撤: {max_dd:>13.2f}%")
    
    # 生成 HTML 报告
    reporter = ReportGenerator()
    report_path = reporter.generate(cerebro, strat, analyzers)
    
    print("\n" + "=" * 60)
    print("✅ 回测流程完成！")
    print("=" * 60)
    
    return cerebro, results


if __name__ == '__main__':
    run_backtest()