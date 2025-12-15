# bt/backtest_final.py

"""
Backtrader 回测主程序 - 最终修复版本

修复内容：
1. 数据质量过滤：只加载有足够数据的股票
2. 统一时间范围：确保所有股票数据时间范围一致
3. 调仓日过滤：移除在数据时间范围之外的调仓日
4. 详细调试：增加完整的数据质量检查

作者：Roo
日期：2025-12-15
"""

import os
import sys
import yaml
import json
import logging
import pandas as pd
import backtrader as bt
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bt.data.feeds import FactorPandasData
from bt.core.base_strategy import BacktestStrategy
from bt.core.constants import ActionPriority, ActionType
from bt.triggers.stop_loss import StopLossTrigger
from bt.triggers.rebalance_day import RebalanceDayTrigger
from bt.pipeline.selectors import TopNSelector
from bt.pipeline.allocators import EqualWeightAllocator
from bt.pipeline.capital import FullPositionManager
from bt.utils.report_generator import ReportGenerator


def run_backtest(config_path: str = 'config/strategy_main.yaml') -> tuple:
    """
    运行回测主程序 - 最终修复版本
    
    参数:
        config_path: 策略配置文件路径
    
    返回:
        (cerebro, results) 元组
    """
    print("=" * 60)
    print("🚀 Backtrader MVP 回测系统启动 (最终修复版本)")
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
        trading_days_config = json.load(f)
        trading_days = trading_days_config['dates']
    
    # 从配置中读取回测时间范围
    backtest_config = strat_conf.get('backtest', {})
    start_date = backtest_config.get('start_date', '2024-01-01')
    end_date = backtest_config.get('end_date', '2024-12-31')
    
    print(f"  策略名称: {strat_conf['strategy']['name']}")
    print(f"  初始资金: {broker_conf['initial_cash']:,}")
    print(f"  佣金率: {broker_conf['commission']['rate']:.4f}")
    print(f"  调仓日数量: {len(trading_days)}")
    print(f"  选股数量: {strat_conf['pipeline']['selector']['params']['n']}")
    print(f"  回测时间范围: {start_date} 到 {end_date}")
    
    # 获取止损阈值
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
    # 3. 加载数据 - 最终修复：严格的数据质量过滤
    # ========================================
    print("\n[3/6] 📊 加载数据（数据质量过滤）...")
    
    data_dir = './bt/data_export/'
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"数据目录不存在: {data_dir}\n"
            f"请先运行数据导出流程。"
        )
    
    stock_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    if not stock_files:
        raise FileNotFoundError(f"数据目录为空: {data_dir}")
    
    print(f"  发现 {len(stock_files)} 个股票数据文件")
    
    # 转换时间范围
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # 🔧 关键修复：数据质量过滤
    min_data_days = 180  # 最少需要180天数据（约9个月）
    max_data_days = 400  # 最多不超过400天
    
    loaded_count = 0
    filtered_count = 0
    total_data_points = 0
    data_quality_stats = {'excellent': 0, 'good': 0, 'poor': 0, 'invalid': 0}
    
    valid_trading_days = []  # 在数据时间范围内的调仓日
    
    for s_file in stock_files:
        try:
            file_path = os.path.join(data_dir, s_file)
            df = pd.read_parquet(file_path)
            
            # 确保索引是 DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            file_start = df.index.min()
            file_end = df.index.max()
            file_days = len(df)
            
            # 数据质量评估
            if file_days < 30:  # 少于30天数据
                data_quality_stats['invalid'] += 1
                print(f"  ❌ 跳过 {s_file}: 数据天数太少 ({file_days}天)")
                filtered_count += 1
                continue
            
            if file_days >= min_data_days and file_days <= max_data_days:
                if file_days >= 300:
                    data_quality_stats['excellent'] += 1
                else:
                    data_quality_stats['good'] += 1
            else:
                data_quality_stats['poor'] += 1
                print(f"  ⚠️ 跳过 {s_file}: 数据天数异常 ({file_days}天)")
                filtered_count += 1
                continue
            
            # 时间范围过滤
            if file_end < start_dt or file_start > end_dt:
                print(f"  ⚠️ 跳过 {s_file}: 时间范围 {file_start.date()} - {file_end.date()} 不在回测范围内")
                filtered_count += 1
                continue
            
            # 裁剪数据到回测时间范围
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            if df.empty:
                print(f"  ⚠️ 跳过 {s_file}: 裁剪后无数据")
                filtered_count += 1
                continue
            
            ticker = s_file.replace('.parquet', '')
            data = FactorPandasData(dataname=df, name=ticker)
            cerebro.adddata(data)
            loaded_count += 1
            total_data_points += len(df)
            
            # 更新调仓日列表（第一次加载时）
            if not valid_trading_days:
                valid_trading_days = [
                    d for d in trading_days 
                    if pd.to_datetime(d) >= file_start and pd.to_datetime(d) <= file_end
                ]
            
            # 显示详细信息
            print(f"  ✅ 加载 {ticker}: {len(df)} 天数据 ({df.index.min().date()} - {df.index.max().date()})")
            
        except Exception as e:
            print(f"  ❌ 加载 {s_file} 失败: {e}")
            filtered_count += 1
    
    # 数据质量报告
    print(f"\n  📊 数据质量统计:")
    print(f"    总文件数: {len(stock_files)}")
    print(f"    成功加载: {loaded_count}")
    print(f"    过滤跳过: {filtered_count}")
    print(f"    优秀数据(≥300天): {data_quality_stats['excellent']}")
    print(f"    良好数据(180-299天): {data_quality_stats['good']}")
    print(f"    异常数据: {data_quality_stats['poor']}")
    print(f"    无效数据(<30天): {data_quality_stats['invalid']}")
    print(f"    总数据点: {total_data_points}")
    
    if loaded_count == 0:
        raise ValueError("没有成功加载任何高质量数据，请检查数据文件和过滤条件")
    
    print(f"\n  📅 调仓日过滤:")
    print(f"    原始调仓日: {len(trading_days)} 个")
    print(f"    有效调仓日: {len(valid_trading_days)} 个")
    if valid_trading_days:
        print(f"    第一个调仓日: {valid_trading_days[0]}")
        print(f"    最后一个调仓日: {valid_trading_days[-1]}")
    
    # ========================================
    # 4. 组装策略组件
    # ========================================
    print("\n[4/6] 🔧 组装策略组件...")
    
    # Pipeline 组件
    selector = TopNSelector(top_n=strat_conf['pipeline']['selector']['params']['n'])
    allocator = EqualWeightAllocator()
    capital_manager = FullPositionManager(utilization_ratio=0.95)
    
    # 触发器（使用过滤后的调仓日）
    triggers = [
        lambda s: StopLossTrigger(s, loss_threshold=stop_loss_threshold),
        lambda s: RebalanceDayTrigger(s, trading_days_list=valid_trading_days)
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
    
    # 验证数据时间范围
    if cerebro.datas:
        first_data = cerebro.datas[0]
        if len(first_data) > 0:
            data_start = first_data.datetime.date(1-len(first_data))
            data_end = first_data.datetime.date(0)
            print(f"  📅 实际数据时间范围: {data_start} - {data_end}")
            print(f"  📅 预期回测范围: {start_date} - {end_date}")
            print(f"  📊 回测天数: {len(first_data)} 天")
    
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
    
    # 交易统计
    trades = analyzers['trades']
    total_trades = trades.get('total', {}).get('total', 0)
    print(f"  总交易次数: {total_trades}")
    
    sharpe = analyzers['sharpe'].get('sharperatio')
    if sharpe is not None:
        print(f"  Sharpe Ratio: {sharpe:>11.3f}")
    
    max_dd = analyzers['drawdown'].get('max', {}).get('drawdown')
    if max_dd is not None:
        print(f"  最大回撤: {max_dd:>13.2f}%")
    
    # 关键信息总结
    print(f"\n🔍 调试信息:")
    print(f"  数据源数量: {len(cerebro.datas)}")
    print(f"  回测时间范围: {start_date} - {end_date}")
    print(f"  有效调仓日数量: {len(valid_trading_days)}")
    print(f"  止损阈值: {stop_loss_threshold:.1%}")
    
    # 生成 HTML 报告
    reporter = ReportGenerator()
    report_path = reporter.generate(cerebro, strat, analyzers)
    
    print(f"  ✅ 报告已生成: {report_path}")
    
    print("\n" + "=" * 60)
    print("✅ 回测流程完成！")
    print("=" * 60)
    
    return cerebro, results


if __name__ == '__main__':
    run_backtest()