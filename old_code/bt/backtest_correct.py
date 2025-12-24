# bt/backtest_correct.py

"""
Backtrader 回测主程序 - 正确修复版本

修复内容：
1. 使用配置文件的时间范围裁剪所有数据
2. 保留所有数据源，让backtrader自动处理时间对齐
3. 优雅处理缺失值和停牌日（使用suspended标记）
4. 减少过于严格的数据过滤

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
    运行回测主程序 - 正确修复版本
    
    参数:
        config_path: 策略配置文件路径
    
    返回:
        (cerebro, results) 元组
    """
    # print("=" * 60)
    # print("🚀 Backtrader MVP 回测系统启动 (正确修复版本)")
    # print("=" * 60)
    
    # ========================================
    # 1. 读取配置
    # ========================================
    # print("\n[1/6] 📖 读取配置...")
    
    with open(config_path, encoding='utf-8') as f:
        strat_conf = yaml.safe_load(f)
    
    with open('config/broker.json', encoding='utf-8') as f:
        broker_conf = json.load(f)
    
    with open('config/trading_days.json', encoding='utf-8') as f:
        trading_days_config = json.load(f)
        trading_days = trading_days_config['dates']
    
    # 🔧 关键修复：从配置中读取回测时间范围
    backtest_config = strat_conf.get('backtest', {})
    start_date = backtest_config.get('start_date', '2024-01-01')
    end_date = backtest_config.get('end_date', '2024-12-31')
    
    # print(f"  策略名称: {strat_conf['strategy']['name']}")
    # print(f"  初始资金: {broker_conf['initial_cash']:,}")
    # print(f"  佣金率: {broker_conf['commission']['rate']:.4f}")
    # print(f"  调仓日数量: {len(trading_days)}")
    # print(f"  选股数量: {strat_conf['pipeline']['selector']['params']['n']}")
    # print(f"  回测时间范围: {start_date} 到 {end_date}")
    
    # 获取止损阈值
    stop_loss_threshold = strat_conf['strategy'].get('stop_loss', -0.10)
    # print(f"  止损阈值: {stop_loss_threshold:.0%}")
    
    # ========================================
    # 2. 初始化 Cerebro 引擎
    # ========================================
    # print("\n[2/6] ⚙️ 初始化回测引擎...")
    
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(broker_conf['initial_cash'])
    cerebro.broker.setcommission(commission=broker_conf['commission']['rate'])
    
    # print(f"  ✅ Cerebro 引擎已初始化")
    
    # ========================================
    # 3. 加载数据 - 关键修复：使用时间范围裁剪
    # ========================================
    # print("\n[3/6] 📊 加载数据（时间范围裁剪）...")
    
    data_dir = './bt/data_export/'
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"数据目录不存在: {data_dir}\n"
            f"请先运行数据导出流程。"
        )
    
    stock_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    if not stock_files:
        raise FileNotFoundError(f"数据目录为空: {data_dir}")
    
    # print(f"  发现 {len(stock_files)} 个股票数据文件")
    
    # 🔧 关键修复：转换时间范围
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    loaded_count = 0
    total_data_points = 0
    data_stats = {
        'loaded': 0,
        'time_clipped': 0,
        'empty_after_clip': 0,
        'total_files': len(stock_files)
    }
    
    # 🔧 关键修复：统计在时间范围内的调仓日
    valid_trading_days = [
        d for d in trading_days 
        if start_dt <= pd.to_datetime(d) <= end_dt
    ]
    
    for s_file in stock_files:
        try:
            file_path = os.path.join(data_dir, s_file)
            df = pd.read_parquet(file_path)
            
            # 确保索引是 DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            file_start = df.index.min()
            file_end = df.index.max()
            original_rows = len(df)
            
            # 🔧 关键修复：检查是否需要时间裁剪
            needs_clipping = file_start < start_dt or file_end > end_dt
            
            if needs_clipping:
                data_stats['time_clipped'] += 1
                
                # 裁剪数据到回测时间范围
                df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                
                if df.empty:
                    data_stats['empty_after_clip'] += 1
                    # print(f"  ⚠️ 跳过 {s_file}: 裁剪后无数据 ({original_rows} -> 0 行)")
                    continue
            
            if df.empty:
                data_stats['empty_after_clip'] += 1
                # print(f"  ⚠️ 跳过 {s_file}: 数据为空")
                continue
            
            ticker = s_file.replace('.parquet', '')
            data = FactorPandasData(dataname=df, name=ticker)
            cerebro.adddata(data)
            loaded_count += 1
            total_data_points += len(df)
            
            # 🔧 显示处理结果
            # if needs_clipping:
            #     print(f"  ✅ 加载 {ticker}: {len(df)} 天数据 (裁剪后: {file_start.date()} - {file_end.date()} -> {df.index.min().date()} - {df.index.max().date()})")
            # else:
            #     print(f"  ✅ 加载 {ticker}: {len(df)} 天数据 ({file_start.date()} - {file_end.date()})")
            
        except Exception as e:
            # print(f"  ❌ 加载 {s_file} 失败: {e}")
            pass
    
    # 数据加载统计
    # print(f"\n  📊 数据加载统计:")
    # print(f"    总文件数: {data_stats['total_files']}")
    # print(f"    成功加载: {loaded_count}")
    # print(f"    时间裁剪: {data_stats['time_clipped']}")
    # print(f"    裁剪后为空: {data_stats['empty_after_clip']}")
    # print(f"    总数据点: {total_data_points}")
    # print(f"    平均每股票: {total_data_points/loaded_count:.0f} 天")
    
    # print(f"\n  📅 调仓日过滤:")
    # print(f"    原始调仓日: {len(trading_days)} 个")
    # print(f"    有效调仓日: {len(valid_trading_days)} 个")
    # if valid_trading_days:
    #     print(f"    第一个调仓日: {valid_trading_days[0]}")
    #     print(f"    最后一个调仓日: {valid_trading_days[-1]}")
    
    if loaded_count == 0:
        raise ValueError("没有成功加载任何数据")
    
    # ========================================
    # 4. 组装策略组件
    # ========================================
    # print("\n[4/6] 🔧 组装策略组件...")
    
    # Pipeline 组件
    selector = TopNSelector(top_n=strat_conf['pipeline']['selector']['params']['n'])
    allocator = EqualWeightAllocator()
    capital_manager = FullPositionManager(utilization_ratio=0.95)
    
    # 🔧 触发器（使用过滤后的调仓日）
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
    
    # print(f"  ✅ 选股器: TopNSelector (N={strat_conf['pipeline']['selector']['params']['n']})")
    # print(f"  ✅ 分配器: EqualWeightAllocator")
    # print(f"  ✅ 资金管理: FullPositionManager (95%)")
    # print(f"  ✅ 触发器: StopLoss, RebalanceDay")
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # ========================================
    # 5. 运行回测
    # ========================================
    # print("\n[5/6] 🏃 运行回测...")
    # print("  (这可能需要一些时间...)")
    
    # 验证数据时间范围
    # if cerebro.datas:
    #     first_data = cerebro.datas[0]
    #     if len(first_data) > 0:
    #         data_start = first_data.datetime.date(1-len(first_data))
    #         data_end = first_data.datetime.date(0)
    #         print(f"  📅 实际回测时间范围: {data_start} - {data_end}")
    #         print(f"  📅 配置回测范围: {start_date} - {end_date}")
    #         print(f"  📊 回测天数: {len(first_data)} 天")
    #
    #         # 计算调仓日覆盖度
    #         covered_rebalance_days = sum(1 for d in valid_trading_days
    #                                    if data_start <= pd.to_datetime(d).date() <= data_end)
    #         print(f"  📅 调仓日覆盖度: {covered_rebalance_days}/{len(valid_trading_days)} ({covered_rebalance_days/len(valid_trading_days)*100:.1f}%)")
    
    results = cerebro.run()
    strat = results[0]
    
    # print("  ✅ 回测完成")
    
    # ========================================
    # 6. 生成报告
    # ========================================
    # print("\n[6/6] 📝 生成报告...")
    
    # 提取分析器结果
    analyzers = {
        'sharpe': strat.analyzers.sharpe.get_analysis(),
        'drawdown': strat.analyzers.drawdown.get_analysis(),
        'trades': strat.analyzers.trades.get_analysis()
    }
    
    # 生成 HTML 报告
    reporter = ReportGenerator()
    report_path = reporter.generate(cerebro, strat, analyzers)
    
    print(f"  ✅ 报告已生成: {report_path}")
    
    # print("\n" + "=" * 60)
    print("✅ 回测流程完成！")
    # print("=" * 60)
    
    return cerebro, results


if __name__ == '__main__':
    run_backtest()