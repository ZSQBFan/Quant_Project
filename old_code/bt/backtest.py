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
import concurrent.futures
import pandas as pd
import backtrader as bt

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from old_code.bt.data.feeds import FactorPandasData
from old_code.bt.core.base_strategy import BacktestStrategy
from old_code.bt.triggers.stop_loss import StopLossTrigger
from old_code.bt.triggers.rebalance_day import RebalanceDayTrigger
from old_code.bt.pipeline.selectors import TopNSelector
from old_code.bt.pipeline.allocators import EqualWeightAllocator
from old_code.bt.pipeline.capital import FullPositionManager
from old_code.bt.utils.report_generator import ReportGenerator


def run_backtest(
    config_path: str = 'old_code/config/strategy_main.yaml',
    pbar=None,
    generate_report: bool = True,
) -> tuple:
    """
    运行回测主程序
    
    参数:
        config_path: 策略配置文件路径
        pbar: tqdm 进度条实例
    
    返回:
        (cerebro, results) 元组
    """
    # print("=" * 60)
    # print("🚀 Backtrader MVP 回测系统启动")
    # print("=" * 60)
    
    # ========================================
    # 1. 读取配置
    # ========================================
    # if pbar is None:
    #     print("\n[1/6] 📖 读取配置...")
    
    with open(config_path, encoding='utf-8') as f:
        strat_conf = yaml.safe_load(f)
    
    with open('old_code/config/broker.json', encoding='utf-8') as f:
        broker_conf = json.load(f)

    with open('old_code/config/trading_days.json', encoding='utf-8') as f:
        trading_days = json.load(f)['dates']
    total_steps = len(trading_days)
    
    # if pbar is None:
    #     print(f"  策略名称: {strat_conf['strategy']['name']}")
    #     print(f"  初始资金: {broker_conf['initial_cash']:,}")
    #     print(f"  佣金率: {broker_conf['commission']['rate']:.4f}")
    #     print(f"  调仓日数量: {len(trading_days)}")
    #     print(f"  选股数量: {strat_conf['pipeline']['selector']['params']['n']}")
    
    # 获取止损阈值（从配置读取，如果没有则使用默认值）
    stop_loss_threshold = strat_conf['strategy'].get('stop_loss', -0.10)
    # if pbar is None:
    #     print(f"  止损阈值: {stop_loss_threshold:.0%}")
    
    # ========================================
    # 2. 初始化 Cerebro 引擎
    # ========================================
    # if pbar is None:
    #     print("\n[2/6] ⚙️ 初始化回测引擎...")
    
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(broker_conf['initial_cash'])
    cerebro.broker.setcommission(commission=broker_conf['commission']['rate'])
    
    # 注入进度条到策略参数
    if pbar is not None:
        # 方案 B: 通过策略参数传递
        # 注意：这里需要确保 BacktestStrategy 能够接收 pbar 参数
        pass
    
    # if pbar is None:
    #     print(f"  ✅ Cerebro 引擎已初始化")
    
    # ========================================
    # 3. 加载数据
    # ========================================
    # if pbar is None:
    #     print("\n[3/6] 📊 加载数据...")
    
    data_dir = './bt/data_export/'
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"数据目录不存在: {data_dir}\n"
            f"请先运行数据导出流程。"
        )
    
    stock_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    if not stock_files:
        raise FileNotFoundError(f"数据目录为空: {data_dir}")
    
    # if pbar is None:
    #     print(f"  找到 {len(stock_files)} 个股票数据文件")
    
    # 使用 ThreadPoolExecutor 并行加载 Parquet 文件
    def load_single_stock(s_file: str) -> tuple:
        """加载单个股票文件，返回 (成功标志, 数据或None, ticker或错误)"""
        try:
            file_path = os.path.join(data_dir, s_file)
            df = pd.read_parquet(file_path)
            
            # 确保索引是 DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            ticker = s_file.replace('.parquet', '')
            return (True, df, ticker)
        except Exception as e:
            logging.warning(f"  ⚠️ 加载 {s_file} 失败: {e}")
            return (False, None, str(e))
    
    # 使用合适的 worker 数量
    max_workers = min(32, (os.cpu_count() or 4) * 2)
    
    # 并行加载所有文件
    loaded_count = 0
    loaded_data = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务并保持顺序
        futures = {executor.submit(load_single_stock, s_file): s_file for s_file in stock_files}
        
        # 按提交顺序收集结果
        results = [None] * len(stock_files)
        for i, s_file in enumerate(stock_files):
            future = list(futures.keys())[list(futures.values()).index(s_file)]
            results[i] = future.result()
    
    # 按顺序处理结果并添加到 cerebro
    for success, df, ticker_or_error in results:
        if success:
            data = FactorPandasData(dataname=df, name=ticker_or_error)
            cerebro.adddata(data)
            loaded_count += 1

    # Backtrader 在没有任何 data feed 时，cerebro.run() 可能返回空列表（results=[]），
    # 继续访问 results[0] 会触发 IndexError。
    # 这里提前给出更明确的错误，便于上层捕获并输出合理信息。
    if loaded_count == 0:
        raise RuntimeError(
            "Backtrader 未加载到任何可用数据（bt/data_export/*.parquet 均加载失败/为空/字段不符合）。"
            "可能原因：数据导出目录内容不匹配当前 FactorPandasData 字段要求，或导出的 parquet 全部损坏/空文件。"
        )
    
    # if pbar is None:
    #     print(f"  ✅ 成功加载 {loaded_count} 只股票")
    
    # ========================================
    # 4. 组装策略组件
    # ========================================
    # if pbar is None:
    #     print("\n[4/6] 🔧 组装策略组件...")
    
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
        triggers=triggers,
        pbar=pbar,  # 🚨 通过参数传递进度条
        total_steps=total_steps
    )
    
    # if pbar is None:
    #     print(f"  ✅ 选股器: TopNSelector (N={strat_conf['pipeline']['selector']['params']['n']})")
    #     print(f"  ✅ 分配器: EqualWeightAllocator")
    #     print(f"  ✅ 资金管理: FullPositionManager (95%)")
    #     print(f"  ✅ 触发器: StopLoss, RebalanceDay")
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # ========================================
    # 5. 运行回测
    # ========================================
    # if pbar is None:
    #     print("\n[5/6] 🏃 运行回测...")
    #     print("  (这可能需要一些时间...)")
    
    results = cerebro.run()
    if not results:
        raise RuntimeError(
            "Backtrader 未返回策略结果（results 为空）。"
            "可能原因：未添加策略/未加载任何数据/回测在内部被提前终止。"
        )

    strat = results[0]
    
    # if pbar is None:
    #     print("  ✅ 回测完成")
    
    # ========================================
    # 6. 生成报告（可选）
    # ========================================
    # 报告与结束阶段提示由上层脚本统一控制台输出，避免重复。
    if generate_report:
        # 提取分析器结果
        analyzers = {
            'sharpe': strat.analyzers.sharpe.get_analysis(),
            'drawdown': strat.analyzers.drawdown.get_analysis(),
            'trades': strat.analyzers.trades.get_analysis()
        }

        # 生成 HTML 报告（ReportGenerator 内部会输出一次“✅ 报告已生成: ...”）
        reporter = ReportGenerator()
        _ = reporter.generate(cerebro, strat, analyzers)
    
    return cerebro, results


if __name__ == '__main__':
    run_backtest()
