import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import backtrader as bt
from backtest.reports.report_generator import ReportGenerator

def run_report_example():
    """
    展示如何使用新的报告系统生成报告的完整示例。
    """
    print("🚀 开始运行报告生成示例...")

    # 1. 准备模拟数据 (通常这些数据来自 Backtrader 回测结果)
    print("📊 准备模拟回测数据...")
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    
    # 模拟净值数据
    np.random.seed(42)
    daily_returns = np.random.normal(0.0005, 0.01, len(dates))
    nav_values = (1 + daily_returns).cumprod()
    
    # 模拟基准数据
    benchmark_returns = np.random.normal(0.0003, 0.008, len(dates))
    benchmark_values = (1 + benchmark_returns).cumprod()
    
    nav_df = pd.DataFrame({
        'value': nav_values,
        'benchmark': benchmark_values
    }, index=dates)

    # 模拟交易数据
    trades_df = pd.DataFrame({
        'datetime': [dates[10], dates[20], dates[30], dates[40]],
        'type': ['buy', 'sell', 'buy', 'sell'],
        'price': [nav_values[10], nav_values[20], nav_values[30], nav_values[40]],
        'pnl': [0, 500, 0, -200]
    })

    # 2. 构造报告数据字典
    # 报告生成器支持直接传入包含 'nav' 和 'trades' 的字典
    report_data = {
        'nav': nav_df,
        'trades': trades_df
    }

    # 3. 初始化报告生成器并生成报告
    print("📝 正在生成 HTML 报告...")
    output_dir = "output/test_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "example_report.html")
    
    generator = ReportGenerator()
    generator.generate(
        strategy_results=report_data,
        output_path=report_path,
        title="量化策略回测分析报告 (示例)"
    )

    print(f"✅ 报告已成功生成: {os.path.abspath(report_path)}")
    print("\n你可以直接在浏览器中打开该文件查看结果。")

if __name__ == "__main__":
    run_report_example()
