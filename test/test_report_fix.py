"""
测试修复后的报告生成器
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from backtest.reports.report_generator import ReportGenerator
from datetime import datetime, timedelta

def generate_test_report():
    """生成测试报告"""

    # 创建模拟的净值数据
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日

    # 生成随机但合理的净值曲线
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, len(dates))  # 平均日收益0.05%，波动1.5%
    nav_values = 1000000 * (1 + returns).cumprod()

    nav_df = pd.DataFrame({
        'value': nav_values
    }, index=dates)

    # 创建模拟的交易数据
    trade_dates = pd.date_range(start=start_date, end=end_date, freq='ME')[:20]
    trade_prices = []
    for td in trade_dates:
        # 找到最接近的交易日价格
        closest_idx = nav_df.index[nav_df.index <= td][-1] if any(nav_df.index <= td) else nav_df.index[0]
        trade_prices.append(nav_df.loc[closest_idx, 'value'])

    trades_df = pd.DataFrame({
        'datetime': trade_dates,
        'type': ['buy', 'sell'] * 10,
        'price': trade_prices,
        'pnl': np.random.normal(1000, 500, 20)
    })

    # 准备报告数据
    report_data = {
        'nav': nav_df,
        'trades': trades_df,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # 生成报告
    generator = ReportGenerator()
    output_path = 'output/bt_reports/test_report_fixed.html'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    generator.generate(
        strategy_results=report_data,
        output_path=output_path,
        title=f"回测分析报告 (测试修复) ({start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')})"
    )

    print(f"✅ 测试报告已生成: {output_path}")

    # 验证报告内容
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()

        # 检查是否包含预期内容
        checks = [
            ('生成时间', '生成时间' in content and 'N/A' not in content.split('生成时间')[1].split('</p>')[0]),
            ('核心指标', '核心表现指标' in content),
            ('收益率数据', 'nan%' not in content),
            ('夏普比率', '夏普比率' in content),
            ('最大回撤', '最大回撤' in content),
            ('交易统计', '交易统计摘要' in content),
            ('图表', 'data:image/png;base64,' in content)
        ]

        print("\n检查结果:")
        for name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {name}")

        all_passed = all(passed for _, passed in checks)

        if all_passed:
            print("\n🎉 所有检查通过！报告修复成功。")
        else:
            print("\n⚠️  部分检查未通过，请查看详情。")

        return all_passed

if __name__ == '__main__':
    success = generate_test_report()
    sys.exit(0 if success else 1)
