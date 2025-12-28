"""
测试回测报告生成器（包含换手率和交易统计）
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.reports.report_generator import ReportGenerator


def create_test_data():
    """创建测试数据"""
    # 生成时间序列
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

    # 生成净值曲线（模拟收益）
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, len(dates))
    nav_values = 1000000 * np.cumprod(1 + returns)

    nav_df = pd.DataFrame({
        'value': nav_values
    }, index=dates)

    # 模拟交易统计
    trades_stats = {
        'total_trades': 150,
        'win_rate': 0.58,
        'profit_loss_ratio': 1.35,
        'avg_profit': 2500.50,
        'avg_loss': 1850.25
    }

    # 模拟换手率统计
    turnover_stats = {
        'avg_turnover': 0.0262,  # 2.62%
        'annual_turnover': 6.59,  # 6.59x
        'total_turnover': 0.95,   # 95%
        'total_buy': 5000000.0,
        'total_sell': 4800000.0
    }

    return {
        'nav': nav_df,
        'trades': pd.DataFrame(),  # 空的交易明细
        'trades_stats': trades_stats,
        'turnover_stats': turnover_stats,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def test_report_generation():
    """测试报告生成"""
    print("=" * 70)
    print("测试回测报告生成器（包含换手率和交易统计）")
    print("=" * 70)

    # 创建测试数据
    print("\n📊 创建测试数据...")
    test_data = create_test_data()
    print(f"  净值数据点: {len(test_data['nav'])}")
    print(f"  交易统计: {test_data['trades_stats']}")
    print(f"  换手率统计: {test_data['turnover_stats']}")

    # 生成报告
    print("\n📝 生成HTML报告...")
    generator = ReportGenerator()
    output_path = 'output/test_report_with_turnover.html'

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        generator.generate(
            strategy_results=test_data,
            output_path=output_path,
            title="测试报告（包含换手率和交易统计）"
        )
        print(f"✅ 报告生成成功: {output_path}")

        # 验证报告内容
        print("\n🔍 验证报告内容...")
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        checks = {
            '包含换手率统计': '换手率统计' in content,
            '包含交易统计': '交易统计摘要' in content,
            '包含总交易次数': '150' in content,  # 应该显示150次交易
            '包含平均换手率': '2.62%' in content,
            '包含年化换手率': '6.59x' in content,
            '包含胜率': '58.00%' in content
        }

        all_passed = True
        for check_name, result in checks.items():
            status = "✅" if result else "❌"
            print(f"  {status} {check_name}: {result}")
            if not result:
                all_passed = False

        if all_passed:
            print("\n✅ 所有检查通过！")
            return True
        else:
            print("\n❌ 部分检查未通过")
            return False

    except Exception as e:
        print(f"❌ 报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_report_generation()
    sys.exit(0 if success else 1)
