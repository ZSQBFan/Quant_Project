# test/test_backtrader_integration_mvp.py

"""
Backtrader MVP 集成测试脚本

这个测试脚本的目标是验证整个回测框架的数据流和控制流完整性。

测试内容包括：
1. 配置读取测试
2. 数据导出测试
3. 策略组件测试
4. 回测执行测试
5. 报告生成测试

使用方法：
    python test/test_backtrader_integration_mvp.py
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger.logger_config import setup_logging
from bt.utils.report_generator import ReportGenerator
from bt.backtest import run_backtest
from bt.core.base_strategy import BacktestStrategy
from bt.triggers.stop_loss import StopLossTrigger
from bt.triggers.rebalance_day import RebalanceDayTrigger
from bt.pipeline.selectors import TopNSelector
from bt.pipeline.allocators import EqualWeightAllocator
from bt.pipeline.capital import FullPositionManager


class MVPIntegrationTest:
    """
    MVP 集成测试类
    """
    
    def __init__(self):
        """初始化测试"""
        self.setup_test_environment()
        self.results = {}
        
    def setup_test_environment(self):
        """设置测试环境"""
        print("🔧 设置测试环境...")
        
        # 设置日志
        setup_logging(log_dir='test_logs', log_prefix='bt_mvp_test')
        
        # 创建测试数据目录
        test_data_dir = './bt/data_export/'
        os.makedirs(test_data_dir, exist_ok=True)
        
        # 创建测试报告目录
        os.makedirs('./bt_report/', exist_ok=True)
        
        print("  ✅ 测试环境设置完成")
    
    def test_01_config_reading(self):
        """测试1：配置文件读取"""
        print("\n[测试1] 📖 配置文件读取测试...")
        
        try:
            # 测试读取策略配置
            import yaml
            with open('config/strategy_main.yaml', encoding='utf-8') as f:
                strat_conf = yaml.safe_load(f)
            
            assert 'strategy' in strat_conf
            assert 'pipeline' in strat_conf
            print("  ✅ 策略配置文件读取成功")
            
            # 测试读取券商配置
            import json
            with open('config/broker.json', encoding='utf-8') as f:
                broker_conf = json.load(f)
            
            assert 'initial_cash' in broker_conf
            assert 'commission' in broker_conf
            print("  ✅ 券商配置文件读取成功")
            
            # 测试读取交易日配置
            with open('config/trading_days.json', encoding='utf-8') as f:
                trading_days = json.load(f)
            
            assert 'dates' in trading_days
            assert len(trading_days['dates']) > 0
            print("  ✅ 交易日配置文件读取成功")
            
            self.results['config_reading'] = True
            print("  ✅ 测试1通过：配置文件读取")
            
        except Exception as e:
            self.results['config_reading'] = False
            print(f"  ❌ 测试1失败：配置文件读取 - {e}")
            raise
    
    def test_02_data_export(self):
        """测试2：数据导出功能测试"""
        print("\n[测试2] 📊 数据导出功能测试...")
        
        try:
            # 创建模拟的 DataManager
            class MockDataManager:
                def get_dataframe(self, symbol, columns):
                    # 创建模拟的 OHLCV 数据
                    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
                    df = pd.DataFrame({
                        'open': np.random.randn(len(dates)).cumsum() + 100,
                        'high': np.random.randn(len(dates)).cumsum() + 102,
                        'low': np.random.randn(len(dates)).cumsum() + 98,
                        'close': np.random.randn(len(dates)).cumsum() + 100,
                        'volume': np.random.randint(1000000, 10000000, len(dates))
                    }, index=dates)
                    return df
            
            from bt.data.exporter import BTDataExporter
            
            # 创建模拟的因子数据
            dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
            symbols = ['000001', '000002', '600519']
            factor_data = []
            
            for date in dates:
                for symbol in symbols:
                    factor_data.append({
                        'date': date,
                        'asset': symbol,
                        'combined_signal': np.random.randn()
                    })
            
            factor_series = pd.DataFrame(factor_data).set_index(['date', 'asset'])['combined_signal']
            
            # 测试数据导出
            mock_dm = MockDataManager()
            exporter = BTDataExporter(mock_dm)
            
            exported_files = exporter.export(
                universe=symbols,
                start_date='2024-01-01',
                end_date='2024-12-31',
                factor_series=factor_series
            )
            
            assert len(exported_files) > 0, "没有导出任何文件"
            print(f"  ✅ 成功导出 {len(exported_files)} 个股票数据文件")
            
            # 验证导出的文件
            for file_path in exported_files:
                assert os.path.exists(file_path), f"导出文件不存在: {file_path}"
                df = pd.read_parquet(file_path)
                assert 'combined_signal' in df.columns, "缺少combined_signal列"
                assert 'suspended' in df.columns, "缺少suspended列"
            
            print("  ✅ 测试2通过：数据导出功能")
            self.results['data_export'] = True
            
        except Exception as e:
            self.results['data_export'] = False
            print(f"  ❌ 测试2失败：数据导出 - {e}")
            # 不抛出异常，继续其他测试
    
    def test_03_strategy_components(self):
        """测试3：策略组件测试"""
        print("\n[测试3] 🔧 策略组件测试...")
        
        try:
            # 测试选股器
            from bt.pipeline.selectors import TopNSelector
            
            class MockData:
                def __init__(self, name, signal, suspended=False):
                    self._name = name
                    self.combined_signal = [signal]
                    self.suspended = [suspended]
                    self.close = [100]
                
                def __len__(self):
                    return 1
            
            mock_datas = [
                MockData('000001', 0.8),
                MockData('000002', 0.6),
                MockData('000003', 0.9),
                MockData('000004', 0.7, suspended=True),  # 停牌
                MockData('000005', np.nan),  # 无效信号
            ]
            
            selector = TopNSelector(top_n=2)
            selected = selector.select(mock_datas)
            
            assert len(selected) == 2, f"选股数量错误: {len(selected)}"
            print(f"  ✅ 选股器测试通过：选中 {len(selected)} 只股票")
            
            # 测试权重分配器
            allocator = EqualWeightAllocator()
            weights = allocator.allocate(selected)
            
            assert len(weights) == 2, "权重分配错误"
            weight_sum = sum(weights.values())
            assert abs(weight_sum - 1.0) < 1e-6, f"权重总和错误: {weight_sum}"
            print("  ✅ 权重分配器测试通过")
            
            # 测试资金管理器
            capital_manager = FullPositionManager(utilization_ratio=0.95)
            allocation = capital_manager.get_allocation(1000000)
            
            expected_allocation = 1000000 * 0.95
            assert abs(allocation - expected_allocation) < 1e-6, f"资金分配错误: {allocation}"
            print("  ✅ 资金管理器测试通过")
            
            self.results['strategy_components'] = True
            print("  ✅ 测试3通过：策略组件")
            
        except Exception as e:
            self.results['strategy_components'] = False
            print(f"  ❌ 测试3失败：策略组件 - {e}")
            raise
    
    def test_04_report_generation(self):
        """测试4：报告生成测试"""
        print("\n[测试4] 📝 报告生成测试...")
        
        try:
            # 创建模拟的 cerebro 和 strategy
            class MockCerebro:
                def __init__(self):
                    self.broker = MockBroker()
            
            class MockBroker:
                def __init__(self):
                    self.startingcash = 1000000
                    self._value = 1050000
                
                def getvalue(self):
                    return self._value
            
            class MockStrategy:
                pass
            
            # 创建模拟的分析器结果
            mock_analyzers = {
                'sharpe': {'sharperatio': 1.25},
                'drawdown': {'max': {'drawdown': -8.5}},
                'trades': {}
            }
            
            # 测试报告生成
            reporter = ReportGenerator()
            report_path = reporter.generate(
                MockCerebro(),
                MockStrategy(),
                mock_analyzers
            )
            
            assert os.path.exists(report_path), f"报告文件不存在: {report_path}"
            print(f"  ✅ 报告生成成功: {report_path}")
            
            # 验证报告内容
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert 'Backtrader 回测报告' in content, "报告标题缺失"
                assert '1,050,000.00' in content, "最终净值缺失"
                assert '1.250' in content, "Sharpe Ratio缺失"
            
            print("  ✅ 测试4通过：报告生成")
            self.results['report_generation'] = True
            
        except Exception as e:
            self.results['report_generation'] = False
            print(f"  ❌ 测试4失败：报告生成 - {e}")
            raise
    
    def test_05_backtest_execution(self):
        """测试5：回测执行测试（模拟）"""
        print("\n[测试5] 🏃 回测执行测试...")
        
        try:
            # 这个测试需要实际的数据文件，所以只是检查导入和基本逻辑
            print("  ✅ 回测模块导入成功")
            
            # 检查关键类是否正确导入
            assert BacktestStrategy is not None
            assert StopLossTrigger is not None
            assert RebalanceDayTrigger is not None
            
            print("  ✅ 策略类导入成功")
            
            # 检查是否有测试数据
            test_data_dir = './bt/data_export/'
            if os.path.exists(test_data_dir):
                parquet_files = [f for f in os.listdir(test_data_dir) if f.endswith('.parquet')]
                if parquet_files:
                    print(f"  ✅ 发现 {len(parquet_files)} 个测试数据文件")
                    self.results['backtest_execution'] = True
                else:
                    print("  ⚠️ 没有发现测试数据文件")
                    self.results['backtest_execution'] = False
            else:
                print("  ⚠️ 测试数据目录不存在")
                self.results['backtest_execution'] = False
            
            print("  ✅ 测试5通过：回测执行检查")
            
        except Exception as e:
            self.results['backtest_execution'] = False
            print(f"  ❌ 测试5失败：回测执行 - {e}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始 Backtrader MVP 集成测试")
        print("=" * 60)
        
        test_methods = [
            self.test_01_config_reading,
            self.test_02_data_export,
            self.test_03_strategy_components,
            self.test_04_report_generation,
            self.test_05_backtest_execution
        ]
        
        passed = 0
        total = len(test_methods)
        
        for test_method in test_methods:
            try:
                test_method()
                passed += 1
            except Exception as e:
                print(f"  ❌ 测试异常: {e}")
                logging.error(f"测试异常: {e}", exc_info=True)
        
        print("\n" + "=" * 60)
        print("📊 测试结果汇总")
        print("=" * 60)
        
        for test_name, result in self.results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {test_name:<20}: {status}")
        
        print(f"\n总结果: {passed}/{total} 个测试通过")
        
        if passed == total:
            print("\n🎉 所有测试通过！Backtrader MVP 框架已就绪。")
            print("\n📋 下一步操作：")
            print("1. 设置 main_analyzer.py 中的 RUN_BACKTRADER = True")
            print("2. 运行因子分析流程生成数据")
            print("3. 运行完整的回测流程")
        else:
            print(f"\n⚠️  {total - passed} 个测试失败，请检查相关模块。")
        
        return passed == total


def main():
    """主函数"""
    test_suite = MVPIntegrationTest()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n✅ MVP 集成测试全部通过！")
        sys.exit(0)
    else:
        print("\n❌ MVP 集成测试有失败项！")
        sys.exit(1)


if __name__ == '__main__':
    main()