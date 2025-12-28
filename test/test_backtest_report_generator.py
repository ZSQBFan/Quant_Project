import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import MagicMock, patch
from backtest.reports.utils.metrics import calculate_metrics, calculate_drawdown, calculate_trade_stats
from backtest.reports.analyzers.nav_analyzer import NAVAnalyzer
from backtest.reports.components.charts import NavCurveChart, DrawdownChart, MonthlyHeatmap
from backtest.reports.components.tables import MetricsTable, TradeStatsTable
from backtest.reports.report_generator import ReportGenerator

@pytest.fixture
def mock_nav_data():
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    # 生成一些随机波动的净值数据
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 100)
    nav_values = (1 + returns).cumprod()
    nav_df = pd.DataFrame({'value': nav_values}, index=dates)
    return nav_df

@pytest.fixture
def mock_trades_data():
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    trades = pd.DataFrame({
        'datetime': dates,
        'type': ['buy', 'sell'] * 5,
        'price': [10.0 + i for i in range(10)],
        'pnl': [1.0, -0.5, 2.0, -1.0, 3.0, -1.5, 4.0, -2.0, 5.0, -2.5]
    })
    return trades

class TestMetrics:
    def test_calculate_drawdown(self, mock_nav_data):
        dd = calculate_drawdown(mock_nav_data['value'])
        assert isinstance(dd, pd.Series)
        assert dd.max() <= 0
        assert len(dd) == len(mock_nav_data)

    def test_calculate_metrics(self, mock_nav_data):
        metrics = calculate_metrics(mock_nav_data['value'])
        assert 'total_return' in metrics
        assert 'annual_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert metrics['max_drawdown'] <= 0

    def test_calculate_trade_stats(self, mock_trades_data):
        stats = calculate_trade_stats(mock_trades_data)
        assert stats['total_trades'] == 10
        assert stats['win_rate'] == 0.5
        assert stats['avg_profit'] == 3.0
        assert stats['avg_loss'] == 1.5

class TestAnalyzers:
    def test_nav_analyzer_process_backtrader_results(self):
        analyzer = NAVAnalyzer()
        mock_strategy = MagicMock()
        
        # Mock observers
        mock_value = MagicMock()
        mock_value.value = [100.0, 101.0, 102.0]
        # Backtrader uses float dates
        mock_value.datetime.get.return_value = [738521.0, 738522.0, 738523.0] 
        mock_strategy.observers.value = mock_value
        
        # Mock analyzers
        mock_strategy.analyzers.getitems.return_value = []
        
        with patch('backtrader.num2date') as mock_num2date:
            mock_num2date.side_effect = lambda x: pd.to_datetime('2023-01-01') + pd.Timedelta(days=int(x-738521.0))
            result = analyzer.process_backtrader_results(mock_strategy)
            
        assert 'nav' in result
        assert 'trades' in result
        assert len(result['nav']) == 3
        assert result['nav'].iloc[0]['value'] == 100.0

class TestComponents:
    def test_metrics_table(self, mock_nav_data):
        component = MetricsTable()
        html = component.render({'nav': mock_nav_data})
        assert '核心表现指标' in html
        assert '年化收益率' in html
        assert '夏普比率' in html

    def test_trade_stats_table(self, mock_trades_data):
        component = TradeStatsTable()
        html = component.render({'trades': mock_trades_data})
        assert '交易统计摘要' in html
        assert '胜率' in html
        assert '50.00%' in html

    def test_nav_curve_chart(self, mock_nav_data, mock_trades_data):
        component = NavCurveChart()
        html = component.render({'nav': mock_nav_data, 'trades': mock_trades_data})
        assert 'chart-container' in html
        assert 'data:image/png;base64,' in html

    def test_drawdown_chart(self, mock_nav_data):
        component = DrawdownChart()
        html = component.render({'nav': mock_nav_data})
        assert 'chart-container' in html

    def test_monthly_heatmap(self, mock_nav_data):
        component = MonthlyHeatmap()
        html = component.render({'nav': mock_nav_data})
        assert 'chart-container' in html

class TestReportGenerator:
    def test_generate_report(self, mock_nav_data, mock_trades_data, tmp_path):
        output_file = tmp_path / "report.html"
        generator = ReportGenerator()
        
        report_data = {
            'nav': mock_nav_data,
            'trades': mock_trades_data
        }
        
        path = generator.generate(report_data, str(output_file))
        assert os.path.exists(path)
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert '回测分析报告' in content
            assert 'chart-container' in content
            assert 'metrics-table' in content

    def test_generate_report_with_error_handling(self, tmp_path):
        output_file = tmp_path / "error_report.html"
        generator = ReportGenerator()
        
        # 传入非法数据导致组件渲染失败
        report_data = {'nav': None} 
        
        path = generator.generate(report_data, str(output_file))
        assert os.path.exists(path)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert '无净值数据' in content or '无数据' in content
