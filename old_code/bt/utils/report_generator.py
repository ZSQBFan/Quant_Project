# bt/utils/report_generator.py

"""
回测报告生成器

MVP 实现：生成简单的 HTML 报告，包含：
- 最终净值
- Sharpe Ratio
- 最大回撤
"""

import os
import logging
from datetime import datetime


class ReportGenerator:
    """
    简单的 HTML 报告生成器
    """
    
    def __init__(self, output_dir: str = './bt_report'):
        """
        参数:
            output_dir: 报告输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate(self, cerebro, strategy, analyzers: dict) -> str:
        """
        生成 HTML 报告
        
        参数:
            cerebro: Backtrader 引擎实例
            strategy: 策略实例
            analyzers: 分析器结果字典
        
        返回:
            报告文件路径
        """
        # 获取基础指标
        final_value = cerebro.broker.getvalue()
        initial_value = cerebro.broker.startingcash
        total_return = (final_value / initial_value - 1) * 100
        
        # 获取 Sharpe Ratio
        sharpe_analysis = analyzers.get('sharpe', {})
        sharpe_ratio = sharpe_analysis.get('sharperatio', None)
        sharpe_str = f"{sharpe_ratio:.3f}" if sharpe_ratio is not None else "N/A"
        
        # 获取最大回撤
        drawdown_analysis = analyzers.get('drawdown', {})
        max_drawdown = drawdown_analysis.get('max', {}).get('drawdown', None)
        drawdown_str = f"{max_drawdown:.2f}%" if max_drawdown is not None else "N/A"
        
        # 生成 HTML
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Backtrader 回测报告</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 40px;
            background: #f5f5f5;
        }}
        .container {{ 
            max-width: 800px; 
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        }}
        h1 {{ 
            color: #333;
            margin-bottom: 10px;
        }}
        .timestamp {{ 
            color: #666;
            font-size: 14px;
            margin-bottom: 30px;
        }}
        .metrics {{ 
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 30px;
        }}
        .metric {{ 
            flex: 1;
            min-width: 200px;
            padding: 24px;
            background: #f8f9fa;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{ 
            font-size: 32px;
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 8px;
        }}
        .metric-value.positive {{ color: #4CAF50; }}
        .metric-value.negative {{ color: #f44336; }}
        .metric-label {{ 
            color: #666;
            font-size: 14px;
        }}
        .summary {{
            margin-top: 30px;
            padding: 20px;
            background: #e3f2fd;
            border-radius: 8px;
        }}
        .summary-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #bbdefb;
        }}
        .summary-row:last-child {{ border-bottom: none; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Backtrader 回测报告</h1>
        <div class="timestamp">生成时间: {timestamp}</div>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{final_value:,.2f}</div>
                <div class="metric-label">最终净值</div>
            </div>
            
            <div class="metric">
                <div class="metric-value {'positive' if total_return >= 0 else 'negative'}">{total_return:+.2f}%</div>
                <div class="metric-label">总收益率</div>
            </div>
            
            <div class="metric">
                <div class="metric-value">{sharpe_str}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            
            <div class="metric">
                <div class="metric-value negative">{drawdown_str}</div>
                <div class="metric-label">最大回撤</div>
            </div>
        </div>
        
        <div class="summary">
            <div class="summary-row">
                <span>初始资金</span>
                <span>{initial_value:,.2f}</span>
            </div>
            <div class="summary-row">
                <span>最终净值</span>
                <span>{final_value:,.2f}</span>
            </div>
            <div class="summary-row">
                <span>绝对收益</span>
                <span>{final_value - initial_value:+,.2f}</span>
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        # 保存文件
        report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = os.path.join(self.output_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info(f"✅ 报告已生成: {report_path}")
        # print(f"✅ 报告已生成: {report_path}")
        
        return report_path