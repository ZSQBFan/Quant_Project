# factor_report.py (已修改)

import platform
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import base64
from io import BytesIO
from . import analysis_metrics as metrics

# 设置 Matplotlib 字体以支持中文
system = platform.system()
if system == "Darwin":  # macOS
    matplotlib.rcParams['font.sans-serif'] = [
        'PingFang SC', 'Arial Unicode MS'
    ]
elif system == "Windows":
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
else:  # Linux 或其他系统
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 可根据需要调整
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class FactorReport:
    """
    生成单因子分析的HTML报告。
    """

    # --- 核心修改 (1): 修改 __init__ 方法以接收 benchmark_data ---
    def __init__(self,
                 factor_name: str,
                 factor_data: pd.DataFrame,
                 forward_return_periods: list,
                 benchmark_data: pd.DataFrame = None):
        """
        初始化报告生成器。

        Args:
            factor_name (str): 因子名称。
            factor_data (pd.DataFrame): 包含因子值和未来收益率的数据。
            forward_return_periods (list): 评估周期的列表 (例如 [1, 5, 10])。
            benchmark_data (pd.DataFrame, optional):
                包含基准价格数据 (必须有 'close' 列) 的DataFrame，用于绘制对比收益曲线。
                默认为 None。
        """
        self.factor_name = factor_name
        self.factor_data = factor_data
        self.periods = forward_return_periods
        self.results = {}

        # 如果传入了基准数据，计算其日收益率并储存
        self.benchmark_returns = None
        if benchmark_data is not None and not benchmark_data.empty:
            # 计算基准的日收益率，并移除第一个NaN值
            self.benchmark_returns = benchmark_data['close'].pct_change(
            ).dropna()

    # --- 核心修改 (1) 结束 ---

    def _run_analyses(self):
        """
        执行所有分析计算。
        """
        print(f"  正在为因子 '{self.factor_name}' 执行核心指标计算...")
        for p in self.periods:
            period_results = {}
            # 计算IC序列
            ic_series = metrics.calculate_rank_ic_series(self.factor_data, p)
            period_results['ic_series'] = ic_series
            # 计算IC统计
            period_results['ic_stats'] = metrics.analyze_ic_statistics(
                ic_series)
            # 计算分层收益
            period_results[
                'quantile_returns'] = metrics.calculate_quantile_returns(
                    self.factor_data, p)
            # 计算多空组合收益
            period_results[
                'ls_returns'] = metrics.calculate_factor_portfolio_returns(
                    self.factor_data, p)

            self.results[p] = period_results
        print("  核心指标计算完成。")

    def _fig_to_base64(self, fig):
        """
        将 matplotlib 图像转换为 base64 编码的字符串，用于嵌入HTML。
        """
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)  # 关闭图像，释放内存
        return f'data:image/png;base64,{img_str}'

    def _plot_ic_summary(self, period: int) -> str:
        """
        绘制IC序列图和IC分布直方图。
        """
        ic_series = self.results[period]['ic_series']
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # IC 序列图
        ic_series.plot(ax=axes[0], title=f'{period}日 IC 时间序列', grid=True)
        axes[0].axhline(0, color='gray', linestyle='--')
        axes[0].axhline(ic_series.mean(),
                        color='red',
                        linestyle='--',
                        label=f'均值: {ic_series.mean():.4f}')
        axes[0].legend()

        # IC 分布直方图
        ic_series.hist(bins=50, ax=axes[1], alpha=0.7)
        axes[1].set_title(f'{period}日 IC 分布直方图')
        axes[1].axvline(ic_series.mean(),
                        color='red',
                        linestyle='--',
                        label=f'均值: {ic_series.mean():.4f}')
        axes[1].legend()

        fig.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_quantile_returns(self, period: int) -> str:
        """
        绘制分层收益条形图。
        """
        quantile_returns = self.results[period]['quantile_returns']
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#d62728' if x < 0 else '#2ca02c' for x in quantile_returns]
        quantile_returns.plot(kind='bar', ax=ax, color=colors, alpha=0.8)
        ax.set_title(f'{period}日 因子分层年化收益率', fontsize=16)
        ax.set_ylabel('年化收益率')
        ax.axhline(0, color='gray', linestyle='--')
        ax.tick_params(axis='x', rotation=0)

        # 在条形图上显示数值
        for i, v in enumerate(quantile_returns):
            ax.text(i,
                    v,
                    f'{v*252:.2%}',
                    ha='center',
                    va='bottom' if v > 0 else 'top')

        return self._fig_to_base64(fig)

    # --- 核心修改 (2): 更新绘图函数以包含基准 ---
    def _plot_cumulative_factor_return(self, period: int) -> str:
        """
        绘制多空组合与基准的累计收益曲线。
        """
        ls_returns = self.results[period]['ls_returns']
        cumulative_returns = (1 + ls_returns).cumprod()

        fig, ax = plt.subplots(figsize=(12, 6))

        # 1. 绘制多空组合累计收益
        cumulative_returns.plot(ax=ax,
                                grid=True,
                                label='Long-Short Portfolio',
                                color='royalblue')

        # 2. 如果存在基准收益率，则绘制基准的累计收益
        if self.benchmark_returns is not None:
            # 将基准收益率的索引与多空组合的对齐，确保从同一起点开始比较
            aligned_benchmark_returns = self.benchmark_returns.reindex(
                ls_returns.index).fillna(0)

            # 计算基准的累计收益
            cumulative_benchmark = (1 + aligned_benchmark_returns).cumprod()

            cumulative_benchmark.plot(ax=ax,
                                      label='Benchmark (Buy & Hold)',
                                      linestyle='--',
                                      color='darkorange')

        # 3. 设置图表标题和标签
        ax.set_title(f'{period}日 多空组合 vs. 基准累计收益曲线', fontsize=16)
        ax.set_ylabel('累计净值')
        ax.legend()  # 显示图例

        return self._fig_to_base64(fig)

    # --- 核心修改 (2) 结束 ---

    def generate_html_report(self, output_filename: str):
        """
        生成并保存完整的HTML报告。
        """
        self._run_analyses()
        print(f"  正在为 '{self.factor_name}' 生成HTML报告...")

        html = f"""
        <html>
        <head>
            <title>因子分析报告: {self.factor_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: auto; }}
                .section {{ border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
                .plot {{ text-align: center; margin-top: 15px; }}
                img {{ max-width: 100%; height: auto; }}
                table {{ border-collapse: collapse; width: 50%; margin-top: 15px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>因子分析报告: {self.factor_name}</h1>
        """

        for p in self.periods:
            html += f"<div class='section'><h2>分析周期: {p}个交易日</h2>"

            # IC 统计表格
            ic_stats_df = pd.DataFrame.from_dict(self.results[p]['ic_stats'],
                                                 orient='index',
                                                 columns=['Value'])
            html += "<h3>IC 统计摘要</h3>"
            html += ic_stats_df.to_html()

            # IC 图表
            ic_plot_b64 = self._plot_ic_summary(p)
            html += "<div class='plot'><h3>IC 序列与分布</h3>"
            html += f"<img src='{ic_plot_b64}'></div>"

            # 分层收益图表
            quantile_plot_b64 = self._plot_quantile_returns(p)
            html += "<div class='plot'><h3>因子分层收益</h3>"
            html += f"<img src='{quantile_plot_b64}'></div>"

            # 累计收益图表
            cumulative_plot_b64 = self._plot_cumulative_factor_return(p)
            html += "<div class='plot'><h3>多空组合累计收益</h3>"
            html += f"<img src='{cumulative_plot_b64}'></div>"

            html += "</div>"

        html += """
            </div>
        </body>
        </html>
        """

        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html)
        print("  HTML报告生成完毕。")
