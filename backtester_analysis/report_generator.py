# report_generator.py
import pyfolio as pf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import warnings

# 1. 【新增】导入 quantstats 库
import quantstats


class ReportGenerator:
    """
    一个专门用于根据Backtrader回测结果生成可视化报告的类。
    现在支持 Pyfolio 和 QuantStats 两种报告。
    """

    def __init__(self, results):
        self.results = results
        self.returns = None
        self.positions = None
        self.transactions = None
        self.benchmark_rets = None

        self._prepare_pyfolio_data()  # 这个方法同样适用于 QuantStats

    def _prepare_pyfolio_data(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        pyfolio_analyzer = self.results.analyzers.getbyname('pyfolio')
        self.returns, self.positions, self.transactions, _ = pyfolio_analyzer.get_pf_items(
        )
        if self.returns.index.tz is None:
            self.returns.index = self.returns.index.tz_localize('UTC')

    def set_benchmark(self, benchmark_df):
        if benchmark_df is None or benchmark_df.empty:
            return
        benchmark_series = benchmark_df['close']
        self.benchmark_rets = benchmark_series.pct_change().fillna(0)
        if self.benchmark_rets.index.tz is None:
            self.benchmark_rets.index = self.benchmark_rets.index.tz_localize(
                'UTC')

        # 确保数据对齐
        self.returns.name = 'strategy'
        self.benchmark_rets.name = 'benchmark'
        aligned_df = pd.concat([self.returns, self.benchmark_rets],
                               axis=1,
                               join='inner')
        self.returns = aligned_df['strategy']
        self.benchmark_rets = aligned_df['benchmark']

    # ------------------------------------------------------------------
    # 【新增】生成 QuantStats 报告的方法
    # ------------------------------------------------------------------
    def generate_quantstats_report(self, filename='quantstats_report.html'):
        """
        使用 QuantStats 库生成一份独立的、交互式的HTML性能报告。
        :param filename: 输出的HTML文件名。
        """
        print(f"--- 正在使用 QuantStats 生成HTML报告: {filename} ---")
        try:
            # 【最终修正】为 QuantStats 准备不带时区信息 (timezone-naive) 的数据
            # 1. 复制数据，以防修改原始数据
            strategy_returns_naive = self.returns.copy()

            # 2. 移除索引的时区信息
            strategy_returns_naive.index = strategy_returns_naive.index.tz_localize(
                None)

            # 3. 如果有基准，同样处理
            benchmark_rets_naive = None
            if self.benchmark_rets is not None:
                benchmark_rets_naive = self.benchmark_rets.copy()
                benchmark_rets_naive.index = benchmark_rets_naive.index.tz_localize(
                    None)

            # 4. 将处理后的 naive 数据传递给 quantstats
            quantstats.reports.html(strategy_returns_naive,
                                    benchmark=benchmark_rets_naive,
                                    output=filename,
                                    title='策略表现分析报告 (QuantStats)',
                                    download_filename=filename)
            print(f"✅ QuantStats 报告已成功保存为: {filename}")
        except Exception as e:
            print(f"错误：生成 QuantStats 报告时失败: {e}")

    def generate_custom_report(self,
                               charts_to_plot: list,
                               filename='custom_report.pdf'):
        print(f"--- 正在生成定制化报告: {filename} ---")
        print(f"包含图表: {charts_to_plot}")

        with PdfPages(filename) as pdf:
            for chart_key in charts_to_plot:
                plot_func = self.PLOT_MAPPING.get(chart_key)
                if plot_func is None:
                    print(f"警告：未找到图表 '{chart_key}'，已跳过。")
                    continue

                try:
                    # 【最终修正】为 perf_stats 提供特殊处理逻辑
                    if chart_key == 'perf_stats':
                        # 1. 直接调用函数，它会在后台创建自己的图表
                        plot_func()
                        # 2. 捕获刚刚创建的图表
                        fig = plt.gcf()
                    else:
                        # 对于所有其他标准图表，我们自己创建fig和ax
                        fig, ax = plt.subplots(figsize=(14, 7))

                        # 根据函数及其版本特性，传递不同参数
                        if chart_key == 'returns':
                            plot_func(self.returns, ax=ax)
                        elif chart_key in ['rolling_beta', 'rolling_sharpe']:
                            plot_func(self.returns,
                                      benchmark_rets=self.benchmark_rets,
                                      ax=ax)
                        elif chart_key == 'exposures':
                            plot_func(self.returns, self.positions, ax=ax)
                        elif chart_key == 'turnover':
                            plot_func(self.returns,
                                      self.transactions,
                                      self.positions,
                                      ax=ax)
                        else:
                            plot_func(self.returns, ax=ax)

                    # 将最终得到的图表（我们创建的或捕获的）保存到PDF
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

                except Exception as e:
                    print(f"错误：生成图表 '{chart_key}' 时失败: {e}")
                    plt.close('all')

        print(f"✅ 定制化报告已成功保存为: {filename}")

    def _plot_perf_stats(self):
        """ 
        (私有方法) 专门用于调用“独立”的 performance stats 函数。
        此版本不接受任何参数。
        """
        pf.plotting.show_perf_stats(self.returns)
