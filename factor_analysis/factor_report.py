# factor_analysis/factor_report.py (已重构 - 新增 Hexbin 图)

import platform
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import base64
from io import BytesIO
from factor_analysis import analysis_metrics as metrics
import logging  # <- 【【【新增】】】
from scipy.stats import linregress  # <--- 【【【【【【新增 IMPORT】】】】】】

# 设置 Matplotlib 字体以支持中文
try:
    system = platform.system()
    if system == "Darwin":  # macOS
        matplotlib.rcParams['font.sans-serif'] = [
            'PingFang SC', 'Arial Unicode MS'
        ]
    elif system == "Windows":
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    else:  # Linux 或其他系统
        matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    logging.info("ℹ️ Matplotlib 字体已配置，支持中文显示。")
except Exception as e:
    logging.warning(f"⚠️ Matplotlib 字体配置失败，图表中的中文可能显示为方框: {e}")


class FactorReport:
    """
    生成单因子分析的HTML报告。
    
    【【重构日志】】:
    - 2025-11-09:
      - 引入 'logging' 模块，替换所有 'print' 语句。
    - 2025-11-11: (用户要求)
      - 新增 _plot_rank_return_scatter (Hexbin) 函数，
      - 用于替代（或补充）不稳健的 pd.qcut 分层图。
    """

    def __init__(self,
                 factor_name: str,
                 factor_data: pd.DataFrame,
                 forward_return_periods: list,
                 benchmark_data: pd.DataFrame = None):
        """
        初始化报告生成器。
        """
        self.factor_name = factor_name
        self.factor_data = factor_data
        self.periods = forward_return_periods
        self.results = {}
        logging.info(f"ℹ️ FactorReport for '{self.factor_name}' 已初始化。")

        # 如果传入了基准数据，计算其日收益率并储存
        self.benchmark_returns = None
        if benchmark_data is not None and not benchmark_data.empty:
            self.benchmark_returns = benchmark_data['close'].pct_change(
            ).dropna()
            logging.info("  > 📈 基准数据已加载，将用于对比。")
        else:
            logging.info("  > ℹ️ 未加载基准数据，报告中将不包含对比。")

    def _run_analyses(self):
        """
        执行所有分析计算。
        """
        # 【【【修改】】】
        logging.info(f"  > ⚙️ 正在为因子 '{self.factor_name}' 执行核心指标计算...")
        for p in self.periods:
            logging.debug(f"    > 正在计算周期 {p}d...")
            period_results = {}
            # 计算IC序列
            ic_series = metrics.calculate_rank_ic_series(self.factor_data, p)
            period_results['ic_series'] = ic_series
            # 计算IC统计
            period_results['ic_stats'] = metrics.analyze_ic_statistics(
                ic_series)

            # (注意: 即使 qcut 失败，我们仍然尝试运行它以获取它生成的"Q1"图)
            try:
                # 计算分层收益
                period_results[
                    'quantile_returns'] = metrics.calculate_quantile_returns(
                        self.factor_data, p)
            except Exception as e:
                logging.error(
                    f"❌ _run_analyses: calculate_quantile_returns 失败: {e}")
                # 创建一个空的 Series 以防函数崩溃
                period_results['quantile_returns'] = pd.Series(
                    name=f'mean_return_{p}d')

            # 计算多空组合收益
            period_results[
                'ls_returns'] = metrics.calculate_factor_portfolio_returns(
                    self.factor_data, p)

            self.results[p] = period_results
        # 【【【修改】】】
        logging.info("  > ✅ 核心指标计算完成。")

    def _fig_to_base64(self, fig):
        """
        将 matplotlib 图像转换为 base64 编码的字符串，用于嵌入HTML。
        """
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f'data:image/png;base64,{img_str}'

    def _plot_ic_summary(self, period: int) -> str:
        """
        绘制IC序列图和IC分布直方图。
        """
        logging.debug(f"    > 🎨 正在绘制 {period}d IC 摘要图...")
        ic_series = self.results[period]['ic_series']
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        ic_series.plot(ax=axes[0], title=f'{period}日 IC 时间序列', grid=True)
        axes[0].axhline(0, color='gray', linestyle='--')
        axes[0].axhline(ic_series.mean(),
                        color='red',
                        linestyle='--',
                        label=f'均值: {ic_series.mean():.4f}')
        axes[0].legend()

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
        (注意: 此函数可能因 qcut 失败而只显示 Q1)
        """
        logging.debug(f"    > 🎨 正在绘制 {period}d 分层收益图...")
        quantile_returns = self.results[period]['quantile_returns']

        # (安全检查，如果 qcut 失败)
        if quantile_returns.empty:
            logging.warning(
                f"  > ⚠️ _plot_quantile_returns: {period}d 分层收益序列为空。")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, '分层收益计算失败', ha='center', color='red')
            return self._fig_to_base64(fig)

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#d62728' if x < 0 else '#2ca02c' for x in quantile_returns]
        quantile_returns.plot(kind='bar', ax=ax, color=colors, alpha=0.8)
        ax.set_title(f'{period}日 因子分层年化收益率', fontsize=16)
        ax.set_ylabel('年化收益率')
        ax.axhline(0, color='gray', linestyle='--')
        ax.tick_params(axis='x', rotation=0)

        for i, v in enumerate(quantile_returns):
            ax.text(i,
                    v,
                    f'{v*252:.2%}',
                    ha='center',
                    va='bottom' if v > 0 else 'top')

        return self._fig_to_base64(fig)

    def _plot_cumulative_factor_return(self, period: int) -> str:
        """
        【【【核心修改】】】
        绘制基于重叠组合的多空策略近似累计收益曲线。

        该方法为每个分析周期（period）独立计算一条收益曲线，逻辑如下：
        1. 在每个交易日 `t`，根据因子值构建一个多空组合。
        2. 该组合在未来 `period` 天的总收益，由 `forward_return_{period}d` 给出。
        3. 我们将这个总收益除以 `period`，得到一个近似的“平均每日收益”。
        4. 使用这个每日收益序列，通过复利计算（cumprod）生成累计净值曲线。

        这种方法可以直观地比较不同持有期策略的表现，并与基准进行对比。
        """
        logging.debug(f"    > 🎨 正在为 {period}d 周期绘制近似累计收益图...")

        return_col = f'forward_return_{period}d'

        # 检查所需列是否存在
        if 'factor_value' not in self.factor_data.columns or return_col not in self.factor_data.columns:
            logging.warning(
                f"  > ⚠️ [Cumulative Plot] 缺少 '{return_col}' 或 'factor_value' 列，无法为 {period}d 周期绘图。"
            )
            fig, ax = plt.subplots()
            ax.text(0.5,
                    0.5,
                    f'数据缺失，无法生成{period}d收益图',
                    ha='center',
                    color='red')
            return self._fig_to_base64(fig)

        # 定义一个在 groupby 中应用的函数，用于计算每日的多空收益
        def _calculate_daily_ls_return(df_group: pd.DataFrame) -> float:
            try:
                # 使用 qcut 进行分位数切割，构建多空组合
                df_group['quantile'] = pd.qcut(df_group['factor_value'],
                                               5,
                                               labels=False,
                                               duplicates='drop')

                # 确保最高和最低分位数都存在
                if 4 in df_group['quantile'].values and 0 in df_group[
                        'quantile'].values:
                    long_ret = df_group[df_group['quantile'] ==
                                        4][return_col].mean()
                    short_ret = df_group[df_group['quantile'] ==
                                         0][return_col].mean()

                    # 计算 period-day 的多空总收益
                    total_ls_return = long_ret - short_ret

                    # 【关键】将总收益平均分摊到 period 天，得到近似的每日收益
                    # 对于0或1天的周期，直接使用总收益
                    if period <= 1:
                        return total_ls_return
                    else:
                        return total_ls_return / period
                else:
                    return 0.0  # 如果无法形成多空组合，则当日收益为0
            except Exception:
                # 如果 qcut 失败（例如当日股票数少于5），也返回0
                return 0.0

        # 对每一天的数据应用上述函数，得到每日的近似多空收益序列
        daily_ls_returns = self.factor_data.groupby(
            level='date').apply(_calculate_daily_ls_return)

        # 使用每日收益序列计算累计净值
        cumulative_returns = (1 + daily_ls_returns).cumprod()

        # --- 开始绘图 ---
        fig, ax = plt.subplots(figsize=(12, 6))

        cumulative_returns.plot(ax=ax,
                                grid=True,
                                label=f'多空组合 (持有期: {period}天)',
                                color='royalblue')

        # 绘制基准 (Buy & Hold) 曲线
        if self.benchmark_returns is not None:
            # 对齐基准收益率的索引
            aligned_benchmark_returns = self.benchmark_returns.reindex(
                daily_ls_returns.index).fillna(0)
            cumulative_benchmark = (1 + aligned_benchmark_returns).cumprod()
            cumulative_benchmark.plot(ax=ax,
                                      label='基准 (Buy & Hold)',
                                      linestyle='--',
                                      color='darkorange')

        ax.set_title(f'近似累计收益曲线 (分析周期: {period}天)', fontsize=16)
        ax.set_ylabel('累计净值')
        ax.legend()
        ax.set_yscale('log')  # 使用对数坐标轴，更清晰地展示收益曲线的相对变化

        return self._fig_to_base64(fig)

    # ==============================================================================
    # 【【【【【【 函数功能：新增的绘图函数 】】】】】】
    # ==============================================================================
    def _plot_rank_return_scatter(self, period: int) -> str:
        """
        绘制因子百分位排名 vs. 远期收益率的 Hexbin 密度图。
        这是一个更稳健的、用于替代 qcut 分层图的可视化方法。
        """
        logging.debug(f"    > 🎨 正在绘制 {period}d 因子排名-收益率 Hexbin 图...")

        return_col = f'forward_return_{period}d'

        # 1. 准备数据
        if 'factor_value' not in self.factor_data.columns or return_col not in self.factor_data.columns:
            logging.warning(
                f"  > ⚠️ [Hexbin] 缺少 'factor_value' 或 '{return_col}' 列。")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f'Hexbin 图数据缺失', ha='center', color='red')
            return self._fig_to_base64(fig)

        # (我们必须在 groupby 之前 .copy() 以避免 SettingWithCopyWarning)
        data_subset = self.factor_data[['factor_value', return_col]].copy()

        # 2. 计算截面百分位排名 (0.0 -> 1.0)
        # (这是核心，它替代了 qcut)
        # (self.factor_data 默认索引是 date)
        data_subset['factor_rank_pct'] = data_subset.groupby(
            level='date')['factor_value'].rank(pct=True)

        # 3. 丢弃 NaN 值
        plot_data = data_subset.dropna()

        if plot_data.empty or len(plot_data) < 100:
            logging.warning(
                f"  > ⚠️ [Hexbin] {period}d 清理(dropna)后数据点不足 ({len(plot_data)})。"
            )
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f'Hexbin 图数据不足', ha='center', color='red')
            return self._fig_to_base64(fig)

        fig, ax = plt.subplots(figsize=(10, 6))

        # 4. 绘制 Hexbin 2D 密度图
        # (gridsize 越小，格子越大；cmap 是颜色)
        try:
            im = ax.hexbin(
                x=plot_data['factor_rank_pct'],
                y=plot_data[return_col],
                gridsize=50,  # 可根据数据点调整 (例如 30, 50, 100)
                cmap='viridis',  # (黄-绿-蓝)
                mincnt=1  # (显示至少有1个点的格子)
            )
            fig.colorbar(im, ax=ax, label='数据点密度')
        except Exception as e:
            logging.error(f"  > ❌ [Hexbin] 绘制 hexbin 时出错: {e}", exc_info=True)
            ax.text(0.5, 0.5, f'Hexbin 绘图失败: {e}', ha='center', color='red')
            return self._fig_to_base64(fig)

        # 5. 绘制趋势线 (线性回归)
        try:
            x = plot_data['factor_rank_pct']
            y = plot_data[return_col]

            # (使用 scipy.stats.linregress 计算)
            slope, intercept, r_value, p_value, std_err = linregress(x, y)

            # (生成趋势线上的点)
            trend_x = [0, 1]
            trend_y = [intercept + slope * 0, intercept + slope * 1]

            ax.plot(
                trend_x,
                trend_y,
                color='#d62728',  # (红色)
                linestyle='--',
                linewidth=2,
                label=f'趋势线 (R²: {r_value**2:.4f})')
        except Exception as e:
            logging.warning(f"  > ⚠️ [Hexbin] 无法计算趋势线: {e}")

        # 6. 格式化图表
        ax.set_title(f'{period}日 因子百分位排名 vs. 远期收益率 (Hexbin 图)', fontsize=16)
        ax.set_xlabel('因子值截面百分位排名 (0.0 = 最差, 1.0 = 最好)')
        ax.set_ylabel(f'{period}日 远期收益率')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlim(0, 1)  # X 轴固定为 0 到 1
        ax.legend()

        fig.tight_layout()
        return self._fig_to_base64(fig)

    def generate_html_report(self, output_filename: str):
        """
        生成并保存完整的HTML报告。
        """
        try:
            self._run_analyses()
        except Exception as e:
            logging.error(f"❌ 在运行分析时发生致命错误: {e}", exc_info=True)
            return  # 无法继续生成报告

        # 【【【修改】】】
        logging.info(f"  > ⚙️ 正在为 '{self.factor_name}' 生成 HTML 报告...")

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

        IC_STATS_DISPLAY_MAP = {
            'ic_mean': 'IC 均值',
            'ic_std': 'IC 标准差',
            'ir': '信息比率 (IR)',
            'ic_gt_0_prob': 'IC > 0 概率',
            'ic_abs_gt_0.02_prob': 'IC 绝对值 > 0.02 概率',
            'ic_t_stat': 'IC T-Statistic'
        }

        for p in self.periods:
            try:
                html += f"<div class='section'><h2>分析周期: {p}个交易日</h2>"

                # IC 统计表格
                ic_stats_dict = self.results[p]['ic_stats']
                ic_stats_df = pd.DataFrame.from_dict(ic_stats_dict,
                                                     orient='index',
                                                     columns=['Value'])

                ic_stats_df.index.name = "Metric"
                ic_stats_df = ic_stats_df.rename(
                    index=lambda x: IC_STATS_DISPLAY_MAP.get(x, x))

                html += "<h3>IC 统计摘要</h3>"
                html += ic_stats_df.to_html()

                # IC 图表
                ic_plot_b64 = self._plot_ic_summary(p)
                html += "<div class='plot'><h3>IC 序列与分布</h3>"
                html += f"<img src='{ic_plot_b64}'></div>"

                # ==============================================================================
                # 【【【【【【 HTML 块：新增的图表 】】】】】】
                # ==============================================================================

                # 新增图表：百分位收益率图
                rank_return_plot_b64 = self._plot_rank_return_scatter(p)
                html += "<div class='plot'><h3>因子百分位排名 vs. 收益率 (Hexbin)</h3>"
                html += """
                <p style="text-align:left; font-size: 0.9em; color: #555;">
                此图显示了所有股票在所有日期上的 <b>因子百分位排名 (X轴)</b> 与 <b>远期收益率 (Y轴)</b> 之间的关系。<br>
                图中的颜色表示该区域的数据点密度（颜色越亮，密度越高）。红色虚线是所有数据点的线性回归趋势线。
                </p>
                """
                html += f"<img src='{rank_return_plot_b64}'></div>"

                # ==============================================================================

                # (保留) 分层收益图表 (这个可能会显示失败)
                quantile_plot_b64 = self._plot_quantile_returns(p)
                html += "<div class='plot'><h3>因子分层收益 (旧)</h3>"
                html += f"<img src='{quantile_plot_b64}'></div>"

                # 累计收益图表
                cumulative_plot_b64 = self._plot_cumulative_factor_return(p)
                html += "<div class='plot'><h3>多空组合累计收益</h3>"
                html += f"<img src='{cumulative_plot_b64}'></div>"

                html += "</div>"
            except Exception as e:
                logging.error(f"❌ 在生成周期 {p}d 的图表时出错: {e}", exc_info=True)
                html += f"<div class='section'><h2>分析周期: {p}个交易日</h2>"
                html += f"<p style='color:red;'><b>报告生成失败: {e}</b></p></div>"

        html += """
            </div>
        </body>
        </html>
        """

        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(html)
            # 【【【修改】】】
            logging.info(f"  > ✅ HTML 报告已成功保存到: {output_filename}")
        except Exception as e:
            logging.error(f"❌ 无法将 HTML 报告写入文件: {e}", exc_info=True)
