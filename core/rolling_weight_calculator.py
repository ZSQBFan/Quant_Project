# factor_analysis/rolling_weight_calculator.py (已重构)
import pandas as pd
import numpy as np
from . import analysis_metrics as metrics
from typing import Dict, List
from sklearn.linear_model import LinearRegression
import logging  # <- 【【【新增】】】


class RollingICIRCalculator:
    """
    【【【重构】】】 (原 RollingWeightCalculator)
    负责在回测过程中动态计算【基于 IC / IR】的因子权重。
    
    【【重构日志】】:
    - 2025-11-09:
      - 引入 'logging' 模块，替换所有 'print' 语句。
    """

    def __init__(self, factor_weight_config: Dict[str, str],
                 forward_return_periods: List[int], factor_names: List[str]):
        """
        初始化 【IC/IR】 滚动权重计算器。
        """
        self.config = factor_weight_config
        self.periods = forward_return_periods
        self.factor_names = factor_names
        # 【【【修改】】】
        logging.info("--- RollingICIRCalculator 已初始化 ---")
        logging.info(f"    >  monitored_factors: {self.factor_names}")
        logging.info(f"    > weight_config: {self.config}")

    def _parse_metric_str(self, metric_str: str) -> (str, int):
        """
        辅助函数，解析配置字符串。
        """
        parts = metric_str.split('_')
        period_str = parts[-1]
        if period_str.endswith('d'):
            try:
                period = int(period_str[:-1])
                metric_key = '_'.join(parts[:-1])
                if metric_key == '':
                    metric_key = 'ir'
                return metric_key, period
            except ValueError:
                pass

        default_period = self.periods[0]
        # 【【【修改】】】
        logging.warning(
            f"  > ⚠️  指标 '{metric_str}' 未指定周期或格式错误, 自动使用 {default_period}d")
        return metric_str.replace(f"_{default_period}d", ""), default_period

    def calculate_new_weights(
            self, historical_data_window: pd.DataFrame) -> Dict[str, float]:
        """
        在给定的历史数据窗口上计算所有因子的新【IC/IR】权重。
        """
        logging.debug("  > ⚙️ [RollingICIR] 正在计算新权重...")
        raw_weights = {}

        for factor_name in self.factor_names:
            if factor_name not in self.config:
                # 【【【修改】】】
                logging.warning(
                    f"  > ⚠️ [RollingICIR] 因子 {factor_name} 未在权重配置中, 权重设为 0")
                raw_weights[factor_name] = 0.0
                continue

            metric_str = self.config[factor_name]
            metric_key, period = self._parse_metric_str(metric_str)

            if period not in self.periods:
                # 【【【修改】】】
                logging.error(
                    f"  > ❌ [RollingICIR] 无法计算 '{metric_str}'。周期 {period}d 未在"
                    f" forward_return_periods={self.periods} 中定义。")
                raw_weights[factor_name] = 0.0
                continue

            return_col = f'forward_return_{period}d'

            if factor_name not in historical_data_window.columns:
                # 【【【修改】】】
                logging.warning(
                    f"  > ⚠️ [RollingICIR] 因子 {factor_name} 在历史数据中不存在, 权重设为 0")
                raw_weights[factor_name] = 0.0
                continue

            ic_calc_data = historical_data_window[[factor_name,
                                                   return_col]].copy()
            ic_calc_data.rename(columns={factor_name: 'factor_value'},
                                inplace=True)
            ic_calc_data.dropna(inplace=True)

            if ic_calc_data.empty or len(ic_calc_data) < 10:
                # 【【【修改】】】
                logging.warning(
                    f"  > ⚠️ [RollingICIR] 因子 {factor_name} 在窗口内数据不足 ( {len(ic_calc_data)} < 10)，权重设为 0"
                )
                raw_weights[factor_name] = 0.0
                continue

            ic_series = metrics.calculate_rank_ic_series(ic_calc_data, period)

            if ic_series.empty or ic_series.std() == 0:
                logging.warning(
                    f"  > ⚠️ [RollingICIR] 因子 {factor_name} 的 IC 序列为空或标准差为0，权重设为 0"
                )
                raw_weights[factor_name] = 0.0
                continue

            stats = metrics.analyze_ic_statistics(ic_series)

            if metric_key in stats:
                raw_weights[factor_name] = stats[metric_key]
                logging.debug(
                    f"    > [RollingICIR] {factor_name}: {metric_key} = {stats[metric_key]:.4f}"
                )
            else:
                # 【【【修改】】】
                logging.warning(
                    f"  > ⚠️ [RollingICIR] 指标 '{metric_key}' 无法从 {stats.keys()} 中找到。"
                    f" 因子 {factor_name} 权重设为 0")
                raw_weights[factor_name] = 0.0

        # 归一化权重 (保留符号，基于绝对值归一化)
        total_strength = sum(abs(w) for w in raw_weights.values())

        if total_strength == 0:
            # 【【【修改】】】
            logging.warning("  > ⚠️ [RollingICIR] 滚动窗口内所有因子的IC/IR指标(绝对值)均为0。")
            return {factor: 0.0 for factor in self.factor_names}

        normalized_weights = {
            factor: weight / total_strength
            for factor, weight in raw_weights.items()
        }

        logging.debug(f"  > ✅ [RollingICIR] 新权重计算完毕: {normalized_weights}")
        return normalized_weights


class RollingRegressionCalculator:
    """
    【【【新增】】】
    负责在回测过程中动态计算【基于多元线性回归】的因子权重。
    
    【【重构日志】】:
    - 2025-11-09:
      - 引入 'logging' 模块，替换所有 'print' 语句。
    """

    def __init__(self, target_return_period: int, factor_names: List[str],
                 **kwargs):
        """
        初始化 【回归】 滚动权重计算器。
        """
        self.target_period = target_return_period
        self.return_col = f'forward_return_{self.target_period}d'
        self.factor_names = factor_names
        # 【【【修改】】】
        logging.info("--- RollingRegressionCalculator 已初始化 ---")
        logging.info(f"    > 监测的因子 (X): {self.factor_names}")
        logging.info(f"    > 目标收益 (Y): {self.return_col}")

    def calculate_new_weights(
            self, historical_data_window: pd.DataFrame) -> Dict[str, float]:
        """
        在给定的历史数据窗口上计算所有因子的新【回归】权重。
        """
        logging.debug("  > ⚙️ [RollingRegression] 正在计算新权重...")

        # 1. 准备数据
        for col in self.factor_names + [self.return_col]:
            if col not in historical_data_window.columns:
                # 【【【修改】】】
                logging.error(
                    f"  > ❌ [RollingRegression] 回归所需列 '{col}' 在历史数据中不存在。")
                return {factor: 0.0 for factor in self.factor_names}

        relevant_cols = self.factor_names + [self.return_col]
        data = historical_data_window[relevant_cols].dropna()

        min_samples_needed = len(self.factor_names) + 2
        if data.empty or len(data) < min_samples_needed:
            # 【【【修改】】】
            logging.warning(
                f"  > ⚠️ [RollingRegression] 滚动窗口内数据不足 (需 {min_samples_needed}，"
                f"实际 {len(data)})，无法执行回归。")
            return {factor: 0.0 for factor in self.factor_names}

        X = data[self.factor_names]
        y = data[self.return_col]

        # 2. 拟合回归
        try:
            model = LinearRegression()
            model.fit(X, y)
        except Exception as e:
            # 【【【修改】】】
            logging.error(f"  > ❌ [RollingRegression] 线性回归拟合失败: {e}",
                          exc_info=True)
            return {factor: 0.0 for factor in self.factor_names}

        # 3. 提取系数 (betas)
        # (保留符号)
        raw_weights = {
            name: coef
            for name, coef in zip(self.factor_names, model.coef_)
        }
        logging.debug(
            f"    > [RollingRegression] 原始回归系数 (Betas): {raw_weights}")

        # 4. 归一化权重 (保留符号，基于绝对值归一化)
        total_strength = sum(abs(w) for w in raw_weights.values())

        if total_strength == 0:
            # 【【【修改】】】
            logging.warning(
                "  > ⚠️ [RollingRegression] 滚动窗口内所有因子的回归系数(绝对值)均为0。")
            return {factor: 0.0 for factor in self.factor_names}

        normalized_weights = {
            factor: weight / total_strength
            for factor, weight in raw_weights.items()
        }

        logging.debug(
            f"  > ✅ [RollingRegression] 新权重计算完毕: {normalized_weights}")
        return normalized_weights
