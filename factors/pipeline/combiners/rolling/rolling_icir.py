"""
滚动 ICIR 计算器

基于滚动窗口计算各因子的 IC/IR，并据此动态调整因子权重。
"""

import pandas as pd
import logging
from typing import Dict, List

from factors.core.abstractions import RollingCalculatorBase
from factors.analysis import metrics


class RollingICIRCalculator(RollingCalculatorBase):
    """
    滚动 ICIR 计算器。

    根据因子在滚动窗口内的 IC/IR 表现动态调整因子权重。
    """

    def __init__(self, factor_weight_config: Dict[str, str],
                 forward_return_periods: List[int], **kwargs):
        """
        初始化滚动 ICIR 计算器。

        Args:
            factor_weight_config: 因子权重配置，键为因子名，值为指标字符串（如 'ir_5d'）
            forward_return_periods: 远期收益计算周期列表
        """
        super().__init__(**kwargs)
        self.config = factor_weight_config
        self.periods = forward_return_periods

    def _parse_metric_str(self, s: str) -> tuple:
        """
        解析指标字符串。

        Args:
            s: 指标字符串，如 'ir_5d'

        Returns:
            (指标名, 周期) 元组
        """
        p = s.split('_')[-1]
        k = '_'.join(s.split('_')[:-1]) or 'ir'
        if p.endswith('d'):
            return k, int(p[:-1])
        return s, self.periods[0]

    def _calculate_payload_for_day(self,
                                   hist_df: pd.DataFrame,
                                   current_date: pd.Timestamp) -> Dict[str, float]:
        """
        计算单日的因子权重。

        Args:
            hist_df: 历史数据窗口
            current_date: 当前计算日期
        """
        # 添加数据质量检查
        if hist_df.empty:
            logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingICIRCalculator] 历史数据窗口为空，返回等权权重")
            return {f: 1.0/len(self.factor_names) for f in self.factor_names}

        # 检查每个因子的数据质量
        valid_factors = []
        for fname in self.factor_names:
            m_str = self.config.get(fname, f'ir_{self.periods[0]}d')
            m_key, p = self._parse_metric_str(m_str)
            return_col = f'forward_return_{p}d'

            if return_col not in hist_df.columns:
                logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingICIRCalculator] 缺少 {return_col} 列，跳过因子 {fname}")
                continue

            factor_data = hist_df[[fname, return_col]].dropna()
            if len(factor_data) < 10:  # 至少需要10个数据点进行IC计算
                logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingICIRCalculator] 因子 {fname} 数据点不足({len(factor_data)})，跳过")
                continue

            # 检查因子值是否为常量
            if factor_data[fname].nunique() <= 1:
                logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingICIRCalculator] 因子 {fname} 值为常量，跳过")
                continue

            # 检查收益率是否为常量
            if factor_data[return_col].nunique() <= 1:
                logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingICIRCalculator] 收益率 {return_col} 为常量，跳过因子 {fname}")
                continue

            valid_factors.append((fname, m_key, p))

        # 如果没有有效因子，返回等权权重
        if not valid_factors:
            logging.warning(f"  > ⚠️ [{current_date.date()}] [RollingICIRCalculator] 没有有效因子，返回等权权重")
            return {f: 1.0/len(self.factor_names) for f in self.factor_names}

        # 计算有效因子的权重
        w = {}
        for fname, m_key, p in valid_factors:
            return_col = f'forward_return_{p}d'
            ic_data = hist_df[[fname, return_col]].rename(columns={fname: 'factor_value'})
            try:
                ic_s = metrics.calculate_rank_ic_series(ic_data.dropna(), p)
                w[fname] = metrics.analyze_ic_statistics(ic_s).get(m_key, 0.0)
            except Exception as e:
                logging.warning(f"  > ⚠️ [RollingICIRCalculator] 计算因子 {fname} 的IC时出错: {e}，权重设为0")
                w[fname] = 0.0

        # 对于无效因子，权重设为0
        for fname in self.factor_names:
            if fname not in w:
                w[fname] = 0.0

        tot_s = sum(abs(v) for v in w.values())
        return {f: v / tot_s if tot_s else 0.0 for f, v in w.items()}

    def _combine_factors_for_day(self, payload: Dict[str, float],
                                 daily_df: pd.DataFrame) -> pd.Series:
        """
        根据权重合成因子。

        Args:
            payload: 因子权重字典
            daily_df: 当日因子值 DataFrame

        Returns:
            合成后的因子值 Series
        """
        weights = pd.Series(payload).reindex(daily_df.columns, fill_value=0)
        return (daily_df * weights).sum(axis=1)
