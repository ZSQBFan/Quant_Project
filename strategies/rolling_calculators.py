# strategies/rolling_calculators.py

import pandas as pd
import logging
from typing import Any, Dict, List
from sklearn.linear_model import LinearRegression
from core.abstractions import RollingCalculatorBase, AITrainerBase
from factor_analysis import analysis_metrics as metrics


class RollingICIRCalculator(RollingCalculatorBase):

    def __init__(self, factor_weight_config: Dict[str, str],
                 forward_return_periods: List[int], **kwargs):
        super().__init__(**kwargs)
        self.config, self.periods = factor_weight_config, forward_return_periods

    def _parse_metric_str(self, s: str) -> (str, int):
        p = s.split('_')[-1]
        k = '_'.join(s.split('_')[:-1]) or 'ir'
        if p.endswith('d'): return k, int(p[:-1])
        return s, self.periods[0]

    def _calculate_payload_for_day(self,
                                   hist_df: pd.DataFrame) -> Dict[str, float]:
        w = {}
        for fname in self.factor_names:
            m_str = self.config.get(fname, f'ir_{self.periods[0]}d')
            m_key, p = self._parse_metric_str(m_str)
            ic_data = hist_df[[fname, f'forward_return_{p}d'
                               ]].rename(columns={fname: 'factor_value'})
            ic_s = metrics.calculate_rank_ic_series(ic_data.dropna(), p)
            w[fname] = metrics.analyze_ic_statistics(ic_s).get(m_key, 0.0)
        tot_s = sum(abs(v) for v in w.values())
        return {f: v / tot_s if tot_s else 0.0 for f, v in w.items()}

    def _combine_factors_for_day(self, payload: Dict[str, float],
                                 daily_df: pd.DataFrame) -> pd.Series:
        weights = pd.Series(payload).reindex(daily_df.columns, fill_value=0)
        return (daily_df * weights).sum(axis=1)


class RollingRegressionCalculator(RollingCalculatorBase):

    def __init__(self, target_return_period: int, **kwargs):
        super().__init__(**kwargs)
        self.return_col = f'forward_return_{target_return_period}d'

    def _calculate_payload_for_day(self,
                                   hist_df: pd.DataFrame) -> Dict[str, float]:
        data = hist_df[self.factor_names + [self.return_col]].dropna()
        if len(data) < len(self.factor_names) + 2:
            return {f: 0.0 for f in self.factor_names}
        X, y = data[self.factor_names], data[self.return_col]
        model = LinearRegression().fit(X, y)
        w = {n: c for n, c in zip(self.factor_names, model.coef_)}
        tot_s = sum(abs(v) for v in w.values())
        return {f: v / tot_s if tot_s else 0.0 for f, v in w.items()}

    def _combine_factors_for_day(self, payload: Dict[str, float],
                                 daily_df: pd.DataFrame) -> pd.Series:
        weights = pd.Series(payload).reindex(daily_df.columns, fill_value=0)
        return (daily_df * weights).sum(axis=1)


class RollingAITrainer(RollingCalculatorBase):

    def __init__(self, training_calculator: AITrainerBase, **kwargs):
        super().__init__(**kwargs)
        self.trainer = training_calculator

    def _calculate_payload_for_day(self, hist_df: pd.DataFrame) -> Any:
        return self.trainer.train_model(hist_df, self.factor_names)

    def _combine_factors_for_day(self, payload: Any,
                                 daily_df: pd.DataFrame) -> pd.Series:
        model = payload
        try:
            features = model.feature_name_ if hasattr(
                model, 'feature_name_') else daily_df.columns
            X_today = daily_df[features]
            return pd.Series(model.predict(X_today), index=X_today.index)
        except Exception as e:
            logging.error(f"  > ❌ [RollingAITrainer] 预测失败: {e}")
            return None
