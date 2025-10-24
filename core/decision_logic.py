# decision_logic.py (已重构)
import backtrader as bt
import pandas as pd
from types import SimpleNamespace


class FactorRankingLogic:
    """基于因子排序的选股决策逻辑"""

    def __init__(self,
                 strategy,
                 top_n=10,
                 ranking_asc=False,
                 standardizer_config=None,
                 combiner_config=None):
        self.strategy = strategy
        self.p = type('Params', (), {
            'top_n': top_n,
            'ranking_asc': ranking_asc
        })()

        # ❌ 删除这一行
        # self.log = strategy.log

        # 初始化标准化器
        if standardizer_config:
            standardizer_class, standardizer_params = standardizer_config
            self.standardizer = standardizer_class(**standardizer_params)
        else:
            from core.factor_standardizer import NoStandardizer
            self.standardizer = NoStandardizer()

        # 初始化合成器
        if combiner_config:
            combiner_class, combiner_params = combiner_config
            self.combiner = combiner_class(**combiner_params)
        else:
            from core.factor_combiner import EqualWeightCombiner
            self.combiner = EqualWeightCombiner()

    def decide(self):
        # --- 步骤 1: 收集所有股票的原始信号 ---
        all_raw_signals = []
        for d in self.strategy.datas:
            d_name = d._name
            if len(d) > 0:
                signal_handlers = self.strategy.signal_handlers.get(d_name, [])
                if not signal_handlers:
                    continue
                stock_signals = {'data': d}
                for handler in signal_handlers:
                    signal_name = handler.__class__.__name__
                    stock_signals[signal_name] = handler.get_signal()
                all_raw_signals.append(stock_signals)

        if not all_raw_signals:
            self.strategy.log("没有足够的原始信号数据进行决策。")  # ✅ 改为这样
            return []

        raw_signals_df = pd.DataFrame(all_raw_signals).set_index('data')
        raw_signals_df.dropna(inplace=True)

        if raw_signals_df.empty:
            self.strategy.log("数据清洗后,没有可用的信号进行决策。")  # ✅ 改为这样
            return []

        # --- 步骤 2: 标准化 ---
        standardized_df = self.standardizer.standardize(raw_signals_df)

        # --- 步骤 3: 因子合成 ---
        composite_scores = self.combiner.combine(standardized_df)
        standardized_df['composite_score'] = composite_scores

        # --- 步骤 4: 排序与选股 ---
        standardized_df.sort_values(by='composite_score',
                                    ascending=self.p.ranking_asc,
                                    inplace=True)

        top_stocks_df = standardized_df.head(self.p.top_n)
        target_portfolio = top_stocks_df.index.tolist()

        self.strategy.log(  # ✅ 改为这样
            f"目标持仓 (Top {self.p.top_n}): {[d._name for d in target_portfolio]}"
        )

        return target_portfolio
