# core/factor_standardizer.py (已重构)
import pandas as pd
import logging  # <- 【【【新增】】】


class BaseStandardizer:
    """
    因子标准化器的基类。
    
    【【重构日志】】:
    - 2025-11-09:
      - 引入 'logging' 模块。
      - 增加了 DEBUG 级别的日志调用。
    """

    def __init__(self, **kwargs):
        pass

    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("每个标准化器子类都必须实现 standardize 方法")


class NoStandardizer(BaseStandardizer):
    """
    “无操作”标准化器，直接返回原始信号值。
    """

    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        # 【【【新增】】】 (Debug 级别，默认不显示)
        logging.debug("  > ⚙️ [Standardizer] 执行标准化: 无 (使用原始值)")
        return raw_signals_df


class CrossSectionalZScoreStandardizer(BaseStandardizer):
    """
    截面Z-Score标准化器。
    计算公式: z = (x - mean) / std
    """

    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        # 【【【新增】】】
        logging.debug("  > ⚙️ [Standardizer] 执行标准化: 截面 Z-Score")

        mean = raw_signals_df.mean(axis=0)
        std = raw_signals_df.std(axis=0)

        std[std == 0] = 1.0

        standardized_df = (raw_signals_df - mean) / std
        return standardized_df.fillna(0)


class CrossSectionalQuantileStandardizer(BaseStandardizer):
    """
    截面分位数（百分位）标准化器。
    最终将排名映射到 [-1, 1] 的区间。
    """

    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        # 【【【新增】】】
        logging.debug("  > ⚙️ [Standardizer] 执行标准化: 截面分位数排名")

        percentile_ranks = raw_signals_df.rank(method='average',
                                               pct=True,
                                               na_option='bottom')

        standardized_df = (percentile_ranks * 2) - 1.0
        return standardized_df.fillna(0)
