# factor_standardizer.py
import pandas as pd


class BaseStandardizer:
    """
    因子标准化器的基类。
    所有标准化器都应继承此类并实现 standardize 方法。
    """

    def __init__(self, **kwargs):
        # 此处可以处理未来可能需要的参数
        pass

    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        对输入的原始信号DataFrame进行标准化处理。

        Args:
            raw_signals_df (pd.DataFrame): 
                - 索引 (index): 股票代码 (或 data feed 对象)
                - 列 (columns): 不同的原始信号名称 (如 'MomentumSignal', 'RSISignal')
                - 值 (values): 原始信号值

        Returns:
            pd.DataFrame: 标准化后的信号DataFrame，结构与输入相同。
        """
        raise NotImplementedError("每个标准化器子类都必须实现 standardize 方法")


class NoStandardizer(BaseStandardizer):
    """
    “无操作”标准化器，直接返回原始信号值。
    适用于信号本身已经具有可比性，或希望在决策逻辑中直接使用原始值的场景。
    """

    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        #print("--- 执行标准化: 无 (使用原始值) ---")
        return raw_signals_df


class CrossSectionalZScoreStandardizer(BaseStandardizer):
    """
    截面Z-Score标准化器。
    在每个时间点上，对所有股票的同一个因子值计算其Z-Score。
    这使得不同因子的值可以在同一尺度下进行比较和合成。
    计算公式: z = (x - mean) / std
    """

    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        #print("--- 执行标准化: 截面 Z-Score ---")
        # 使用 a-xis=0 意味着沿着每一列（即每个因子）对所有行（股票）进行计算
        mean = raw_signals_df.mean(axis=0)
        std = raw_signals_df.std(axis=0)

        # 防止除以零
        std[std == 0] = 1.0

        standardized_df = (raw_signals_df - mean) / std
        return standardized_df.fillna(0)  # 处理计算中可能出现的NaN


class CrossSectionalQuantileStandardizer(BaseStandardizer):
    """
    截面分位数（百分位）标准化器。
    在每个时间点上，将每个股票的因子值转换为其在所有股票中的百分位排名。
    例如，一个值为0.9的股票意味着它的因子表现在所有股票中超过了90%。
    最终将排名映射到 [-1, 1] 的区间。
    """

    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        #print("--- 执行标准化: 截面分位数排名 ---")
        # .rank(pct=True) 直接计算百分位排名 (0.0 到 1.0)
        # na_option='bottom' 确保无效值排名最低
        percentile_ranks = raw_signals_df.rank(method='average',
                                               pct=True,
                                               na_option='bottom')

        # 将 0.0-1.0 的排名映射到 -1.0 到 1.0 的区间
        standardized_df = (percentile_ranks * 2) - 1.0
        return standardized_df.fillna(0)
