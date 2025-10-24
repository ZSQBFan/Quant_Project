# factor_combiner.py
import pandas as pd
import numpy as np


class BaseFactorCombiner:
    """
    因子合成器的基类。
    所有合成器都应继承此类并实现 combine 方法。
    """

    def __init__(self, **kwargs):
        # 此处可以处理未来可能需要的参数，例如不同因子的权重
        pass

    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        """
        将一个包含多个标准化因子值的DataFrame合成为一个综合得分Series。

        Args:
            standardized_df (pd.DataFrame):
                - 索引 (index): 股票的 data feed 对象
                - 列 (columns): 不同的标准化信号名称
                - 值 (values): 标准化后的信号值

        Returns:
            pd.Series:
                - 索引 (index): 股票的 data feed 对象
                - 值 (values): 每只股票的最终综合得分
        """
        raise NotImplementedError("每个合成器子类都必须实现 combine 方法")


class EqualWeightCombiner(BaseFactorCombiner):
    """
    等权重合成器。
    这是最简单的合成方法，直接将每只股票的所有标准化因子值相加，
    得到最终的综合得分。
    """

    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        """
        通过对所有因子得分求和来实现等权重合成。
        """
        #print("--- 执行因子合成: 等权重相加 ---")
        # axis=1 表示沿着列的方向（即对每个股票的所有因子）进行求和
        return standardized_df.sum(axis=1)


class DynamicSignificanceCombiner(BaseFactorCombiner):
    """
    动态显著性加权合成器。

    该合成器根据每个因子在特定时间点的相对“显著性”（即标准化后的绝对值大小）
    来动态地分配权重。信号越强的因子，获得的权重越高。
    """

    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        """
        执行动态显著性加权合成。

        Args:
            standardized_df (pd.DataFrame): 经过标准化的因子数据。
                                            索引为股票代码，列为不同因子。

        Returns:
            pd.Series: 每只股票的最终综合得分。
        """
        # 1. 计算每个因子的绝对值，代表其“显著性”或“信号强度”
        abs_significance = standardized_df.abs()

        # 2. 计算每行（每只股票）的“总显著性”
        #    为了防止所有因子值都为0导致除以0的错误，我们用一个很小的数 epsilon 替代0
        total_significance = abs_significance.sum(axis=1)
        total_significance.replace(0, np.finfo(float).eps, inplace=True)

        # 3. 计算每个因子的动态权重
        #    权重 = 自身显著性 / 总显著性
        dynamic_weights = abs_significance.div(total_significance, axis=0)

        # 4. 使用动态权重对原始的（带符号的）标准化因子值进行加权求和
        #    (standardized_df * dynamic_weights) 实现了元素级的乘法
        combined_score = (standardized_df * dynamic_weights).sum(axis=1)

        return combined_score


# 未来可以轻松在这里添加更多合成器，例如：
# class ICWeightCombiner(BaseFactorCombiner):
#     def __init__(self, factor_weights, **kwargs):
#         super().__init__(**kwargs)
#         self.factor_weights = factor_weights # 接收一个包含因子权重的字典
#
#     def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
#         print("--- 执行因子合成: IC加权 ---")
#         # 确保权重和DataFrame的列对齐
#         aligned_weights = pd.Series(self.factor_weights).reindex(standardized_df.columns).fillna(0)
#         # 执行加权求和
#         return (standardized_df * aligned_weights).sum(axis=1)
