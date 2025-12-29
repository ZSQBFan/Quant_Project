"""
复合因子基类

用于处理全量宽表数据的复合因子（跨股票向量化计算）
不进行行业中性化处理
"""

import pandas as pd
import logging


class ComplexFactorBase:
    """
    复合因子基类

    提供通用的全量宽表数据处理方法。
    复合因子不继承 BaseFactor，因为它们使用不同的计算范式（全量宽表）。
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        """
        计算因子（子类必须实现）

        Args:
            all_data_df: 全量宽表数据 (MultiIndex: date, asset)

        Returns:
            因子值 Series
        """
        raise NotImplementedError("子类必须实现 calculate 方法")
