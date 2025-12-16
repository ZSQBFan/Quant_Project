"""
因子基类

定义因子的标准接口和通用功能。
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseFactor(ABC):
    """
    因子基类

    所有因子必须继承此类并实现 calculate 方法。

    设计原则：
    - 单一职责：每个因子只计算一个指标
    - 参数化：通过 params 传递所有参数
    - 容错性：处理异常数据
    """

    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化因子

        Args:
            params: 因子参数字典
        """
        self.params = params or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        计算因子值

        Args:
            df: 包含所需列的 DataFrame（如 close, high, low, volume 等）

        Returns:
            因子值 Series
        """
        raise NotImplementedError("子类必须实现 calculate 方法")

    def validate_input(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        验证输入数据

        Args:
            df: 输入 DataFrame
            required_columns: 必需的列名列表

        Returns:
            验证是否通过
        """
        if df is None or df.empty:
            self.logger.warning("输入数据为空")
            return False

        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            self.logger.error(f"缺少必需的列: {missing_columns}")
            return False

        return True

    def safe_divide(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """
        安全除法（避免除零）

        Args:
            numerator: 分子
            denominator: 分母

        Returns:
            除法结果，分母为0的位置填充 NaN
        """
        safe_denom = denominator.replace(0, np.nan)
        return numerator / safe_denom

    def __repr__(self):
        return f"{self.__class__.__name__}(params={self.params})"
