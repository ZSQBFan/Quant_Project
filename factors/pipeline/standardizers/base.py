"""标准化器基类"""
import pandas as pd
from abc import ABC, abstractmethod

class BaseStandardizer(ABC):
    """标准化器基类"""
    @abstractmethod
    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        """标准化因子 - 子类必须实现"""
        pass
