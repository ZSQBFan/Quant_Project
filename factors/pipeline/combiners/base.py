"""因子合成器基类"""
import pandas as pd
from abc import ABC, abstractmethod

class BaseFactorCombiner(ABC):
    """因子合成器基类"""
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        """合成因子 - 子类必须实现"""
        pass
