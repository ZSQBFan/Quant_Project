"""
数据提供者基类
"""

import random
import pandas as pd
from typing import List, Optional


class BaseDataProvider:
    """
    数据提供者的基础抽象类。

    所有具体的数据提供者（Akshare, Tushare, SQLite等）都应继承此类。
    """

    def __init__(self, **kwargs):
        self.retries = kwargs.get('retries', 2)
        self.delay = kwargs.get('delay', 3 + random.uniform(-1.0, 1.0))

    def fetch_data(self, symbol: str, start_date: str,
                   end_date: str) -> pd.DataFrame | None:
        """
        获取指定股票在指定时间范围内的数据。

        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            包含 OHLCV 数据的 DataFrame，如果失败则返回 None
        """
        raise NotImplementedError("每个数据提供者子类都必须实现 fetch_data 方法。")

    def get_all_symbols(self, target_date: Optional[str] = None) -> List[str]:
        """
        获取全部股票代码列表。

        Args:
            target_date: 目标日期 (YYYY-MM-DD)，用于获取该时间点的股票列表
                        如果为 None，根据数据源特性自动选择：
                        - 外部数据源(Akshare/Tushare): 使用当前日期
                        - 数据库(SQLite): 使用目标日期以避免幸存者偏差

        Returns:
            股票代码列表

        Raises:
            NotImplementedError: 如果该数据提供者不支持此功能
        """
        raise NotImplementedError(f"{self.__class__.__name__} 不支持获取全部股票代码功能。")

    def get_all_stock_data(self, start_date: str, end_date: str, 
                          required_columns: List[str] = None) -> pd.DataFrame | None:
        """
        直接获取全股票数据（跳过股票代码获取步骤）
        
        优化功能：直接查询时间段内的数据，获取时间段内存在过的全部股票数据。
        后续因子计算程序支持处理NaN值，无需严格筛选数据完整性。
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            required_columns: 需要的数据列列表，如 ['open', 'close', 'volume'] 等
                           如果为 None，则获取所有可用列

        Returns:
            MultiIndex (date, asset) 的 DataFrame，包含所有股票在时间段内的数据
            如果失败则返回 None
            
        Note:
            - 对于数据库数据源(SQLite)：直接执行时间段查询，性能最优
            - 对于外部数据源(Akshare/Tushare)：可能需要先获取股票列表再分批查询
        """
        raise NotImplementedError(f"{self.__class__.__name__} 不支持直接获取全股票数据功能。")
