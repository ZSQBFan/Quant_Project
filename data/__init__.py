"""
数据层模块

提供统一的数据管理、数据提供者和交易日历功能。
"""

from .manager import DataProviderManager
from .calendar import (
    BaseTradingCalendar,
    TushareTradingCalendar,
    AkshareTradingCalendar,
    CachedTradingCalendar,
    create_trading_calendar,
    DataIntegrityChecker
)
from .providers import (
    BaseDataProvider,
    AkshareDataProvider,
    TushareDataProvider,
    SQLiteDataProvider
)
from .handlers import DatabaseHandler

__all__ = [
    # 数据管理器
    'DataProviderManager',

    # 交易日历
    'BaseTradingCalendar',
    'TushareTradingCalendar',
    'AkshareTradingCalendar',
    'CachedTradingCalendar',
    'create_trading_calendar',
    'DataIntegrityChecker',

    # 数据提供者
    'BaseDataProvider',
    'AkshareDataProvider',
    'TushareDataProvider',
    'SQLiteDataProvider',

    # 数据库处理
    'DatabaseHandler',
]
