"""
交易日历模块

提供 A 股交易日历获取功能，支持本地缓存。
"""

import pandas as pd
import akshare as ak
import tushare as ts
from typing import List, Optional
import logging
import json
import os
from datetime import datetime, timedelta
from pathlib import Path


class BaseTradingCalendar:
    """交易日历获取器基类。"""

    def __init__(self, **kwargs):
        pass

    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """
        获取指定日期范围内的交易日列表。

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            交易日期字符串列表
        """
        raise NotImplementedError("子类必须实现 get_trading_days 方法")


class TushareTradingCalendar(BaseTradingCalendar):
    """使用 Tushare 获取 A 股交易日历。"""

    def __init__(self, token: str, **kwargs):
        super().__init__(**kwargs)
        self.pro = ts.pro_api(token)
        logging.info("TushareTradingCalendar 已初始化。")

    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        try:
            df = self.pro.trade_cal(exchange='SSE',
                                    start_date=start_date.replace('-', ''),
                                    end_date=end_date.replace('-', ''),
                                    is_open='1')
            if df.empty:
                return []
            df['cal_date'] = pd.to_datetime(df['cal_date'], format='%Y%m%d')
            df = df.sort_values('cal_date')
            return df['cal_date'].dt.strftime('%Y-%m-%d').tolist()
        except Exception as e:
            logging.error(f"[Tushare日历] 获取交易日历失败: {e}", exc_info=True)
            return []


class AkshareTradingCalendar(BaseTradingCalendar):
    """使用 Akshare 获取 A 股交易日历。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logging.info("AkshareTradingCalendar 已初始化。")

    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        try:
            df = ak.tool_trade_date_hist_sina()
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            mask = (df['trade_date'] >= pd.to_datetime(start_date)) & (
                df['trade_date'] <= pd.to_datetime(end_date))
            df_filtered = df[mask].sort_values('trade_date')
            return df_filtered['trade_date'].dt.strftime('%Y-%m-%d').tolist()
        except Exception as e:
            logging.error(f"[Akshare日历] 获取交易日历失败: {e}", exc_info=True)
            return []


class CachedTradingCalendar(BaseTradingCalendar):
    """
    带缓存的交易日历包装器。

    包装任意日历提供者，添加本地 JSON 缓存功能。
    """

    def __init__(self,
                 inner_provider: BaseTradingCalendar,
                 cache_file: str = "./temp/trading_calendar.json",
                 cache_days: int = 30,
                 auto_update: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.inner_provider = inner_provider
        self.cache_file = Path(cache_file)
        self.cache_days = cache_days
        self.auto_update = auto_update
        self._cache = None
        self._cache_loaded = False

        # 确保缓存目录存在
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"CachedTradingCalendar 已初始化，缓存文件: {self.cache_file}")

    def _load_cache(self) -> dict:
        """加载缓存文件。"""
        if self._cache_loaded:
            return self._cache or {}

        self._cache_loaded = True

        if not self.cache_file.exists():
            self._cache = {}
            return self._cache

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._cache = data
                logging.info(f"已从缓存加载交易日历数据 (最后更新: {data.get('last_updated', 'unknown')})")
                return self._cache
        except Exception as e:
            logging.warning(f"加载缓存文件失败: {e}")
            self._cache = {}
            return self._cache

    def _save_cache(self):
        """保存缓存到文件。"""
        if self._cache is None:
            return

        try:
            self._cache['last_updated'] = datetime.now().isoformat()
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
            logging.info(f"交易日历缓存已保存到 {self.cache_file}")
        except Exception as e:
            logging.warning(f"保存缓存文件失败: {e}")

    def _is_cache_valid(self) -> bool:
        """检查缓存是否有效。"""
        cache = self._load_cache()

        if not cache or 'last_updated' not in cache:
            return False

        try:
            last_updated = datetime.fromisoformat(cache['last_updated'])
            age_days = (datetime.now() - last_updated).days

            if age_days >= self.cache_days:
                logging.info(f"缓存已过期 ({age_days} 天)")
                return False

            return True
        except Exception:
            return False

    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日列表，优先使用缓存。
        """
        cache = self._load_cache()
        cache_key = f"{start_date}_{end_date}"

        # 尝试从缓存获取
        if cache_key in cache.get('ranges', {}) and self._is_cache_valid():
            logging.debug(f"从缓存获取交易日历: {start_date} ~ {end_date}")
            return cache['ranges'][cache_key]

        # 如果有完整日历数据，从中过滤
        if 'all_trading_days' in cache and self._is_cache_valid():
            all_days = cache['all_trading_days']
            filtered = [d for d in all_days if start_date <= d <= end_date]
            if filtered:
                logging.debug(f"从缓存的完整日历过滤: {start_date} ~ {end_date}")
                return filtered

        # 从网络获取
        logging.info(f"从网络获取交易日历: {start_date} ~ {end_date}")
        trading_days = self.inner_provider.get_trading_days(start_date, end_date)

        # 更新缓存
        if trading_days:
            if 'ranges' not in cache:
                cache['ranges'] = {}
            cache['ranges'][cache_key] = trading_days
            self._cache = cache
            self._save_cache()

        return trading_days


def create_trading_calendar(config: dict, tushare_token: Optional[str] = None) -> BaseTradingCalendar:
    """
    根据配置创建交易日历实例。

    Args:
        config: calendar.yaml 中的 provider 配置
        tushare_token: Tushare token（可选）

    Returns:
        BaseTradingCalendar 实例
    """
    default_provider = config.get('default', 'akshare')
    cache_config = config.get('cache', {})

    # 创建内部提供者
    inner_provider = None

    if default_provider == 'tushare':
        tushare_config = config.get('tushare', {})
        token = tushare_token or os.getenv('TUSHARE_TOKEN')

        if token and tushare_config.get('enabled', True):
            try:
                inner_provider = TushareTradingCalendar(token=token)
                logging.info("使用 Tushare 作为日历提供者")
            except Exception as e:
                logging.warning(f"Tushare 日历初始化失败: {e}")

    if inner_provider is None and default_provider in ['akshare', 'tushare']:
        # Tushare 失败或默认使用 Akshare
        akshare_config = config.get('akshare', {})
        if akshare_config.get('enabled', True):
            inner_provider = AkshareTradingCalendar()
            logging.info("使用 Akshare 作为日历提供者")

    if inner_provider is None:
        logging.warning("无法创建日历提供者，使用默认 Akshare")
        inner_provider = AkshareTradingCalendar()

    # 如果启用缓存，包装提供者
    if cache_config.get('enabled', False):
        return CachedTradingCalendar(
            inner_provider=inner_provider,
            cache_file=cache_config.get('cache_file', './temp/trading_calendar.json'),
            cache_days=cache_config.get('cache_days', 30),
            auto_update=cache_config.get('auto_update', True)
        )

    return inner_provider


class DataIntegrityChecker:
    """数据完整性检查器。"""

    def __init__(self, db_handler, calendar_provider: BaseTradingCalendar):
        self.db_handler = db_handler
        self.calendar_provider = calendar_provider

    def check_symbol_integrity(self, symbol: str, start_date: str,
                               end_date: str) -> bool:
        """
        检查单个标的在指定日期范围内是否数据完整。

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            True 如果数据完整，False 如果需要下载
        """
        try:
            trading_days = set(
                self.calendar_provider.get_trading_days(start_date, end_date))
            if not trading_days:
                logging.warning(
                    f"无法获取 {start_date} 至 {end_date} 的交易日历，跳过完整性检查。")
                return False

            query = "SELECT DISTINCT date FROM stock_daily_prices WHERE code = ? AND DATE(date) BETWEEN ? AND ?"
            existing_dates_df = self.db_handler.query_data(query,
                                                           params=(symbol,
                                                                   start_date,
                                                                   end_date))
            existing_dates = set(existing_dates_df.index.strftime(
                '%Y-%m-%d')) if not existing_dates_df.empty else set()

            missing_days = trading_days - existing_dates
            if missing_days:
                logging.info(
                    f"[{symbol}] 缺失 {len(missing_days)} 个交易日数据，需要下载。")
                return False
            else:
                logging.info(f"[{symbol}] 数据完整，跳过下载。")
                return True
        except Exception as e:
            logging.error(f"在检查 '{symbol}' 数据完整性时出错: {e}", exc_info=True)
            return False
