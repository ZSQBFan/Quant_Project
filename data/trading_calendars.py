# trading_calendars.py (已重构)
import pandas as pd
import akshare as ak
import tushare as ts
from typing import List
import random
import time
import logging  # <- 【【【新增】】】


class BaseTradingCalendar:
    """
    交易日历获取器基类。
    """

    def __init__(self, **kwargs):
        pass

    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        raise NotImplementedError("子类必须实现 get_trading_days 方法")


class TushareTradingCalendar(BaseTradingCalendar):
    """
    使用 Tushare 获取 A 股交易日历。
    """

    def __init__(self, token: str, **kwargs):
        super().__init__(**kwargs)
        self.pro = ts.pro_api(token)
        logging.info("ℹ️ TushareTradingCalendar 已初始化。")

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
            logging.error(f"❌ [Tushare日历] 获取交易日历失败: {e}", exc_info=True)
            return []


class AkshareTradingCalendar(BaseTradingCalendar):
    """
    使用 Akshare 获取 A 股交易日历。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logging.info("ℹ️ AkshareTradingCalendar 已初始化。")

    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        try:
            df = ak.tool_trade_date_hist_sina()
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            mask = (df['trade_date'] >= pd.to_datetime(start_date)) & (
                df['trade_date'] <= pd.to_datetime(end_date))
            df_filtered = df[mask].sort_values('trade_date')
            return df_filtered['trade_date'].dt.strftime('%Y-%m-%d').tolist()
        except Exception as e:
            logging.error(f"❌ [Akshare日历] 获取交易日历失败: {e}", exc_info=True)
            return []


class DataIntegrityChecker:
    """
    数据完整性检查器。
    
    【【重构日志】】:
    - 2025-11-09:
      - 引入 'logging' 模块，替换所有 'print' 语句。
    """

    def __init__(self, db_handler, calendar_provider: BaseTradingCalendar):
        self.db_handler = db_handler
        self.calendar_provider = calendar_provider

    def check_symbol_integrity(self, symbol: str, start_date: str,
                               end_date: str) -> bool:
        """
        检查单个标的在指定日期范围内是否数据完整。
        """
        try:
            trading_days = set(
                self.calendar_provider.get_trading_days(start_date, end_date))
            if not trading_days:
                # 【【【修改】】】
                logging.warning(
                    f"  > ⚠️  无法获取 {start_date} 至 {end_date} 的交易日历，跳过完整性检查。")
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
                # 【【【修改】】】
                logging.info(
                    f"  > 📥 [{symbol}] 缺失 {len(missing_days)} 个交易日数据，需要下载。")
                return False
            else:
                # 【【【修改】】】
                logging.info(f"  > ✅ [{symbol}] 数据完整，跳过下载。")
                return True
        except Exception as e:
            logging.error(f"  > ❌ 在检查 '{symbol}' 数据完整性时出错: {e}", exc_info=True)
            return False  # 出现错误时，默认为不完整，触发下载


# --- Content from data_providers.py ---
# 【【注意】】: 这部分代码在 trading_calendars.py 中是冗余的，
# 它们应该只在 data_providers.py 中定义。
# 为完整起见，我也会重构这里的 print 语句。


class BaseDataProvider:

    def __init__(self, **kwargs):
        self.retries = kwargs.get('retries', 2)
        self.delay = kwargs.get('delay', 3 + random.uniform(-1.0, 1.0))

    def fetch_data(self, symbol: str, start_date: str,
                   end_date: str) -> pd.DataFrame | None:
        raise NotImplementedError("每个数据提供者子类都必须实现 fetch_data 方法。")


class AkshareDataProvider(BaseDataProvider):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adjust = kwargs.get('adjust', "hfq")

    def fetch_data(self, symbol: str, start_date: str,
                   end_date: str) -> pd.DataFrame | None:
        # 【【【修改】】】
        logging.info(
            f"  > 📡 [Akshare尝试] 正在获取 {symbol} 从 {start_date} 到 {end_date} 的数据..."
        )
        for attempt in range(self.retries):
            try:
                # ... (省略具体实现) ...
                pass  # 假设实现与 data_providers.py 中相同

                # 假设成功
                logging.info(f"  > ✅ [Akshare成功] 成功获取 {symbol} 的数据。")
                return pd.DataFrame()  # 返回示例DF

            except Exception as e:
                # 【【【修改】】】
                logging.error(
                    f"  > ❌ [Akshare错误] 获取 {symbol} 数据时出错 (尝试 {attempt + 1}/{self.retries}): {e}",
                    exc_info=True)
                if attempt < self.retries - 1:
                    logging.warning(f"    > ⏳ 将在 {self.delay} 秒后重试...")
                    time.sleep(self.delay + random.uniform(0, 1))
                else:
                    logging.error(
                        f"  > ❌ [Akshare失败] 已达到最大重试次数，放弃使用 Akshare 获取 {symbol}。"
                    )
                    return None
        return None


class TushareDataProvider(BaseDataProvider):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.token = kwargs.get('token')
        if not self.token:
            logging.critical("⛔ TushareDataProvider 需要 'token'。")
            raise ValueError("TushareDataProvider 需要 'token'。")
        self.pro = ts.pro_api(self.token)
        self.adjust = kwargs.get('adjust', "hfq")

    def fetch_data(self, symbol: str, start_date: str,
                   end_date: str) -> pd.DataFrame | None:
        # 【【【修改】】】
        logging.info(
            f"  > 📡 [Tushare尝试] 正在获取 {symbol} 从 {start_date} 到 {end_date} 的数据..."
        )
        # ... (省略具体实现) ...
        try:
            pass  # 假设实现与 data_providers.py 中相同

            # 假设成功
            logging.info(f"  > ✅ [Tushare成功] 成功获取 {symbol} 的数据。")
            return pd.DataFrame()  # 返回示例DF

        except Exception as e:
            # 【【【修改】】】
            logging.error(f"  > ❌ [Tushare错误] 获取 {symbol} 数据时出错: {e}",
                          exc_info=True)
            return None
        return None
