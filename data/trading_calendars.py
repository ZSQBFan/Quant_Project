# trading_calendars.py
import pandas as pd
import akshare as ak
import tushare as ts
from typing import List
import random
import time


class BaseTradingCalendar:
    """
    交易日历获取器基类。
    所有具体实现需继承此类并实现 get_trading_days 方法。
    """

    def __init__(self, **kwargs):
        pass

    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """
        获取指定日期范围内的交易日列表。
        返回格式为 ['YYYY-MM-DD', ...]
        """
        raise NotImplementedError("子类必须实现 get_trading_days 方法")


class TushareTradingCalendar(BaseTradingCalendar):
    """
    使用 Tushare 获取 A 股交易日历。
    """

    def __init__(self, token: str, **kwargs):
        super().__init__(**kwargs)
        self.pro = ts.pro_api(token)

    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        df = self.pro.trade_cal(exchange='SSE',
                                start_date=start_date.replace('-', ''),
                                end_date=end_date.replace('-', ''),
                                is_open='1')
        if df.empty:
            return []
        df['cal_date'] = pd.to_datetime(df['cal_date'], format='%Y%m%d')
        df = df.sort_values('cal_date')
        return df['cal_date'].dt.strftime('%Y-%m-%d').tolist()


class AkshareTradingCalendar(BaseTradingCalendar):
    """
    使用 Akshare 获取 A 股交易日历。
    """

    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        df = ak.tool_trade_date_hist_sina()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        mask = (df['trade_date'] >= pd.to_datetime(start_date)) & (
            df['trade_date'] <= pd.to_datetime(end_date))
        df_filtered = df[mask].sort_values('trade_date')
        return df_filtered['trade_date'].dt.strftime('%Y-%m-%d').tolist()


class DataIntegrityChecker:
    """
    数据完整性检查器。
    在下载前检查数据库中是否已包含指定时间段内所有交易日的数据。
    """

    def __init__(self, db_handler, calendar_provider: BaseTradingCalendar):
        self.db_handler = db_handler
        self.calendar_provider = calendar_provider

    def check_symbol_integrity(self, symbol: str, start_date: str,
                               end_date: str) -> bool:
        """
        检查单个标的在指定日期范围内是否数据完整。
        """
        trading_days = set(
            self.calendar_provider.get_trading_days(start_date, end_date))
        if not trading_days:
            print(f"  ⚠️  无法获取 {start_date} 至 {end_date} 的交易日历，跳过完整性检查。")
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
            print(f"  📥 [{symbol}] 缺失 {len(missing_days)} 个交易日数据，需要下载。")
            return False
        else:
            print(f"  ✅ [{symbol}] 数据完整，跳过下载。")
            return True


# --- Content from data_providers.py ---
class BaseDataProvider:
    """
    【重构】数据提供者的基础抽象类。
    """

    def __init__(self, **kwargs):
        self.retries = kwargs.get('retries', 2)
        self.delay = kwargs.get('delay', 3 + random.uniform(-1.0, 1.0))

    def fetch_data(self, symbol: str, start_date: str,
                   end_date: str) -> pd.DataFrame | None:
        raise NotImplementedError("每个数据提供者子类都必须实现 fetch_data 方法。")


class AkshareDataProvider(BaseDataProvider):
    """
    使用 Akshare 作为数据源的具体实现。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adjust = kwargs.get('adjust', "hfq")

    def fetch_data(self, symbol: str, start_date: str,
                   end_date: str) -> pd.DataFrame | None:
        print(
            f"  [Akshare尝试] 正在获取 {symbol} 从 {start_date} 到 {end_date} 的数据...")
        for attempt in range(self.retries):
            try:
                if symbol.startswith('sh') or symbol.startswith('sz'):
                    df_raw = ak.stock_zh_index_daily(symbol=symbol)
                else:
                    df_raw = ak.stock_zh_a_hist(
                        symbol=symbol,
                        start_date=start_date.replace('-', ''),
                        end_date=end_date.replace('-', ''),
                        adjust=self.adjust)

                if df_raw is None or df_raw.empty or '日期' not in df_raw.columns:
                    print(
                        f"  🟡 [Akshare警告] 在 {start_date} - {end_date} 范围内未返回 '{symbol}' 的有效数据。"
                    )
                    return None

                ak_columns = [
                    '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅',
                    '涨跌额', '换手率'
                ]
                db_columns = [
                    'date', 'open', 'close', 'high', 'low', 'volume',
                    'turnover', 'amplitude', 'pct_change', 'price_change',
                    'turnover_rate'
                ]
                df_raw.rename(columns=dict(zip(ak_columns, db_columns)),
                              inplace=True)

                for col in db_columns:
                    if col not in df_raw.columns:
                        df_raw[col] = None

                df = df_raw[db_columns].copy()
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

                numeric_cols = [
                    'open', 'close', 'high', 'low', 'turnover', 'amplitude',
                    'pct_change', 'price_change', 'turnover_rate'
                ]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['volume'] = pd.to_numeric(
                    df['volume'], errors='coerce').fillna(0).astype(int)

                df_final = df.loc[start_date:end_date]
                if not df_final.empty:
                    print(
                        f"  ✅ [Akshare成功] 成功获取 {symbol} 的 {len(df_final)} 条数据。"
                    )
                    return df_final
                else:
                    return None

            except Exception as e:
                print(
                    f"  ❌ [Akshare错误] 获取 {symbol} 数据时出错 (尝试 {attempt + 1}/{self.retries}): {e}"
                )
                if attempt < self.retries - 1:
                    print(f"    将在 {self.delay} 秒后重试...")
                    time.sleep(self.delay + random.uniform(0, 1))
                else:
                    print(f"  ❌ [Akshare失败] 已达到最大重试次数，放弃使用 Akshare 获取该数据。")
                    return None
        return None


class TushareDataProvider(BaseDataProvider):
    """
    使用 Tushare 作为数据源的具体实现。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.token = kwargs.get('token')
        if not self.token:
            raise ValueError("TushareDataProvider 需要一个有效的 'token' 参数。")
        self.pro = ts.pro_api(self.token)
        self.adjust = kwargs.get('adjust', "hfq")

    def _convert_symbol_to_ts_code(self, symbol):
        if symbol.startswith('sh') or symbol.startswith('sz'):
            return f"{symbol.replace('sh', '').replace('sz', '')}.SH" if symbol.startswith(
                'sh') else f"{symbol.replace('sh', '').replace('sz', '')}.SZ"
        return f"{symbol}.SH" if symbol.startswith('6') else f"{symbol}.SZ"

    def fetch_data(self, symbol: str, start_date: str,
                   end_date: str) -> pd.DataFrame | None:
        print(
            f"  [Tushare尝试] 正在获取 {symbol} 从 {start_date} 到 {end_date} 的数据...")
        ts_code = self._convert_symbol_to_ts_code(symbol)

        for attempt in range(self.retries):
            try:
                df_raw = self.pro.daily(ts_code=ts_code,
                                        start_date=start_date.replace('-', ''),
                                        end_date=end_date.replace('-', ''))

                if df_raw is None or df_raw.empty:
                    print(
                        f"  🟡 [Tushare警告] 在 {start_date} - {end_date} 范围内未返回 '{symbol}' 的有效数据。"
                    )
                    return None

                adj_factor = self.pro.adj_factor(
                    ts_code=ts_code,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', ''))

                if not adj_factor.empty:
                    df_raw = pd.merge(df_raw,
                                      adj_factor[['trade_date', 'adj_factor']],
                                      on='trade_date',
                                      how='left')
                    df_raw['adj_factor'] = df_raw['adj_factor'].ffill()

                    for col in ['open', 'high', 'low', 'close']:
                        df_raw[col] = df_raw[col] * df_raw['adj_factor']

                ts_columns = [
                    'trade_date', 'open', 'close', 'high', 'low', 'vol',
                    'amount', 'pct_chg', 'change'
                ]
                db_columns = [
                    'date', 'open', 'close', 'high', 'low', 'volume',
                    'turnover', 'pct_change', 'price_change'
                ]
                df_raw.rename(columns=dict(zip(ts_columns, db_columns)),
                              inplace=True)

                df_raw['volume'] = pd.to_numeric(
                    df_raw['volume'],
                    errors='coerce').fillna(0).astype(int) * 100
                df_raw['turnover'] = pd.to_numeric(
                    df_raw['turnover'],
                    errors='coerce').fillna(0).astype(float) * 1000

                full_db_columns = [
                    'date', 'open', 'close', 'high', 'low', 'volume',
                    'turnover', 'amplitude', 'pct_change', 'price_change',
                    'turnover_rate'
                ]
                for col in full_db_columns:
                    if col not in df_raw.columns:
                        df_raw[col] = None

                df = df_raw[full_db_columns].copy()
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(ascending=True, inplace=True)

                print(f"  ✅ [Tushare成功] 成功获取 {symbol} 的 {len(df)} 条数据。")
                return df

            except Exception as e:
                print(
                    f"  ❌ [Tushare错误] 获取 {symbol} 数据时出错 (尝试 {attempt + 1}/{self.retries}): {e}"
                )
                if attempt < self.retries - 1:
                    print(f"    将在 {self.delay} 秒后重试...")
                    time.sleep(self.delay + random.uniform(0, 1))
                else:
                    print(f"  ❌ [Tushare失败] 已达到最大重试次数，放弃使用 Tushare 获取该数据。")
                    return None
        return None
