# data_providers.py
import pandas as pd
import akshare as ak
import tushare as ts
import time
import random
from tqdm import tqdm
from .database_handler import DatabaseHandler


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
        tqdm.write(
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
                    tqdm.write(
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
                    tqdm.write(
                        f"  ✅ [Akshare成功] 成功获取 {symbol} 的 {len(df_final)} 条数据。"
                    )
                    return df_final
                else:
                    return None

            except Exception as e:
                tqdm.write(
                    f"  ❌ [Akshare错误] 获取 {symbol} 数据时出错 (尝试 {attempt + 1}/{self.retries}): {e}"
                )
                if attempt < self.retries - 1:
                    tqdm.write(f"    将在 {self.delay} 秒后重试...")
                    time.sleep(self.delay + random.uniform(0, 1))
                else:
                    tqdm.write(
                        f"  ❌ [Akshare失败] 已达到最大重试次数，放弃使用 Akshare 获取该数据。")
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
        tqdm.write(
            f"  [Tushare尝试] 正在获取 {symbol} 从 {start_date} 到 {end_date} 的数据...")
        ts_code = self._convert_symbol_to_ts_code(symbol)

        for attempt in range(self.retries):
            try:
                df_raw = self.pro.daily(ts_code=ts_code,
                                        start_date=start_date.replace('-', ''),
                                        end_date=end_date.replace('-', ''))

                if df_raw is None or df_raw.empty:
                    tqdm.write(
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

                tqdm.write(f"  ✅ [Tushare成功] 成功获取 {symbol} 的 {len(df)} 条数据。")
                return df

            except Exception as e:
                tqdm.write(
                    f"  ❌ [Tushare错误] 获取 {symbol} 数据时出错 (尝试 {attempt + 1}/{self.retries}): {e}"
                )
                if attempt < self.retries - 1:
                    tqdm.write(f"    将在 {self.delay} 秒后重试...")
                    time.sleep(self.delay + random.uniform(0, 1))
                else:
                    tqdm.write(
                        f"  ❌ [Tushare失败] 已达到最大重试次数，放弃使用 Tushare 获取该数据。")
                    return None
        return None


class SQLiteDataProvider(BaseDataProvider):
    """
    使用另一个SQLite数据库作为数据源的具体实现。
    它会连接到指定的源数据库文件，查询数据，然后返回给 DataProviderManager，
    后者会将其存入回测专用的数据库中。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source_db_path = kwargs.get('db_path')
        if not self.source_db_path:
            raise ValueError("SQLiteDataProvider 需要一个有效的 'db_path' 参数。")

        self.source_db_handler = DatabaseHandler(db_path=self.source_db_path)
        self.table_name = kwargs.get('table_name', 'stock_daily_prices')

        default_mapping = {
            'ticker': 'code',
            '_date': 'date',
            '_open': 'open',
            '_high': 'high',
            '_low': 'low',
            '_close': 'close',
            '_volume': 'volume',
            '_value': 'turnover',
            '_return': 'pct_change'
        }
        self.column_mapping = kwargs.get('column_mapping', default_mapping)

        tqdm.write(
            f"  [SQLiteProvider初始化] 已连接到源数据库: {self.source_db_path}, 表: {self.table_name}"
        )

    def fetch_data(self, symbol: str, start_date: str,
                   end_date: str) -> pd.DataFrame | None:
        tqdm.write(
            f"  [SQLite适配器] 正在从表'{self.table_name}'获取 {symbol} ({start_date} to {end_date}) 的数据..."
        )
        try:
            # 1. 【查询修正】
            int_start_date = int(start_date.replace('-', ''))
            int_end_date = int(end_date.replace('-', ''))
            try:
                int_symbol = int(symbol)
            except ValueError:
                tqdm.write(f"  🟡 [SQLite警告] 股票代码 '{symbol}' 无法转换为整数，已跳过。")
                return None

            query = f"SELECT * FROM {self.table_name} WHERE ticker = ? AND _date BETWEEN ? AND ?"
            params = (int_symbol, int_start_date, int_end_date)
            df_raw = self.source_db_handler.query_data(query, params=params)

            if df_raw is None or df_raw.empty:
                tqdm.write(f"  🟡 [SQLite警告] 在源数据库中未找到 '{symbol}' 的有效数据。")
                return None

            # 2. 【数据转换】
            tqdm.write(f"  [SQLite适配器] 已获取 {len(df_raw)} 条原始数据，正在进行格式转换...")
            df_transformed = pd.DataFrame()
            df_transformed['date'] = pd.to_datetime(
                df_raw[self.column_mapping.get('date', '_date')],
                format='%Y%m%d')

            target_to_source_map = {
                v: k
                for k, v in self.column_mapping.items()
            }
            numeric_cols = [
                'open', 'high', 'low', 'close', 'turnover', 'pct_change'
            ]

            for col in numeric_cols:
                source_col = target_to_source_map.get(col)
                if source_col and source_col in df_raw.columns:
                    df_transformed[col] = pd.to_numeric(df_raw[source_col],
                                                        errors='coerce')

            vol_source_col = target_to_source_map.get('volume')
            if vol_source_col and vol_source_col in df_raw.columns:
                df_transformed['volume'] = pd.to_numeric(
                    df_raw[vol_source_col],
                    errors='coerce').fillna(0).astype('int64')

            # =================================================================
            # 【【【2.5 新增 - 数据清洗】】】
            # 在保存到数据库前，移除任何包含无效数据的行
            # =================================================================
            tqdm.write(f"  [数据清洗] 清洗前共 {len(df_transformed)} 条数据。")

            # 定义核心列，这些列在回测数据库中是 NOT NULL 的
            critical_cols = ['open', 'high', 'low', 'close', 'volume']

            # 步骤 A: 丢弃任何核心列是 NaN (空值) 的行
            df_transformed.dropna(subset=critical_cols, inplace=True)

            # 步骤 B: 丢弃价格 <= 0 或成交量 < 0 的行 (成交量为0有时是正常停牌，但为负一定是坏数据)
            # 为严格起见，我们移除所有价格为0或负数的数据
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                df_transformed = df_transformed[df_transformed[col] > 0]

            df_transformed = df_transformed[df_transformed['volume'] >= 0]

            tqdm.write(f"  [数据清洗] 清洗后剩余 {len(df_transformed)} 条有效数据。")

            # 如果清洗后数据为空，则直接返回
            if df_transformed.empty:
                tqdm.write(f"  🟡 [SQLite警告] 清洗后，'{symbol}' 无剩余有效数据。")
                return None
            # =================================================================
            # 【数据清洗结束】
            # =================================================================

            # 3. 【格式统一】
            final_columns = [
                'open', 'high', 'low', 'close', 'volume', 'turnover',
                'amplitude', 'pct_change', 'price_change', 'turnover_rate'
            ]
            for col in final_columns:
                if col not in df_transformed.columns:
                    df_transformed[col] = None

            df_transformed.set_index('date', inplace=True)

            tqdm.write(
                f"  ✅ [SQLite成功] 成功转换并清洗 {symbol} 的 {len(df_transformed)} 条数据。"
            )
            return df_transformed[final_columns]

        except Exception as e:
            tqdm.write(f"  ❌ [SQLite错误] 处理源数据库数据时出错: {e}")
            return None

    def __del__(self):
        if hasattr(self, 'source_db_handler'):
            self.source_db_handler.close_connection()
