"""
Tushare 数据提供者
"""

import pandas as pd
import tushare as ts
import time
import random
import logging

from .base import BaseDataProvider


class TushareDataProvider(BaseDataProvider):
    """
    使用 Tushare 作为数据源的具体实现。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.token = kwargs.get('token')
        if not self.token:
            logging.critical("TushareDataProvider 需要一个有效的 'token' 参数。")
            raise ValueError("TushareDataProvider 需要一个有效的 'token' 参数。")
        self.pro = ts.pro_api(self.token)
        self.adjust = kwargs.get('adjust', "hfq")
        logging.info("TushareDataProvider 已初始化。")

    def _convert_symbol_to_ts_code(self, symbol):
        if symbol.startswith('sh') or symbol.startswith('sz'):
            return f"{symbol.replace('sh', '').replace('sz', '')}.SH" if symbol.startswith(
                'sh') else f"{symbol.replace('sh', '').replace('sz', '')}.SZ"
        return f"{symbol}.SH" if symbol.startswith('6') else f"{symbol}.SZ"

    def fetch_data(self, symbol: str, start_date: str,
                   end_date: str) -> pd.DataFrame | None:
        logging.info(
            f"[Tushare] 正在获取 {symbol} 从 {start_date} 到 {end_date} 的数据..."
        )
        ts_code = self._convert_symbol_to_ts_code(symbol)

        for attempt in range(self.retries):
            try:
                df_raw = self.pro.daily(ts_code=ts_code,
                                        start_date=start_date.replace('-', ''),
                                        end_date=end_date.replace('-', ''))

                if df_raw is None or df_raw.empty:
                    logging.warning(
                        f"[Tushare] 在 {start_date} - {end_date} 范围内未返回 '{symbol}' 的有效数据。"
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

                logging.info(
                    f"[Tushare] 成功获取 {symbol} 的 {len(df)} 条数据。")
                return df

            except Exception as e:
                logging.error(
                    f"[Tushare] 获取 {symbol} 数据时出错 (尝试 {attempt + 1}/{self.retries}): {e}",
                    exc_info=True)
                if attempt < self.retries - 1:
                    logging.warning(f"将在 {self.delay} 秒后重试...")
                    time.sleep(self.delay + random.uniform(0, 1))
                else:
                    logging.error(
                        f"[Tushare] 已达到最大重试次数，放弃获取 {symbol} 的数据。"
                    )
                    return None
        return None
