"""
Akshare 数据提供者
"""

import pandas as pd
import akshare as ak
import time
import random
import logging

from .base import BaseDataProvider


class AkshareDataProvider(BaseDataProvider):
    """
    使用 Akshare 作为数据源的具体实现。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adjust = kwargs.get('adjust', "hfq")

    def fetch_data(self, symbol: str, start_date: str,
                   end_date: str) -> pd.DataFrame | None:
        logging.info(
            f"[Akshare] 正在获取 {symbol} 从 {start_date} 到 {end_date} 的数据..."
        )
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
                    logging.warning(
                        f"[Akshare] 在 {start_date} - {end_date} 范围内未返回 '{symbol}' 的有效数据。"
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
                    logging.info(
                        f"[Akshare] 成功获取 {symbol} 的 {len(df_final)} 条数据。"
                    )
                    return df_final
                else:
                    return None

            except Exception as e:
                logging.error(
                    f"[Akshare] 获取 {symbol} 数据时出错 (尝试 {attempt + 1}/{self.retries}): {e}",
                    exc_info=True)
                if attempt < self.retries - 1:
                    logging.warning(f"将在 {self.delay} 秒后重试...")
                    time.sleep(self.delay + random.uniform(0, 1))
                else:
                    logging.error(
                        f"[Akshare] 已达到最大重试次数，放弃获取 {symbol} 的数据。"
                    )
                    return None
        return None
