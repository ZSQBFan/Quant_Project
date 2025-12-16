"""
SQLite 数据提供者
"""

import pandas as pd
import logging

from .base import BaseDataProvider
from ..handlers import DatabaseHandler


class SQLiteDataProvider(BaseDataProvider):
    """
    使用另一个 SQLite 数据库作为数据源的具体实现。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source_db_path = kwargs.get('db_path')
        if not self.source_db_path:
            logging.critical("SQLiteDataProvider 需要一个有效的 'db_path' 参数。")
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

        logging.info(
            f"[SQLiteProvider] 已连接到源数据库: {self.source_db_path}, 表: {self.table_name}"
        )

    def fetch_data(self, symbol: str, start_date: str,
                   end_date: str) -> pd.DataFrame | None:
        logging.info(
            f"[SQLite] 正在从表'{self.table_name}'获取 {symbol} ({start_date} to {end_date}) 的数据..."
        )
        try:
            int_start_date = int(start_date.replace('-', ''))
            int_end_date = int(end_date.replace('-', ''))
            try:
                int_symbol = int(symbol)
            except ValueError:
                logging.warning(
                    f"[SQLite] 股票代码 '{symbol}' 无法转换为整数，已跳过。")
                return None

            query = f"SELECT * FROM {self.table_name} WHERE ticker = ? AND _date BETWEEN ? AND ?"
            params = (int_symbol, int_start_date, int_end_date)
            df_raw = self.source_db_handler.query_data(query, params=params)

            if df_raw is None or df_raw.empty:
                logging.warning(
                    f"[SQLite] 在源数据库中未找到 '{symbol}' 的有效数据。")
                return None

            logging.info(
                f"[SQLite] 已获取 {len(df_raw)} 条原始数据，正在进行格式转换...")
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

            # 数据清洗
            logging.info(f"[数据清洗] 清洗前共 {len(df_transformed)} 条数据。")

            critical_cols = ['open', 'high', 'low', 'close', 'volume']
            df_transformed.dropna(subset=critical_cols, inplace=True)

            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                df_transformed = df_transformed[df_transformed[col] > 0]
            df_transformed = df_transformed[df_transformed['volume'] >= 0]

            logging.info(f"[数据清洗] 清洗后剩余 {len(df_transformed)} 条有效数据。")

            if df_transformed.empty:
                logging.warning(f"[SQLite] 清洗后，'{symbol}' 无剩余有效数据。")
                return None

            # 格式统一
            final_columns = [
                'open', 'high', 'low', 'close', 'volume', 'turnover',
                'amplitude', 'pct_change', 'price_change', 'turnover_rate'
            ]
            for col in final_columns:
                if col not in df_transformed.columns:
                    df_transformed[col] = None

            df_transformed.set_index('date', inplace=True)

            logging.info(
                f"[SQLite] 成功转换并清洗 {symbol} 的 {len(df_transformed)} 条数据。"
            )
            return df_transformed[final_columns]

        except Exception as e:
            logging.error(f"[SQLite] 处理源数据库数据时出错: {e}", exc_info=True)
            return None

    def __del__(self):
        if hasattr(self, 'source_db_handler'):
            self.source_db_handler.close_connection()
