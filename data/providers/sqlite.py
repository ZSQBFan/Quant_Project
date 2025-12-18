"""
SQLite 数据提供者
"""

import pandas as pd
import logging
from typing import List, Optional

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
            f"[SQLite] [{symbol}] 开始获取数据 - 表: '{self.table_name}', 日期范围: {start_date} ~ {end_date}"
        )
        try:
            # 获取字段名（从配置映射中获取）
            symbol_col = 'ticker'
            date_col = '_date'
            
            for db_col, standard_col in self.column_mapping.items():
                if standard_col in ['code', 'symbol']:
                    symbol_col = db_col
                elif standard_col == 'date':
                    date_col = db_col

            # 准备两种可能的日期格式
            int_start_date = int(start_date.replace('-', ''))
            int_end_date = int(end_date.replace('-', ''))
            str_start_date = start_date
            str_end_date = end_date

            # 处理股票代码
            try:
                query_symbol = int(symbol)
            except ValueError:
                query_symbol = symbol

            # 执行查询 - 尝试整数日期格式 (YYYYMMDD)
            query = f"SELECT * FROM {self.table_name} WHERE {symbol_col} = ? AND {date_col} BETWEEN ? AND ?"
            params = (query_symbol, int_start_date, int_end_date)
            df_raw = self.source_db_handler.query_data(query, params=params)

            # 如果没查到，尝试字符串日期格式 (YYYY-MM-DD)
            if df_raw is None or df_raw.empty:
                params = (symbol, str_start_date, str_end_date)
                df_raw = self.source_db_handler.query_data(query, params=params)

            if df_raw is None or df_raw.empty:
                logging.warning(
                    f"[SQLite] [{symbol}] 在源数据库中未找到有效数据。")
                return None

            logging.info(
                f"[SQLite] [{symbol}] 成功获取 {len(df_raw)} 条原始数据，开始格式转换..."
            )
            
            # 自动检测日期解析格式
            sample_date = df_raw[date_col].iloc[0]
            date_fmt = None
            if isinstance(sample_date, (int, float)) or (isinstance(sample_date, str) and str(sample_date).isdigit()):
                date_fmt = '%Y%m%d'
            elif isinstance(sample_date, str) and '-' in sample_date:
                date_fmt = '%Y-%m-%d'

            df_transformed = pd.DataFrame()
            df_transformed['date'] = pd.to_datetime(
                df_raw[date_col],
                format=date_fmt)

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
            logging.info(f"[SQLite] [{symbol}] 数据清洗前共 {len(df_transformed)} 条数据。")

            critical_cols = ['open', 'high', 'low', 'close', 'volume']
            df_transformed.dropna(subset=critical_cols, inplace=True)

            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                df_transformed = df_transformed[df_transformed[col] > 0]
            df_transformed = df_transformed[df_transformed['volume'] >= 0]

            logging.info(f"[SQLite] [{symbol}] 数据清洗后剩余 {len(df_transformed)} 条有效数据。")

            if df_transformed.empty:
                logging.warning(f"[SQLite] [{symbol}] 清洗后无剩余有效数据。")
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
                f"[SQLite] [{symbol}] 数据转换和清洗完成 - 最终数据条数: {len(df_transformed)}"
            )
            return df_transformed[final_columns]

        except Exception as e:
            logging.error(f"[SQLite] [{symbol}] 处理源数据库数据时出错: {e}", exc_info=True)
            return None

    def __del__(self):
        if hasattr(self, 'source_db_handler'):
            self.source_db_handler.close_connection()

    def get_all_symbols(self, target_date: Optional[str] = None) -> List[str]:
        """
        获取全部股票代码列表（SQLite实现）

        Args:
            target_date: 目标日期，用于获取该时间点存在的股票列表
                        如果为None，使用开始日期以避免幸存者偏差

        Returns:
            股票代码列表
        """
        logging.info(f"[SQLite] 正在获取{target_date or '指定日期'}的全部股票代码列表...")
        
        try:
            # 获取股票代码字段名（从配置映射中获取）
            symbol_col = None
            for db_col, standard_col in self.column_mapping.items():
                if standard_col == 'code' or standard_col == 'symbol':
                    symbol_col = db_col
                    break
            
            if not symbol_col:
                # 如果配置映射中没有，使用默认值
                symbol_col = 'ticker'  # 基于原有代码的默认值
            
            # 获取日期字段名
            date_col = None
            for db_col, standard_col in self.column_mapping.items():
                if standard_col == 'date':
                    date_col = db_col
                    break
            
            if not date_col:
                date_col = '_date'  # 基于原有代码的默认值
            
            # 构建查询
            if target_date:
                # 转换日期格式（支持字符串、datetime.date 和 datetime.datetime 类型）
                if hasattr(target_date, 'strftime'):
                    # datetime.date 或 datetime.datetime 对象
                    target_date_int = int(target_date.strftime('%Y%m%d'))
                else:
                    # 字符串
                    target_date_int = int(str(target_date).replace('-', ''))
                
                # 准备两种可能的日期格式
                str_target_date = str(target_date)
                
                query = f"SELECT DISTINCT {symbol_col} FROM {self.table_name} WHERE {date_col} = ?"
                
                # 尝试整数日期
                logging.info(f"[SQLite] 执行查询 (int date): {query} params=({target_date_int},)")
                df = self.source_db_handler.query_data(query, params=(target_date_int,))
                
                # 如果没查到且输入是字符串，尝试字符串日期
                if (df is None or df.empty) and str_target_date:
                    logging.info(f"[SQLite] 执行查询 (str date): {query} params=({str_target_date},)")
                    df = self.source_db_handler.query_data(query, params=(str_target_date,))
            else:
                # 如果没有指定目标日期，获取所有有数据的股票
                query = f"SELECT DISTINCT {symbol_col} FROM {self.table_name}"
                logging.info(f"[SQLite] 执行查询 (all symbols): {query}")
                df = self.source_db_handler.query_data(query, params=())
            
            if df is None or df.empty:
                logging.warning("[SQLite] 未能获取到股票代码列表")
                return []
            
            # 提取股票代码
            symbols = []
            for symbol in df[symbol_col]:
                if pd.notna(symbol):
                    # 转换为字符串并标准化
                    # 如果是纯数字，处理可能的前导零
                    if isinstance(symbol, (int, float)) or (isinstance(symbol, str) and symbol.isdigit()):
                        symbol_str = str(int(float(symbol)))
                        if len(symbol_str) <= 6:
                            symbols.append(symbol_str.zfill(6))
                        else:
                            symbols.append(symbol_str)
                    else:
                        symbols.append(str(symbol))
            
            # 去重并排序
            symbols = sorted(list(set(symbols)))
            
            logging.info(f"[SQLite] 成功获取 {len(symbols)} 只股票代码")
            return symbols
            
        except Exception as e:
            logging.error(f"[SQLite] 获取全部股票代码失败: {e}", exc_info=True)
            return []
