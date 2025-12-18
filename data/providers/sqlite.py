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
    通用 SQLite 数据提供者。

    该类采用配置驱动设计，通过传入不同的参数支持多种 SQLite 数据源：
    - CSMR 数据库: 配置见 configs/data/providers/sqlite_csmr.yaml
    - JY 数据库: 配置见 configs/data/providers/sqlite_jy.yaml
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

        # 获取配置文件中的日期格式
        self.date_format = kwargs.get('date_format', '%Y-%m-%d')

        logging.info(
            f"[SQLiteProvider] 已连接到源数据库: {self.source_db_path}, 表: {self.table_name}, 日期格式: {self.date_format}"
        )

    def _prepare_date_formats(self, start_date: str, end_date: str):
        """
        统一的日期格式处理函数
        
        优先使用配置文件中的日期格式，失败时回退到其他格式
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            tuple: (int_start_date, int_end_date, str_start_date, str_end_date, date_format)
        """
        logging.debug(f"[SQLite] 处理日期格式: 配置格式={self.date_format}, 开始日期={start_date} ({type(start_date)}), 结束日期={end_date} ({type(end_date)})")
        
        # 强制转换为字符串以防万一
        start_date_str = str(start_date)
        end_date_str = str(end_date)
        
        # 尝试配置文件中指定的日期格式
        if self.date_format == '%Y-%m-%d':
            # 配置格式为 YYYY-MM-DD，直接使用字符串格式
            str_start_date = start_date_str
            str_end_date = end_date_str
            
            # 尝试转换为整数格式
            try:
                # 确保日期字符串只包含数字和连字符
                cleaned_start = start_date_str.replace('-', '').replace(' ', '')
                cleaned_end = end_date_str.replace('-', '').replace(' ', '')
                
                if cleaned_start.isdigit() and cleaned_end.isdigit():
                    int_start_date = int(cleaned_start)
                    int_end_date = int(cleaned_end)
                    logging.debug(f"[SQLite] 整数日期格式转换成功: {int_start_date} ~ {int_end_date}")
                else:
                    raise ValueError(f"包含非数字字符: {cleaned_start}, {cleaned_end}")
            except ValueError as e:
                logging.warning(f"[SQLite] 整数日期格式转换失败: {e}，将使用字符串格式")
                int_start_date = None
                int_end_date = None
            
        elif self.date_format == '%Y%m%d':
            # 配置格式为 YYYYMMDD，优先使用整数格式
            try:
                int_start_date = int(float(start_date_str))
                int_end_date = int(float(end_date_str))
                logging.debug(f"[SQLite] 整数日期格式转换成功: {int_start_date} ~ {int_end_date}")
            except (ValueError, TypeError) as e:
                logging.warning(f"[SQLite] 整数日期格式转换失败: {e}，将使用字符串格式")
                int_start_date = None
                int_end_date = None
            
            str_start_date = start_date_str
            str_end_date = end_date_str
            
        else:
            # 其他格式，使用字符串格式
            str_start_date = start_date_str
            str_end_date = end_date_str
            
            # 尝试转换为整数格式（备用）
            try:
                # 确保日期字符串只包含数字和连字符
                cleaned_start = start_date_str.replace('-', '').replace(' ', '')
                cleaned_end = end_date_str.replace('-', '').replace(' ', '')
                
                if cleaned_start.isdigit() and cleaned_end.isdigit():
                    int_start_date = int(cleaned_start)
                    int_end_date = int(cleaned_end)
                    logging.debug(f"[SQLite] 备用整数日期格式转换成功: {int_start_date} ~ {int_end_date}")
                else:
                    raise ValueError(f"包含非数字字符: {cleaned_start}, {cleaned_end}")
            except ValueError as e:
                logging.warning(f"[SQLite] 备用整数日期格式转换失败: {e}，将使用字符串格式")
                int_start_date = None
                int_end_date = None
        
        return int_start_date, int_end_date, str_start_date, str_end_date, self.date_format

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

            # 使用统一的日期格式处理函数
            int_start_date, int_end_date, str_start_date, str_end_date, date_format = self._prepare_date_formats(start_date, end_date)

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
        """
        logging.info(
            f"[SQLite] 开始优化获取全股票数据 - 时间段: {start_date} ~ {end_date}"
        )
        
        try:
            # 获取字段名（从配置映射中获取）
            symbol_col = None
            date_col = None
            
            for db_col, standard_col in self.column_mapping.items():
                if standard_col == 'code' or standard_col == 'symbol':
                    symbol_col = db_col
                elif standard_col == 'date':
                    date_col = db_col
            
            if not symbol_col:
                symbol_col = 'ticker'  # 默认值
            if not date_col:
                date_col = '_date'  # 默认值
            
            # 使用统一的日期格式处理函数
            int_start_date, int_end_date, str_start_date, str_end_date, date_format = self._prepare_date_formats(start_date, end_date)
            
            # 构建查询 - 获取时间段内的所有数据
            query = f"SELECT * FROM {self.table_name} WHERE {date_col} BETWEEN ? AND ? ORDER BY {date_col}, {symbol_col}"
            
            logging.info(f"[SQLite] 执行全表查询: {query} 日期范围: {start_date} ~ {end_date}")
            
            # 尝试整数日期格式
            df_raw = self.source_db_handler.query_data(query, params=(int_start_date, int_end_date))
            
            # 如果没查到，尝试字符串日期格式
            if df_raw is None or df_raw.empty:
                logging.info(f"[SQLite] 整数日期查询失败，尝试字符串日期格式...")
                df_raw = self.source_db_handler.query_data(query, params=(str_start_date, str_end_date))
            
            if df_raw is None or df_raw.empty:
                logging.warning(f"[SQLite] 在时间段 {start_date} ~ {end_date} 内未找到任何数据")
                return None
            
            logging.info(f"[SQLite] 原始查询返回 {len(df_raw)} 条数据，开始转换...")
            
            # 数据转换（复用 fetch_data 的转换逻辑，但不完全相同）
            df_transformed = pd.DataFrame()
            
            # 日期转换
            if date_col in df_raw.columns:
                # 自动检测日期解析格式
                sample_date = df_raw[date_col].iloc[0]
                date_fmt = None
                
                # 转换为字符串进行检测
                sample_date_str = str(sample_date)
                if '-' in sample_date_str:
                    date_fmt = '%Y-%m-%d'
                elif len(sample_date_str.split('.')[0]) == 8 and sample_date_str.split('.')[0].isdigit():
                    date_fmt = '%Y%m%d'
                
                logging.debug(f"[SQLite] 自动检测日期格式: 样本={sample_date} ({type(sample_date)}), 识别格式={date_fmt}")
                
                if date_fmt == '%Y%m%d':
                    # 对于 YYYYMMDD 格式，先转为字符串再解析，避免被误认为纳秒
                    df_transformed['date'] = pd.to_datetime(df_raw[date_col].astype(str).str.split('.').str[0], format='%Y%m%d', errors='coerce')
                else:
                    df_transformed['date'] = pd.to_datetime(df_raw[date_col], format=date_fmt, errors='coerce')
            
            # 股票代码转换
            if symbol_col in df_raw.columns:
                # 标准化股票代码
                symbols = []
                for symbol in df_raw[symbol_col]:
                    if pd.notna(symbol):
                        # 如果是纯数字，处理可能的前导零
                        if isinstance(symbol, (int, float)) or (isinstance(symbol, str) and symbol.isdigit()):
                            symbol_str = str(int(float(symbol)))
                            if len(symbol_str) <= 6:
                                symbols.append(symbol_str.zfill(6))
                            else:
                                symbols.append(symbol_str)
                        else:
                            symbols.append(str(symbol))
                    else:
                        symbols.append(None)
                
                df_transformed['asset'] = symbols
            
            # 数据列映射和转换
            target_to_source_map = {
                v: k for k, v in self.column_mapping.items()
            }
            
            # 默认需要的数据列
            default_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'pct_change']
            
            # 如果指定了required_columns，只处理这些列
            if required_columns:
                cols_to_process = required_columns
            else:
                cols_to_process = default_cols
            
            # 处理各个数据列
            for col in cols_to_process:
                source_col = target_to_source_map.get(col)
                if source_col and source_col in df_raw.columns:
                    if col == 'volume':
                        # volume列特殊处理
                        df_transformed[col] = pd.to_numeric(
                            df_raw[source_col], errors='coerce').fillna(0).astype('int64')
                    else:
                        # 其他数值列
                        df_transformed[col] = pd.to_numeric(df_raw[source_col], errors='coerce')
            
            # 移除无效行（没有日期或资产代码的行）
            initial_count = len(df_transformed)
            df_transformed = df_transformed.dropna(subset=['date', 'asset'])
            final_count = len(df_transformed)
            
            if initial_count != final_count:
                logging.info(f"[SQLite] 数据清洗: 移除了 {initial_count - final_count} 条无效数据")
            
            if df_transformed.empty:
                logging.warning(f"[SQLite] 转换后无有效数据")
                return None
            
            # 设置索引为MultiIndex (date, asset)
            df_transformed.set_index(['date', 'asset'], inplace=True)
            df_transformed.sort_index(inplace=True)
            
            # 可选：移除完全为NaN的行（但保留部分有数据的行，因为因子计算支持NaN）
            logging.info(f"[SQLite] 优化数据获取完成！")
            logging.info(f"  - 时间段: {start_date} ~ {end_date}")
            logging.info(f"  - 总记录数: {len(df_transformed)}")
            logging.info(f"  - 股票数量: {df_transformed.index.get_level_values('asset').nunique()}")
            logging.info(f"  - 日期数量: {df_transformed.index.get_level_values('date').nunique()}")
            logging.info(f"  - 数据列: {list(df_transformed.columns)}")
            
            return df_transformed
            
        except Exception as e:
            logging.error(f"[SQLite] 获取全股票数据失败: {e}", exc_info=True)
            return None
