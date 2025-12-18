"""
数据库处理器

处理 SQLite 数据库的所有交互操作。
"""

import sqlite3
import pandas as pd
import threading
import logging


class DatabaseHandler:
    """
    SQLite 数据库交互处理器。

    支持多线程安全的数据库连接管理。
    """

    def __init__(self, db_path='quant_data.db'):
        self.db_path = db_path
        self._local = threading.local()
        try:
            self._create_tables()
        except Exception as e:
            logging.critical(f"数据库初始化失败: 无法在 '{self.db_path}' 创建表。",
                             exc_info=True)
            raise e

    def _get_connection(self):
        """获取当前线程的数据库连接。"""
        if not hasattr(self._local, 'connection'):
            try:
                self._local.connection = sqlite3.connect(self.db_path)
                logging.debug(
                    f"[线程 {threading.get_ident()}] 创建了新的数据库连接 (-> {self.db_path})。"
                )
            except sqlite3.Error as e:
                logging.error(
                    f"[线程 {threading.get_ident()}] 数据库连接失败: {e}",
                    exc_info=True)
                return None
        return self._local.connection

    def _create_tables(self):
        """创建必要的数据表。"""
        conn = self._get_connection()
        if conn is None:
            logging.error("无法创建表，因为数据库连接为 None。")
            return

        create_daily_prices_table_sql = """
        CREATE TABLE IF NOT EXISTS stock_daily_prices (
            code TEXT NOT NULL, date DATE NOT NULL, open REAL NOT NULL,
            high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
            volume INTEGER NOT NULL, turnover REAL, amplitude REAL,
            pct_change REAL, price_change REAL, turnover_rate REAL,
            PRIMARY KEY (code, date)
        );
        """
        create_stock_kind_table_sql = """
        CREATE TABLE IF NOT EXISTS stock_kind (
            Stkcd TEXT PRIMARY KEY,
            Nnindnme TEXT
        );
        """
        create_stock_fundamentals_table_sql = """
        CREATE TABLE IF NOT EXISTS stock_fundamentals (
            asset TEXT NOT NULL,
            date DATE NOT NULL,
            ep_ratio REAL,
            fcf_per_share REAL,
            PRIMARY KEY (asset, date)
        );
        """

        create_code_index_sql = "CREATE INDEX IF NOT EXISTS idx_code ON stock_daily_prices (code);"
        create_date_index_sql = "CREATE INDEX IF NOT EXISTS idx_date ON stock_daily_prices (date);"

        try:
            cursor = conn.cursor()
            cursor.execute(create_daily_prices_table_sql)
            cursor.execute(create_stock_kind_table_sql)
            cursor.execute(create_stock_fundamentals_table_sql)
            cursor.execute(create_code_index_sql)
            cursor.execute(create_date_index_sql)
            conn.commit()
            logging.info(
                "数据库表 'stock_daily_prices' (及 'stock_kind', 'stock_fundamentals') 和索引已准备就绪。"
            )
        except sqlite3.Error as e:
            logging.error(f"创建数据表或索引失败: {e}", exc_info=True)

    def save_data(self, df, table_name):
        """保存 DataFrame 到指定表。"""
        conn = self._get_connection()
        if conn is None or df.empty:
            if conn is None:
                logging.error(f"无法保存数据到 '{table_name}'，因为数据库连接为 None。")
            return

        try:
            # 调试日志：检查DataFrame结构和date列
            logging.debug(
                f"[线程 {threading.get_ident()}] DataFrame 列: {df.columns.tolist()}"
            )
            logging.debug(
                f"[线程 {threading.get_ident()}] DataFrame 索引: {df.index.name}"
            )
            if 'date' in df.columns:
                null_count = df['date'].isnull().sum()
                logging.debug(
                    f"[线程 {threading.get_ident()}] date列中NULL值数量: {null_count}"
                )
                if null_count > 0:
                    logging.warning(
                        f"[线程 {threading.get_ident()}] 发现 {null_count} 个NULL的date值！"
                    )
            else:
                logging.warning(
                    f"[线程 {threading.get_ident()}] DataFrame中没有date列！"
                )
            
            # 如果date是索引，将其重置为列
            if df.index.name == 'date' or (hasattr(df.index, 'names') and 'date' in df.index.names):
                logging.debug(
                    f"[线程 {threading.get_ident()}] 将date索引重置为列"
                )
                df = df.reset_index()
            
            # 处理重复数据：使用REPLACE INTO
            if table_name == 'stock_daily_prices' and 'code' in df.columns and 'date' in df.columns:
                logging.debug(f"[线程 {threading.get_ident()}] 检测到stock_daily_prices表，使用REPLACE INTO处理重复数据")
                
                # 创建临时表来存储新数据
                temp_table = f"temp_stock_data_{threading.get_ident()}"
                df.to_sql(name=temp_table, con=conn, if_exists='replace', index=False)
                
                # 使用REPLACE INTO语句处理重复数据
                cursor = conn.cursor()
                
                # 获取列名（排除自增ID列）
                columns = [col for col in df.columns if col != 'index']
                cols_str = ', '.join(columns)
                placeholders = ', '.join(['?' for _ in columns])
                
                replace_sql = f"""
                REPLACE INTO {table_name} ({cols_str})
                SELECT {cols_str} FROM {temp_table}
                """
                
                cursor.execute(replace_sql)
                cursor.execute(f"DROP TABLE IF EXISTS {temp_table}")
                
                logging.info(f"[线程 {threading.get_ident()}] 成功向 '{table_name}' 表插入/更新了 {len(df)} 条数据。")
                
            else:
                logging.debug(
                    f"[线程 {threading.get_ident()}] 正在向 '{table_name}' 表追加 {len(df)} 条数据..."
                )
                df.to_sql(name=table_name,
                          con=conn,
                          if_exists='append',
                          index=False)
                logging.info(
                    f"[线程 {threading.get_ident()}] 成功向 '{table_name}' 表追加了 {len(df)} 条数据。"
                )
                
        except Exception as e:
            logging.error(
                f"[线程 {threading.get_ident()}] 数据保存到 '{table_name}' 失败: {e}",
                exc_info=True)

    def query_data(self, query, params=None):
        """执行查询并返回 DataFrame。"""
        conn = self._get_connection()
        if conn is None:
            logging.error(f"无法执行查询，因为数据库连接为 None。 Query: {query}")
            return pd.DataFrame()
        try:
            logging.debug(
                f"[线程 {threading.get_ident()}] 正在执行查询: {query} | Params: {params}"
            )
            df = pd.read_sql(query, conn, params=params)

            logging.debug(f"查询返回数据形状: {df.shape}")

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

            return df
        except Exception as e:
            logging.error(
                f"[线程 {threading.get_ident()}] 数据查询失败: {e}. Query: {query}",
                exc_info=True)
            return pd.DataFrame()

    def close_connection(self):
        """关闭当前线程的数据库连接。"""
        if hasattr(self._local, 'connection'):
            conn = self._get_connection()
            if conn:
                conn.close()
                logging.debug(f"[线程 {threading.get_ident()}] 数据库连接已关闭。")

    def __getstate__(self):
        """
        Pickle序列化时排除threading.local对象。
        
        Returns:
            dict: 排除_local属性后的状态字典
        """
        state = self.__dict__.copy()
        # 排除threading.local对象，因为它无法被pickle序列化
        state.pop('_local', None)
        return state

    def __setstate__(self, state):
        """
        Pickle反序列化时重建threading.local对象。
        
        Args:
            state: 从pickle加载的状态字典
        """
        self.__dict__.update(state)
        # 在新进程中重建threading.local对象
        self._local = threading.local()
