# database_handler.py
import sqlite3
import pandas as pd
import threading
from tqdm import tqdm


class DatabaseHandler:
    """
    一个专门用于处理SQLite数据库所有交互的类。
    【已重构】使用 threading.local() 来管理线程安全的数据库连接，
    确保每个线程都使用自己独立的连接对象。
    """

    def __init__(self, db_path='quant_data.db'):
        self.db_path = db_path
        # 2. 不在初始化时创建全局连接，而是创建一个线程本地存储对象
        self._local = threading.local()
        # 在主线程中先创建一次表，确保表存在
        self._create_tables()

    def _get_connection(self):
        """
        【核心】获取当前线程的数据库连接。
        如果当前线程还没有连接，则创建一个新的连接并存储起来。
        """
        # 3. 检查当前线程的本地存储中是否已有 'connection' 属性
        if not hasattr(self._local, 'connection'):
            # 如果没有，为这个线程创建一个新的连接
            try:
                self._local.connection = sqlite3.connect(self.db_path)
                tqdm.write(f"✅ [线程 {threading.get_ident()}] 创建了新的数据库连接。")
            except sqlite3.Error as e:
                tqdm.write(f"❌ [线程 {threading.get_ident()}] 数据库连接失败: {e}")
                return None
        # 4. 返回当前线程专属的连接对象
        return self._local.connection

    def _create_tables(self):
        conn = self._get_connection()  # 使用当前线程的连接
        if conn is None:
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
        create_code_index_sql = "CREATE INDEX IF NOT EXISTS idx_code ON stock_daily_prices (code);"
        create_date_index_sql = "CREATE INDEX IF NOT EXISTS idx_date ON stock_daily_prices (date);"

        try:
            cursor = conn.cursor()
            cursor.execute(create_daily_prices_table_sql)
            cursor.execute(create_code_index_sql)
            cursor.execute(create_date_index_sql)
            conn.commit()
            tqdm.write("✅ 数据表 'stock_daily_prices' 及索引已准备就绪。")
        except sqlite3.Error as e:
            tqdm.write(f"❌ 创建数据表或索引失败: {e}")

    def save_data(self, df, table_name):
        conn = self._get_connection()  # 使用当前线程的连接
        if conn is None or df.empty:
            return
        try:
            df.to_sql(name=table_name,
                      con=conn,
                      if_exists='append',
                      index=False)
            tqdm.write(
                f"✅ [线程 {threading.get_ident()}] 成功向 '{table_name}' 表中追加了 {len(df)} 条数据。"
            )
        except Exception as e:
            tqdm.write(f"❌ [线程 {threading.get_ident()}] 数据保存失败: {e}")

    def query_data(self, query, params=None):
        conn = self._get_connection()  # 使用当前线程的连接
        if conn is None:
            return pd.DataFrame()
        try:
            df = pd.read_sql(query, conn, params=params)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        except Exception as e:
            tqdm.write(f"❌ [线程 {threading.get_ident()}] 数据查询失败: {e}")
            return pd.DataFrame()

    def close_connection(self):
        if hasattr(self._local, 'connection'):
            conn = self._get_connection()
            if conn:
                conn.close()
                tqdm.write(f"✅ [线程 {threading.get_ident()}] 数据库连接已关闭。")
