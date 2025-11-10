# database_handler.py (已重构)
import sqlite3
import pandas as pd
import threading
# import tqdm  <- 【【【移除】】】 不再需要
import logging  # <- 【【【新增】】】


class DatabaseHandler:
    """
    一个专门用于处理SQLite数据库所有交互的类。
    
    【【重构日志】】:
    - 2025-11-09:
      - 引入 'logging' 模块。
      - 将所有的 'tqdm.write()' 调用替换为
        'logging.info()', 'logging.warning()', 'logging.error()'。
      - 确保所有数据库交互日志都能被 logger_config.py 捕获，
        从而解决 Tqdm 进度条冲突问题。
    """

    def __init__(self, db_path='quant_data.db'):
        self.db_path = db_path
        self._local = threading.local()
        # 在主线程中先创建一次表，确保表存在
        try:
            self._create_tables()
        except Exception as e:
            # 在初始化时提供更明确的错误
            logging.critical(f"⛔ 数据库初始化失败: 无法在 '{self.db_path}' 创建表。",
                             exc_info=True)
            raise e

    def _get_connection(self):
        """
        【核心】获取当前线程的数据库连接。
        """
        if not hasattr(self._local, 'connection'):
            try:
                self._local.connection = sqlite3.connect(self.db_path)
                # 【【【修改】】】
                logging.debug(
                    f"  > 🗄️ [线程 {threading.get_ident()}] 创建了新的数据库连接 (-> {self.db_path})。"
                )
            except sqlite3.Error as e:
                # 【【【修改】】】
                logging.error(
                    f"  > ❌ [线程 {threading.get_ident()}] 数据库连接失败: {e}",
                    exc_info=True)
                return None
        return self._local.connection

    def _create_tables(self):
        conn = self._get_connection()
        if conn is None:
            # 【【【新增】】】 记录连接失败
            logging.error("  > ❌ 无法创建表，因为数据库连接 (conn) 为 None。")
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
        # (新增) 行业/基本面示例表
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
            cursor.execute(create_stock_kind_table_sql)  # (新增)
            cursor.execute(create_stock_fundamentals_table_sql)  # (新增)
            cursor.execute(create_code_index_sql)
            cursor.execute(create_date_index_sql)
            conn.commit()
            # 【【【修改】】】
            logging.info(
                "✅ 数据库表 'stock_daily_prices' (及 'stock_kind', 'stock_fundamentals') 和索引已准备就绪。"
            )
        except sqlite3.Error as e:
            # 【【【修改】】】
            logging.error(f"❌ 创建数据表或索引失败: {e}", exc_info=True)

    def save_data(self, df, table_name):
        conn = self._get_connection()
        if conn is None or df.empty:
            if conn is None:
                logging.error(f"❌ 无法保存数据到 '{table_name}'，因为数据库连接为 None。")
            # (df.empty 不是错误，是正常情况)
            return

        try:
            # 【【【新增】】】 增加详细日志
            logging.debug(
                f"  > ⚙️ [线程 {threading.get_ident()}] 正在向 '{table_name}' 表追加 {len(df)} 条数据..."
            )
            df.to_sql(name=table_name,
                      con=conn,
                      if_exists='append',
                      index=False)
            # 【【【修改】】】
            logging.info(
                f"  > ✅ [线程 {threading.get_ident()}] 成功向 '{table_name}' 表追加了 {len(df)} 条数据。"
            )
        except Exception as e:
            # 【【【修改】】】
            # (改为 error 级别, 并增加 exc_info=True 以在日志文件中记录堆栈跟踪)
            logging.error(
                f"  > ❌ [线程 {threading.get_ident()}] 数据保存到 '{table_name}' 失败: {e}",
                exc_info=True)

    def query_data(self, query, params=None):
        conn = self._get_connection()
        if conn is None:
            logging.error(f"❌ 无法执行查询，因为数据库连接为 None。 Query: {query}")
            return pd.DataFrame()
        try:
            # 【【【新增】】】 增加 DEBUG 日志
            logging.debug(
                f"  > ⚙️ [线程 {threading.get_ident()}] 正在执行查询: {query} | Params: {params}"
            )
            df = pd.read_sql(query, conn, params=params)
            # (查询成功不需要打印 info 日志，否则会刷屏)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        except Exception as e:
            # 【【【修改】】】
            logging.error(
                f"❌ [线程 {threading.get_ident()}] 数据查询失败: {e}. Query: {query}",
                exc_info=True)
            return pd.DataFrame()

    def close_connection(self):
        if hasattr(self._local, 'connection'):
            conn = self._get_connection()
            if conn:
                conn.close()
                # 【【【修改】】】
                logging.debug(f"  > 🗄️ [线程 {threading.get_ident()}] 数据库连接已关闭。")
