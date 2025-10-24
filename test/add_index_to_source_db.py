# add_index_to_source_db.py
import sqlite3
import time
import os

# ======================================================================
# 【请在这里配置您的源数据库文件的准确路径】
# 例如: './database/JY_database/sqlite/JY_database.sqlite'
# 或: 'D:/my_project/data/JY_database.sqlite'
SOURCE_DB_PATH = "./database/JY_database/sqlite/JY_database.sqlite"
# ======================================================================

# 要添加索引的表名
TABLE_NAME = "JY_t_price_daily"
# 索引名称 (可自定义)
INDEX_NAME = "idx_ticker_date"
# 要索引的列 (顺序很重要，查询频率高的放前面)
COLUMNS_TO_INDEX = ("ticker", "_date")


def add_index():
    """连接到源数据库并创建复合索引。"""
    if not os.path.exists(SOURCE_DB_PATH):
        print(f"❌ 错误：找不到数据库文件，请检查路径配置是否正确: {SOURCE_DB_PATH}")
        return

    print(f"正在连接到数据库: {SOURCE_DB_PATH} ...")
    conn = None
    try:
        conn = sqlite3.connect(SOURCE_DB_PATH)
        cursor = conn.cursor()

        print(
            f"正在为表 '{TABLE_NAME}' 在列 {COLUMNS_TO_INDEX} 上创建索引 '{INDEX_NAME}'..."
        )
        print("这个过程可能需要几分钟，具体时间取决于您的数据库大小和磁盘性能，请耐心等待...")

        start_time = time.time()

        # 使用 "IF NOT EXISTS" 可以安全地重复运行此脚本，如果索引已存在则不会报错
        sql_command = f"""
        CREATE INDEX IF NOT EXISTS {INDEX_NAME} 
        ON {TABLE_NAME} ({', '.join(COLUMNS_TO_INDEX)});
        """

        cursor.execute(sql_command)
        conn.commit()

        end_time = time.time()

        print("\n" + "=" * 50)
        print(f"✅ 成功！索引 '{INDEX_NAME}' 已成功创建或已存在。")
        print(f"   耗时: {end_time - start_time:.2f} 秒")
        print("=" * 50 + "\n")

    except sqlite3.Error as e:
        print(f"\n❌ 数据库操作失败: {e}")
    finally:
        if conn:
            conn.close()
            print("数据库连接已关闭。")


if __name__ == "__main__":
    add_index()
