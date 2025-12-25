
import sqlite3
import pandas as pd
import os
import glob

def check_data():
    db_path = './database/quant_data.db'
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT COUNT(DISTINCT code) FROM stock_daily_prices")
        count = cursor.fetchone()[0]
        print(f"quant_data.db 中的股票总数: {count}")
        
        # 检查特定日期范围
        cursor = conn.execute("SELECT MIN(date), MAX(date) FROM stock_daily_prices")
        dates = cursor.fetchone()
        print(f"数据日期范围: {dates[0]} ~ {dates[1]}")
        
        conn.close()
    else:
        print("quant_data.db 不存在")

    # 检查导出的 parquet 文件
    parquet_files = glob.glob('temp/data_explore/*.parquet')
    if parquet_files:
        sample_file = parquet_files[0]
        try:
            df = pd.read_parquet(sample_file)
            print(f"示例文件 {sample_file} 中的股票数量: {len(df)}")
            print(f"列名: {df.columns.tolist()}")
        except Exception as e:
            print(f"读取 parquet 失败: {e}")
    else:
        print("未找到导出的 parquet 文件")

if __name__ == '__main__':
    check_data()
