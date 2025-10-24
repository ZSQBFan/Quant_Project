# !!!重要: 请根据你的数据库结构修改这里的表名和列名!!!
# ----------------------------------------------------
# TABLE_NAME = 'stock_daily_prices'  # <--- 修改为你的表名
# STOCK_CODE_COLUMN = 'code'  # <--- 修改为存储股票代码的列名
# DATE_COLUMN = 'date'  # <--- 修改为存储日期的列名
# ----------------------------------------------------

import sqlite3
import pandas as pd
import akshare as ak


def find_missing_dates(db_path, stock_code, start_date, end_date):
    """
    找出指定股票在给定时间段内缺失记录的交易日。

    参数:
    db_path (str): SQLite数据库文件路径。
    stock_code (str): 要查询的股票代码 (例如 '000001')。
    start_date (str): 查询的开始日期 (格式: 'YYYY-MM-DD')。
    end_date (str): 查询的结束日期 (格式: 'YYYY-MM-DD')。
    
    返回:
    list: 缺失日期的字符串列表 (格式: 'YYYY-MM-DD')。
    """

    # --- 第1步: 获取指定范围内的所有交易日 ---
    try:
        trade_dates_df = ak.tool_trade_date_hist_sina()
        trade_dates_df['trade_date'] = pd.to_datetime(
            trade_dates_df['trade_date']).dt.date

        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date()

        # 使用集合(set)以便进行高效的差集运算
        all_trade_dates = set(trade_dates_df[
            (trade_dates_df['trade_date'] >= start_dt)
            & (trade_dates_df['trade_date'] <= end_dt)]['trade_date'])

        if not all_trade_dates:
            print(f"在 {start_date} 到 {end_date} 之间没有找到任何交易日。")
            return []

    except Exception as e:
        print(f"使用akshare获取交易日历时出错: {e}")
        return []

    # --- 第2步: 从数据库中获取已存在的记录日期 ---

    # !!!重要: 请根据你的数据库结构修改这里的表名和列名!!!
    # ----------------------------------------------------
    TABLE_NAME = 'stock_daily_prices'  # <--- 修改为你的表名
    STOCK_CODE_COLUMN = 'code'  # <--- 修改为存储股票代码的列名
    DATE_COLUMN = 'date'  # <--- 修改为存储日期的列名
    # ----------------------------------------------------

    dates_with_records = set()

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # --- 修改开始 ---
        # 使用 SUBSTR 函数直接截取日期字符串，而不是依赖 date() 函数的解析
        query = f"""
        SELECT DISTINCT SUBSTR({DATE_COLUMN}, 1, 10) 
        FROM {TABLE_NAME} 
        WHERE {STOCK_CODE_COLUMN} = ? 
        AND SUBSTR({DATE_COLUMN}, 1, 10) BETWEEN ? AND ?
        """
        # --- 修改结束 ---

        cursor.execute(query, (stock_code, start_date, end_date))

        rows = cursor.fetchall()
        for row in rows:
            dates_with_records.add(pd.to_datetime(row[0]).date())

    except sqlite3.Error as e:
        print(f"数据库查询时出错: {e}")
        print(
            f"请检查您的表名 '{TABLE_NAME}' 和列名 '{STOCK_CODE_COLUMN}', '{DATE_COLUMN}' 是否正确。"
        )
        return []
    finally:
        if 'conn' in locals() and conn:
            conn.close()

    # --- 第3步: 计算差集，找到缺失的日期 ---
    missing_dates_set = all_trade_dates - dates_with_records

    # 转换为排序后的字符串列表
    missing_dates_list = sorted(
        [d.strftime('%Y-%m-%d') for d in missing_dates_set])

    return missing_dates_list


# --- 使用示例 ---
if __name__ == '__main__':
    # 数据库文件路径
    db_file = 'quant_data.db'

    # 设置你要查询的参数
    stock_to_check = '000001'
    start_period = '2018-01-01'
    end_period = '2018-12-31'

    # 调用函数查找缺失的日期
    missing_days = find_missing_dates(db_file, stock_to_check, start_period,
                                      end_period)

    # 打印结果
    print(f"查询报告：")
    print(f"股票代码: {stock_to_check}")
    print(f"时间范围: {start_period} to {end_period}\n")

    if missing_days:
        print(f"⚠️ 在数据库中找到 {len(missing_days)} 个缺失记录的交易日:")
        # 每行打印5个日期，让输出更美观
        for i in range(0, len(missing_days), 5):
            print("   ".join(missing_days[i:i + 5]))
    else:
        print("✅ 在该时间段内的所有交易日数据完整，没有缺失。")
