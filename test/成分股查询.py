import akshare as ak
import pandas as pd


def get_csi300_code_list():
    """
    使用 akshare 获取最新的沪深300指数成分股，并返回一个仅包含股票代码的列表。

    Returns:
        list: 包含所有成分股代码的Python列表。
              如果获取失败，则返回一个空列表。
    """
    try:
        # 调用akshare接口获取沪深300成分股 DataFrame
        stock_cons_df = ak.index_stock_cons(symbol="000300")

        # --- 核心代码：提取代码列并转换为列表 ---
        # 1. stock_cons_df['品种代码'] -> 选择名为'品种代码'的这一列
        # 2. .tolist() -> 将这一列的所有内容转换成一个Python列表
        code_list = stock_cons_df['品种代码'].tolist()

        print("成功获取沪深300成分股代码列表！")
        return code_list

    except Exception as e:
        print(f"获取数据时发生错误: {e}")
        return []  # 返回空列表


# --- 主程序入口 ---
if __name__ == "__main__":
    # 获取纯股票代码的列表
    csi300_codes = get_csi300_code_list()

    # 如果列表不为空，则打印它
    if csi300_codes:
        print(f"\n成分股总数: {len(csi300_codes)}")
        print("代码列表如下:")
        # 使用 print 直接输出列表
        print(csi300_codes)

        # 如果你想让输出格式更像你示例的那样，每行一个（或几个）
        # 可以取消下面这段代码的注释
        # print("\n--- 格式化输出示例 ---")
        # for code in csi300_codes:
        #     print(f"'{code}',")
