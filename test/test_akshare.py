import akshare as ak


def get_csi300_stocks():
    """获取沪深300指数的最新成分股列表"""
    try:
        cons_df = ak.index_stock_cons(symbol="000300")
        # 最新 akshare 返回的股票代码列名为 '品种代码'
        stock_list = cons_df['品种代码'].tolist()
        stock_list = sorted(list(set(cons_df['品种代码'].tolist())))
        print(f"成功获取到 {len(stock_list)} 只沪深300成分股。")
        return stock_list
    except Exception as e:
        print(f"错误：获取指数成分股失败: {e}")
        return []


if __name__ == '__main__':
    csi300_universe = get_csi300_stocks()
    print(csi300_universe)
