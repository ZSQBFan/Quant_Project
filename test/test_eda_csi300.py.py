import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: 获取数据 ---
print("正在获取沪深300指数日线数据...")

# 获取指数数据，'sh000300' 是沪深300的代码
# AKShare 指数数据默认就是后复权
try:
    index_df = ak.stock_zh_index_daily(symbol="sz399006")
    print("✅ 数据获取成功！")
    
    # --- Step 2: 数据预处理 ---
    # AKShare返回的日期是 object 类型，我们需要将其转换为 datetime 类型，并设为索引
    # 这对于后续按时间进行分析和绘图至关重要
    index_df['date'] = pd.to_datetime(index_df['date'])
    index_df.set_index('date', inplace=True)

    # 为了便于分析，我们只保留核心字段
    index_df = index_df[['open', 'high', 'low', 'close', 'volume']]
    
    # 筛选出最近3个月的数据进行分析
    df = index_df.last('300D') # 90个日历天数大约等于3个月

    print("\n--- 数据预览 (最近5条) ---")
    print(df.tail())

    # --- Step 3: 计算核心指标 ---
    # 1. 计算日收益率
    df['daily_return'] = df['close'].pct_change()

    # 2. 计算移动平均线 (Moving Averages)
    df['ma20'] = df['close'].rolling(window=5).mean() # 5日均线
    df['ma60'] = df['close'].rolling(window=30).mean() # 30日均线 (季线)
    
    # 删除前期的空值 (因为计算移动平均线和收益率会导致前面几天没有数据)
    df.dropna(inplace=True)

    print("\n--- 计算指标后预览 ---")
    print(df.tail())

    # --- Step 4: 数据可视化 ---
    # Matplotlib 全局设置，防止中文乱码
    plt.rcParams['font.sans-serif'] = ['Heiti TC'] # Mac系统下通常有的中文字体
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

    # 1. 绘制收盘价和移动平均线
    plt.figure(figsize=(15, 7))
    plt.title('收盘价 & 移动平均线')
    plt.plot(df.index, df['close'], label='收盘价', alpha=0.7)
    plt.plot(df.index, df['ma20'], label='20日均线', linestyle='--')
    plt.plot(df.index, df['ma60'], label='60日均线', linestyle=':')
    plt.legend()
    plt.grid(True)
    plt.show() # 显示图表

    # 2. 绘制日收益率分布直方图
    plt.figure(figsize=(15, 7))
    plt.title('指数日收益率分布')
    sns.histplot(df['daily_return'], bins=100, kde=True) # kde=True 会画一条拟合曲线
    plt.grid(True)
    plt.show()

    print("\n✅ EDA 分析完成！请查看弹出的图表。")

except Exception as e:
    print(f"❌ 程序出错: {e}")