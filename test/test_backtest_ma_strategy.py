import backtrader as bt
import akshare as ak
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# --- 第1步: 定义策略 ---
# 我们将之前定义的交易逻辑，写成一个 Backtrader 的策略类
class MovingAverageCrossStrategy(bt.Strategy):
    # 定义策略的参数，这里是短期和长期均线的周期
    params = (
        ('short_ma', 20),
        ('long_ma', 60),
    )

    def __init__(self):
        # 初始化策略时会自动调用
        # self.dataclose 指的是每日的收盘价序列
        self.dataclose = self.datas[0].close

        # 计算移动平均线
        self.short_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.short_ma)
        self.long_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.long_ma)
        
        # 使用 bt.indicators.CrossOver 来判断金叉和死叉
        # a.crossover(b) > 0 表示 a 上穿 b (金叉)
        # a.crossover(b) < 0 表示 a 下穿 b (死叉)
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)

    def next(self):
        # 这是策略的核心，每个交易日都会被调用一次
        
        # 如果我们已经有持仓
        if self.position:
            # 检查是否出现死叉
            if self.crossover < 0:
                print(f'{self.datas[0].datetime.date(0)}: 死叉信号，卖出！')
                self.close() # 平掉所有仓位
        
        # 如果我们没有持仓
        elif self.crossover > 0:
            # 检查是否出现金叉
            print(f'{self.datas[0].datetime.date(0)}: 金叉信号，买入！')
            # 计算可以买入的股数（这里用95%的资金，留一点现金）
            size = int(self.broker.get_cash() / self.datas[0].close[0] * 0.95)
            self.buy(size=size)

# --- 第2步: 准备数据 ---
print("正在获取历史数据...")
# 获取沪深300指数从2015年至今的数据
try:
    df = ak.stock_zh_index_daily(symbol="sh000300")
    # 确保日期是正确的datetime类型
    df['date'] = pd.to_datetime(df['date'])
    # 先按日期排序
    df = df.sort_values('date')
    # 然后筛选日期
    df = df[df['date'] > '2015-01-01']
    # 重新设置索引
    df.set_index('date', inplace=True)
    
    # Backtrader 需要的格式：datetime, open, high, low, close, volume
    df = df[['open', 'high', 'low', 'close', 'volume']]
    # 将数据喂给 Backtrader
    data = bt.feeds.PandasData(dataname=df)
    print("✅ 数据准备完成！")

except Exception as e:
    print(f"❌ 数据获取失败: {e}")
    exit()

# --- 第3步: 配置回测引擎 ---
# 创建Cerebro引擎实例
cerebro = bt.Cerebro()

# 将数据添加到引擎中
cerebro.adddata(data)

# 将我们写好的策略添加到引擎中
cerebro.addstrategy(MovingAverageCrossStrategy)

# 设置初始资金
cerebro.broker.setcash(1000000.0)

# 设置交易手续费（这里设置为万分之三）
cerebro.broker.setcommission(commission=0.0003)

# --- 第4步: 运行回测并分析结果 ---
print("\n--- 回测开始 ---")
# 记录回测开始时的资金
start_value = cerebro.broker.getvalue()
print(f'初始资金: {start_value:,.2f}')

# 运行回测
cerebro.run()

# 记录回测结束时的资金
end_value = cerebro.broker.getvalue()
print(f'最终资金: {end_value:,.2f}')

# 计算年化收益率
portfolio_return = (end_value / start_value) - 1
time_period_years = (df.index[-1] - df.index[0]).days / 365.25
annualized_return = (1 + portfolio_return) ** (1 / time_period_years) - 1

print(f"策略总收益率: {portfolio_return:.2%}")
print(f"策略年化收益率: {annualized_return:.2%}")
print("--- 回测结束 ---\n")

# --- 第5步: 绘制结果图 ---
print("正在绘制结果图...")
# 设置图表大小
plt.rcParams['figure.figsize'] = [15, 12]
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

cerebro.plot(style='candlestick')