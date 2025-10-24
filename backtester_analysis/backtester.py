# backtester.py
import backtrader as bt

class Backtester:
    def __init__(self, strategy_class, cash=1000000.0, commission=0.0003, strategy_params={}):
        self.cerebro = bt.Cerebro()
        self.strategy_class = strategy_class
        self.cash = cash
        self.commission = commission
        self.strategy_params = strategy_params
        self.results = None

    def run(self):
        """
        执行回测并保存结果。
        """
        print("\n--- 回测开始 ---")
        
        self.cerebro.addstrategy(self.strategy_class, **self.strategy_params)
        self.cerebro.broker.setcash(self.cash)
        self.cerebro.broker.setcommission(commission=self.commission)

        # 仍然需要PyFolio分析器来提取数据
        self.cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
        
        start_value = self.cerebro.broker.getvalue()
        print(f'初始资金: {start_value:,.2f}')
        
        # 将运行结果保存到 self.results
        run_results = self.cerebro.run()
        self.results = run_results[0]

        end_value = self.cerebro.broker.getvalue()
        print(f'最终资金: {end_value:,.2f}')
        print("--- 回测结束 ---\n")