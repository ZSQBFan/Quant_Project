"""
换手率分析器

用于计算回测过程中的换手率指标，衡量组合的交易活跃度。
"""

import backtrader as bt
from collections import defaultdict


class TurnoverAnalyzer(bt.Analyzer):
    """
    换手率分析器
    
    计算换手率以衡量组合的交易活跃度。
    
    换手率计算公式：
    - 单期换手率 = (买入金额 + 卖出金额) / (2 × 组合价值)
    - 年化换手率 = 平均单期换手率 × 252（假设每年252个交易日）
    
    Attributes:
        buy_value (float): 当期买入总金额
        sell_value (float): 当期卖出总金额
        portfolio_value (float): 当期组合总价值
        turnover_list (list): 每期换手率列表
    """
    
    def __init__(self):
        """初始化换手率分析器"""
        super(TurnoverAnalyzer, self).__init__()
        
        # 每期统计
        self.buy_value = 0.0
        self.sell_value = 0.0
        self.portfolio_value = 0.0
        
        # 累计统计
        self.total_buy = 0.0
        self.total_sell = 0.0
        
        # 换手率列表（用于绘图和分析）
        self.turnover_list = []
        
        # 每期换手率详情
        self.period_details = []
        
        # 当前日期
        self.current_date = None
        
    def start(self):
        """开始分析"""
        self.strategy = self.strategy
        
    def notify_order(self, order):
        """
        订单通知钩子
        
        Args:
            order: 订单对象
        """
        # 只处理已完成订单
        if order.status in [order.Completed]:
            if order.isbuy():
                # 买入订单：记录买入金额
                self.buy_value += order.executed.value
            elif order.issell():
                # 卖出订单：记录卖出金额
                self.sell_value += order.executed.value
            
    def next(self):
        """
        每周期执行
        """
        # 获取当前日期
        self.current_date = self.strategy.datetime.date(0)
        
        # 获取当前组合总价值
        self.portfolio_value = self.strategy.broker.getvalue()
        
        # 计算当期换手率
        # 使用公式：(买入金额 + 卖出金额) / (2 × 组合价值)
        # 换手率范围：0 ~ 1
        
        if self.portfolio_value > 0:
            # 单期换手率
            period_turnover = (self.buy_value + self.sell_value) / (2 * self.portfolio_value)
        else:
            period_turnover = 0.0
        
        # 记录换手率
        self.turnover_list.append({
            'date': self.current_date,
            'turnover': period_turnover,
            'buy_value': self.buy_value,
            'sell_value': self.sell_value,
            'portfolio_value': self.portfolio_value
        })
        
        # 更新累计值
        self.total_buy += self.buy_value
        self.total_sell += self.sell_value
        
        # 记录详情
        self.period_details.append({
            'date': self.current_date,
            'turnover': period_turnover,
            'buy_value': self.buy_value,
            'sell_value': self.sell_value,
            'portfolio_value': self.portfolio_value
        })
        
        # 重置当期值
        self.buy_value = 0.0
        self.sell_value = 0.0
        
    def get_analysis(self):
        """
        获取分析结果
        
        Returns:
            dict: 包含以下键的字典：
                - total_turnover: 总换手率
                - avg_turnover: 平均单期换手率
                - annual_turnover: 年化换手率
                - turnover_list: 每期换手率列表
                - period_details: 每期详细数据
        """
        # 计算统计数据
        num_periods = len(self.turnover_list)
        
        if num_periods == 0:
            return {
                'total_turnover': 0.0,
                'avg_turnover': 0.0,
                'annual_turnover': 0.0,
                'turnover_list': [],
                'period_details': []
            }
        
        # 提取换手率列表
        turnover_rates = [p['turnover'] for p in self.turnover_list]
        
        # 平均单期换手率
        avg_turnover = sum(turnover_rates) / num_periods if num_periods > 0 else 0.0
        
        # 年化换手率（假设252个交易日）
        annual_turnover = avg_turnover * 252
        
        # 总换手率（所有期换手率之和）
        total_turnover = sum(turnover_rates)
        
        return {
            'total_turnover': total_turnover,
            'avg_turnover': avg_turnover,
            'annual_turnover': annual_turnover,
            'turnover_list': turnover_rates,
            'period_details': self.period_details,
            'total_buy': self.total_buy,
            'total_sell': self.total_sell
        }
