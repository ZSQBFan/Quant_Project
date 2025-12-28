# bt/data/feeds.py

"""
Backtrader 自定义 DataFeed

将包含合成因子的 DataFrame 加载到 Backtrader
"""

import backtrader as bt


class FactorPandasData(bt.feeds.PandasData):
    """
    扩展的 PandasData，支持加载合成因子、停牌标记和行业信息

    自定义数据线 (Lines):
    - combined_signal: 合成后的大因子信号值
    - suspended: 停牌标记 (True/False)
    - industry: 行业分类信息

    使用示例:
        df = pd.read_parquet('000001.parquet')
        data = FactorPandasData(dataname=df, name='000001')
        cerebro.adddata(data)
    """

    # 添加自定义数据线
    lines = ('combined_signal', 'suspended', 'industry',)

    # 参数定义：列名映射
    params = (
        ('datetime', None),      # 索引自动作为日期
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', 'openinterest'),
        # 自定义列映射
        ('combined_signal', 'combined_signal'),
        ('suspended', 'suspended'),
        ('industry', 'industry'),
    )