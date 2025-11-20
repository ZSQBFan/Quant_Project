# factor_configs.py
"""
【【【因子注册与配置中心】】】
每个因子的配置结构:
'FactorName': {
    'category': 'simple' | 'complex',  # 【新增】决定计算引擎
    'params': { ... },
    'required_columns': [ ... ],
    'description': "..."
}
"""

FACTOR_REGISTRY = {
    # ==========================================================================
    # 1. 量价因子 (Category: simple)
    #    -> 使用 FactorCalculator (多进程，逐只计算)
    # ==========================================================================
    'Momentum': {
        'category': 'simple',
        'params': {
            'period': 20
        },
        'required_columns': ['close'],
        'description': "计算N日价格变化率"
    },
    'Reversal20D': {
        'category': 'simple',
        'params': {
            'period': 40,
            'decay': 20.0
        },
        'required_columns': ['close'],
        'description': "计算20日加权反转因子"
    },
    'RSI': {
        'category': 'simple',
        'params': {
            'rsi_period': 14
        },
        'required_columns': ['close'],
        'description': "相对强弱指数"
    },
    'BollingerBands': {
        'category': 'simple',
        'params': {
            'period': 20,
            'std_dev': 2.0
        },
        'required_columns': ['close'],
        'description': "布林带指标"
    },

    # ==========================================================================
    # 2. 复合/截面因子 (Category: complex)
    #    -> 使用 data_manager.get_all_data (全量宽表，向量化计算)
    # ==========================================================================
    'IndNeu_Momentum': {
        'category': 'complex',
        'params': {
            'period': 20
        },
        'required_columns': ['close', 'industry'],
        'description': "行业中性的动量因子"
    },
    'IndNeu_Reversal20D': {
        'category': 'complex',
        'params': {},
        'required_columns': ['close', 'industry'],
        'description': "行业中性的反转因子"
    },
    'IndNeu_VolumeCV': {
        'category': 'complex',
        'params': {},
        'required_columns': ['volume', 'industry'],
        'description': "行业中性的成交量变异系数"
    },

    # ==========================================================================
    # 3. 基本面因子
    # ==========================================================================
    'IndNeu_EP': {
        'category':
        'complex',
        'params': {},
        'required_columns':
        ['close', 'share_capital', 'net_profit_parent', 'industry'],
        'description':
        "行业中性的市盈率倒数 (E/P)"
    },
    'IndNeu_BP': {
        'category':
        'complex',
        'params': {},
        'required_columns':
        ['close', 'share_capital', 'total_equity_parent', 'industry'],
        'description':
        "行业中性的市净率倒数 (B/P)"
    },
    'IndNeu_ROE': {
        'category':
        'complex',
        'params': {},
        'required_columns':
        ['net_profit_parent', 'total_equity_parent', 'industry'],
        'description':
        "行业中性的净资产收益率 (ROE)"
    },
    # 'IndNeu_SalesGrowth': {
    #     'category': 'complex',
    #     'params': {},
    #     'required_columns': [],
    #     'description': "行业中性的营收同比增长率 (YoY)"
    # },
    'IndNeu_CFOP': {
        'category':
        'complex',
        'params': {},
        'required_columns':
        ['net_cash_flow_ops', 'close', 'share_capital', 'industry'],
        'description':
        "行业中性的经营现金流市价率 (CFO/P)"
    },
    'IndNeu_GPM': {
        'category': 'complex',
        'params': {},
        'required_columns':
        ['total_revenue', 'cost_of_goods_sold', 'industry'],
        'description': "行业中性的毛利率 (Gross Profit Margin)"
    },
    'IndNeu_AssetTurnover': {
        'category': 'complex',
        'params': {},
        'required_columns': ['total_revenue', 'total_assets', 'industry'],
        'description': "行业中性的总资产周转率"
    },
    'IndNeu_CurrentRatio': {
        'category': 'complex',
        'params': {},
        'required_columns':
        ['current_assets', 'current_liabilities', 'industry'],
        'description': "行业中性的流动比率 (偿债能力)"
    },
}
