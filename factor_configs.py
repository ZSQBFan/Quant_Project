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
    # 3. 基本面因子 (Category: simple - 虽然涉及财务数据，但计算逻辑通常是单只股票内部的)
    # ==========================================================================
}
