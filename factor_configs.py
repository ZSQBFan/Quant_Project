# factor_configs.py
"""
【【【因子注册与配置中心】】】
这是定义、配置和注册所有因子的唯一地方。

每个因子的配置结构:
'FactorName': {
    'params': { ... },                 # 因子计算所需的参数
    'required_columns': [ ... ],       # 因子计算所需的数据列 (使用最终在DataFrame中看到的列名)
    'description': "..."               # (可选) 因子的简要说明
}
"""

FACTOR_REGISTRY = {
    # ==========================================================================
    # 1. 现有量价因子 (Type 1 & Type 2)
    # ==========================================================================
    'Momentum': {
        'params': {
            'period': 20
        },
        'required_columns': ['close'],
        'description': "计算N日价格变化率"
    },
    'Reversal20D': {
        'params': {
            'period': 40,
            'decay': 20.0
        },
        'required_columns': ['close'],
        'description': "计算20日加权反转因子"
    },
    'RSI': {
        'params': {
            'rsi_period': 14
        },
        'required_columns': ['close'],
        'description': "相对强弱指数"
    },
    'BollingerBands': {
        'params': {
            'period': 20,
            'std_dev': 2.0
        },
        'required_columns': ['close'],
        'description': "布林带指标"
    },
    'IndNeu_Momentum': {
        # 这是一个复合因子，它需要基础价格数据和行业数据
        'params': {
            'period': 20
        },
        'required_columns': ['close', 'industry'],
        'description': "行业中性的动量因子"
    },
    'IndNeu_Reversal20D': {
        'params': {},
        'required_columns': ['close', 'industry'],
        'description': "行业中性的反转因子"
    },
    'IndNeu_VolumeCV': {
        'params': {},
        'required_columns': ['volume', 'industry'],
        'description': "行业中性的成交量变异系数"
    },
    'IndNeu_ADXDMI': {
        'params': {},
        'required_columns': ['high', 'low', 'close', 'industry'],
        'description': "行业中性的ADX/DMI趋势强度"
    }
}
