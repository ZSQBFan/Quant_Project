#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试RollingICIRCalculator修复效果的脚本
"""

import pandas as pd
import numpy as np
import logging
from factors.pipeline.combiners.rolling import RollingICIRCalculator
from factors.core.abstractions import RollingCalculatorBase

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_data():
    """创建测试数据，包含预热期的常量数据"""
    # 创建日期范围
    dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
    
    # 创建股票代码
    assets = ['STOCK_001', 'STOCK_002', 'STOCK_003', 'STOCK_004', 'STOCK_005']
    
    # 创建多索引
    index = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    
    # 创建DataFrame
    data = pd.DataFrame(index=index)
    
    # 添加因子数据
    # 因子1：前30天为常量，之后有变化（模拟预热期）
    data['factor_1'] = np.random.normal(0, 1, len(data))
    data.loc[data.index.get_level_values('date') < '2023-02-01', 'factor_1'] = 0.5
    
    # 因子2：正常变化的因子
    data['factor_2'] = np.random.normal(0, 1, len(data))
    
    # 添加收益率数据
    # 前30天收益率为常量，之后有变化
    data['forward_return_5d'] = np.random.normal(0.001, 0.02, len(data))
    data.loc[data.index.get_level_values('date') < '2023-02-01', 'forward_return_5d'] = 0.001
    
    data['forward_return_10d'] = np.random.normal(0.002, 0.03, len(data))
    data.loc[data.index.get_level_values('date') < '2023-02-01', 'forward_return_10d'] = 0.002
    
    return data

def test_rolling_icir_calculator():
    """测试RollingICIRCalculator"""
    print("开始测试RollingICIRCalculator...")
    
    # 创建测试数据
    test_data = create_test_data()
    print(f"测试数据形状: {test_data.shape}")
    
    # 配置因子权重
    factor_weight_config = {
        'factor_1': 'ir_5d',
        'factor_2': 'ir_10d'
    }
    
    # 创建RollingICIRCalculator实例
    calculator = RollingICIRCalculator(
        factor_weight_config=factor_weight_config,
        forward_return_periods=[5, 10],
        factor_names=['factor_1', 'factor_2'],
        rolling_window_days=20,
        rebalance_frequency='MS'
    )
    
    # 计算复合因子
    try:
        composite_factor = calculator.calculate_composite_factor(test_data)
        print(f"✅ 复合因子计算成功，形状: {composite_factor.shape}")
        print(f"复合因子前5个值:\n{composite_factor.head()}")
        return True
    except Exception as e:
        print(f"❌ 复合因子计算失败: {e}")
        return False

if __name__ == "__main__":
    success = test_rolling_icir_calculator()
    if success:
        print("\n🎉 测试通过！RollingICIRCalculator修复成功。")
    else:
        print("\n💥 测试失败！需要进一步调试。")