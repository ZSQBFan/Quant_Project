#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试各种滚动策略在预热期的表现
"""

import pandas as pd
import numpy as np
import logging
from strategies.rolling_calculators import RollingICIRCalculator, RollingRegressionCalculator
from core.abstractions import RollingCalculatorBase
from sklearn.linear_model import LinearRegression

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_data_with_constants():
    """创建包含更多常量数据的测试集"""
    # 创建日期范围
    dates = pd.date_range('2023-01-01', '2023-02-15', freq='D')
    
    # 创建股票代码
    assets = ['STOCK_001', 'STOCK_002', 'STOCK_003']
    
    # 创建多索引
    index = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    
    # 创建DataFrame
    data = pd.DataFrame(index=index)
    
    # 添加因子数据 - 全部为常量（极端情况）
    data['factor_1'] = 0.5  # 所有值都是0.5
    
    # 添加收益率数据 - 全部为常量
    data['forward_return_5d'] = 0.001
    
    return data

def create_test_data_with_variations():
    """创建有变化的数据作为对比"""
    # 创建日期范围
    dates = pd.date_range('2023-01-01', '2023-02-15', freq='D')
    
    # 创建股票代码
    assets = ['STOCK_001', 'STOCK_002', 'STOCK_003']
    
    # 创建多索引
    index = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    
    # 创建DataFrame
    data = pd.DataFrame(index=index)
    
    # 添加有变化的因子数据
    data['factor_1'] = np.random.normal(0, 1, len(data))
    
    # 添加有变化的收益率数据
    data['forward_return_5d'] = np.random.normal(0.001, 0.02, len(data))
    
    return data

def test_rolling_regression_with_constants():
    """测试RollingRegressionCalculator在常量数据下的表现"""
    print("=" * 60)
    print("测试 RollingRegressionCalculator 在常量数据下的表现")
    print("=" * 60)
    
    # 测试常量数据
    test_data_const = create_test_data_with_constants()
    print(f"常量数据形状: {test_data_const.shape}")
    print(f"因子值唯一值数量: {test_data_const['factor_1'].nunique()}")
    print(f"收益率唯一值数量: {test_data_const['forward_return_5d'].nunique()}")
    
    calculator = RollingRegressionCalculator(
        target_return_period=5,
        factor_names=['factor_1'],
        rolling_window_days=20,
        rebalance_frequency='MS'
    )
    
    try:
        composite = calculator.calculate_composite_factor(test_data_const)
        print(f"✅ 常量数据 - 计算成功，结果形状: {composite.shape}")
        print(f"前3个值: {composite.head(3).values}")
    except Exception as e:
        print(f"❌ 常量数据 - 计算失败: {e}")
    
    # 测试有变化的数据
    test_data_var = create_test_data_with_variations()
    print(f"\n有变化数据形状: {test_data_var.shape}")
    print(f"因子值唯一值数量: {test_data_var['factor_1'].nunique()}")
    print(f"收益率唯一值数量: {test_data_var['forward_return_5d'].nunique()}")
    
    try:
        composite = calculator.calculate_composite_factor(test_data_var)
        print(f"✅ 有变化数据 - 计算成功，结果形状: {composite.shape}")
        print(f"前3个值: {composite.head(3).values}")
    except Exception as e:
        print(f"❌ 有变化数据 - 计算失败: {e}")

def test_linear_regression_with_constants():
    """直接测试LinearRegression在常量数据下的表现"""
    print("\n" + "=" * 60)
    print("直接测试 LinearRegression 在常量数据下的表现")
    print("=" * 60)
    
    # 创建常量数据
    X_const = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]])
    y_const = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    
    print(f"X 值: {X_const.flatten()}")
    print(f"y 值: {y_const}")
    
    try:
        model = LinearRegression().fit(X_const, y_const)
        print(f"✅ LinearRegression 成功，系数: {model.coef_}")
        print(f"   截距: {model.intercept_}")
    except Exception as e:
        print(f"❌ LinearRegression 失败: {e}")

def test_linear_regression_with_variations():
    """直接测试LinearRegression在有变化数据下的表现"""
    print("\n" + "=" * 60)
    print("直接测试 LinearRegression 在有变化数据下的表现")
    print("=" * 60)
    
    # 创建有变化的数据
    X_var = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y_var = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    
    print(f"X 值: {X_var.flatten()}")
    print(f"y 值: {y_var}")
    
    try:
        model = LinearRegression().fit(X_var, y_var)
        print(f"✅ LinearRegression 成功，系数: {model.coef_}")
        print(f"   截距: {model.intercept_}")
    except Exception as e:
        print(f"❌ LinearRegression 失败: {e}")

def test_spearman_with_constants():
    """直接测试spearmanr在常量数据下的表现"""
    print("\n" + "=" * 60)
    print("直接测试 spearmanr 在常量数据下的表现")
    print("=" * 60)
    
    from scipy.stats import spearmanr
    
    # 创建常量数据
    x_const = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    y_const = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    
    print(f"x 值: {x_const}")
    print(f"y 值: {y_const}")
    
    try:
        # 捕获警告
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            corr, _ = spearmanr(x_const, y_const)
            if w:
                for warning in w:
                    print(f"⚠️ 警告: {warning.message}")
            print(f"✅ spearmanr 完成，相关系数: {corr}")
    except Exception as e:
        print(f"❌ spearmanr 失败: {e}")

if __name__ == "__main__":
    print("开始测试各种滚动策略在不同数据下的表现...\n")
    
    # 测试回归模型
    test_linear_regression_with_constants()
    test_linear_regression_with_variations()
    
    # 测试spearman
    test_spearman_with_constants()
    
    # 测试滚动策略
    test_rolling_regression_with_constants()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
