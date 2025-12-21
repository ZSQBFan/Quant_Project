#!/usr/bin/env python3
"""
测试脚本：验证 RollingRegressionCalculator 在异常情况下的行为

该脚本模拟以下异常情况：
- 因子值为常量
- 收益率为常量
- 数据点不足
- 正常数据
- 负权重场景（混合相关性）

验证在这些异常情况下，RollingRegressionCalculator 是否能正确触发警告并退回到等权分配。
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from io import StringIO
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factors.pipeline.combiners.rolling.rolling_regression import RollingRegressionCalculator

# 配置日志捕获
def setup_logging():
    """设置日志配置以捕获警告信息"""
    log_capture_string = StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.WARNING)
    
    # 获取根记录器
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    logger.addHandler(ch)
    
    return log_capture_string

def get_logged_warnings(log_capture_string):
    """获取捕获的警告日志"""
    log_contents = log_capture_string.getvalue()
    log_capture_string.seek(0)
    log_capture_string.truncate(0)
    return log_contents

def create_test_data(date, factor_names, num_stocks=10, signal_strength=1.0, correlation_type='positive'):
    """创建测试数据
    Args:
        signal_strength: 信号强度，1.0为正常，>1.0为强信号
        correlation_type: 'positive'(正相关), 'negative'(负相关), 'mixed'(混合相关)
    """
    np.random.seed(42)  # 确保可重现性
    
    # 创建多级索引
    dates = [date] * num_stocks
    stock_codes = [f"stock_{i:03d}" for i in range(num_stocks)]
    index = pd.MultiIndex.from_arrays([dates, stock_codes], names=['date', 'asset'])
    
    data = {}
    
    # 创建正常因子数据
    for factor in factor_names:
        data[factor] = np.random.normal(0, 1, num_stocks)
    
    # 创建收益率数据，添加与因子的相关性
    returns = np.random.normal(0.02, 0.05, num_stocks)
    
    # 根据相关性类型添加因子信号
    if correlation_type == 'positive':
        # 所有因子都与收益率正相关
        if len(factor_names) >= 2:
            returns += signal_strength * (data[factor_names[0]] * 0.3 + data[factor_names[1]] * 0.2)
    elif correlation_type == 'negative':
        # 所有因子都与收益率负相关
        if len(factor_names) >= 2:
            returns += signal_strength * (-data[factor_names[0]] * 0.3 - data[factor_names[1]] * 0.2)
    elif correlation_type == 'mixed':
        # 混合相关性：第一个因子正相关，第二个因子负相关
        if len(factor_names) >= 2:
            returns += signal_strength * (data[factor_names[0]] * 0.3 - data[factor_names[1]] * 0.2)
    
    data['forward_return_5d'] = returns
    
    return pd.DataFrame(data, index=index)

def test_empty_data_window():
    """测试 1: 空数据窗口"""
    print("=" * 60)
    print("测试 1: 空数据窗口")
    print("=" * 60)
    
    log_capture = setup_logging()
    
    factor_names = ['factor_a', 'factor_b']
    calculator = RollingRegressionCalculator(
        factor_names=factor_names,
        target_return_period=5,
        rolling_window_days=20
    )
    
    current_date = pd.Timestamp('2021-01-01')
    empty_data = pd.DataFrame()
    
    result = calculator._calculate_payload_for_day(empty_data, current_date)
    
    # 验证结果
    assert result is None, f"期望返回None，但得到 {result}"
    
    # 检查日志
    warnings = get_logged_warnings(log_capture)
    assert "历史数据窗口为空" in warnings, f"期望包含'历史数据窗口为空'的警告，但得到: {warnings}"
    
    print("✅ 空数据窗口测试通过")
    print(f"⚠️ 警告信息: {warnings.strip()}")
    print()

def test_insufficient_data():
    """测试 2: 数据点不足"""
    print("=" * 60)
    print("测试 2: 数据点不足")
    print("=" * 60)
    
    log_capture = setup_logging()
    
    factor_names = ['factor_a', 'factor_b', 'factor_c']
    calculator = RollingRegressionCalculator(
        factor_names=factor_names,
        target_return_period=5,
        rolling_window_days=20
    )
    
    current_date = pd.Timestamp('2021-01-01')
    # 创建只有2行数据，但有3个因子的情况
    insufficient_data = create_test_data(current_date, factor_names, num_stocks=2)
    
    result = calculator._calculate_payload_for_day(insufficient_data, current_date)
    
    # 验证结果
    assert result is None, f"期望返回None，但得到 {result}"
    
    # 检查日志
    warnings = get_logged_warnings(log_capture)
    assert "数据点不足" in warnings, f"期望包含'数据点不足'的警告，但得到: {warnings}"
    
    print("✅ 数据点不足测试通过")
    print(f"⚠️ 警告信息: {warnings.strip()}")
    print()

def test_constant_factor_values():
    """测试 3: 因子值为常量"""
    print("=" * 60)
    print("测试 3: 因子值为常量")
    print("=" * 60)
    
    log_capture = setup_logging()
    
    factor_names = ['factor_a', 'factor_b']
    calculator = RollingRegressionCalculator(
        factor_names=factor_names,
        target_return_period=5,
        rolling_window_days=20
    )
    
    current_date = pd.Timestamp('2021-01-01')
    test_data = create_test_data(current_date, factor_names, num_stocks=10)
    
    # 设置 factor_a 为常量
    test_data['factor_a'] = 5.0
    
    result = calculator._calculate_payload_for_day(test_data, current_date)
    
    # 验证结果：当只有1个有效因子时，返回None跳过计算
    assert result is None, f"期望返回None，但得到 {result}"
    
    # 检查日志
    warnings = get_logged_warnings(log_capture)
    assert "因子 factor_a 值为常量" in warnings, f"期望包含'因子 factor_a 值为常量'的警告，但得到: {warnings}"
    # 在这种情况下，有效因子数量不足，应该返回None
    assert "有效因子数量不足" in warnings, f"期望包含'有效因子数量不足'的警告，但得到: {warnings}"
    
    print("✅ 因子值为常量测试通过")
    print(f"⚠️ 警告信息: {warnings.strip()}")
    print()

def test_constant_return_values():
    """测试 4: 收益率为常量"""
    print("=" * 60)
    print("测试 4: 收益率为常量")
    print("=" * 60)
    
    log_capture = setup_logging()
    
    factor_names = ['factor_a', 'factor_b']
    calculator = RollingRegressionCalculator(
        factor_names=factor_names,
        target_return_period=5,
        rolling_window_days=20
    )
    
    current_date = pd.Timestamp('2021-01-01')
    test_data = create_test_data(current_date, factor_names, num_stocks=10)
    
    # 设置收益率为常量
    test_data['forward_return_5d'] = 0.02
    
    result = calculator._calculate_payload_for_day(test_data, current_date)
    
    # 验证结果：没有有效因子，应该返回None跳过计算
    assert result is None, f"期望返回None，但得到 {result}"
    
    # 检查日志
    warnings = get_logged_warnings(log_capture)
    assert "收益率 forward_return_5d 为常量" in warnings, f"期望包含'收益率为常量'的警告，但得到: {warnings}"
    
    print("✅ 收益率为常量测试通过")
    print(f"⚠️ 警告信息: {warnings.strip()}")
    print()

def test_normal_data():
    """测试 5: 正常数据"""
    print("=" * 60)
    print("测试 5: 正常数据")
    print("=" * 60)
    
    log_capture = setup_logging()
    
    factor_names = ['factor_a', 'factor_b']
    calculator = RollingRegressionCalculator(
        factor_names=factor_names,
        target_return_period=5,
        rolling_window_days=20
    )
    
    current_date = pd.Timestamp('2021-01-01')
    # 创建有强信号的数据
    test_data = create_test_data(current_date, factor_names, num_stocks=30, signal_strength=3.0)
    
    result = calculator._calculate_payload_for_day(test_data, current_date)
    
    # 验证结果：权重绝对值总和应该为1
    weight_sum = sum(result.values())
    abs_weight_sum = sum(abs(v) for v in result.values())
    assert abs(abs_weight_sum - 1.0) < 1e-6, f"权重绝对值总和应为1，但得到 {abs_weight_sum}"
    
    # 权重应该合理分布（不一定是严格的非等权，但应该是有效的）
    assert all(v >= 0 for v in result.values()), f"所有权重应该非负: {result}"
    
    # 检查没有错误性警告
    warnings = get_logged_warnings(log_capture)
    assert "历史数据窗口为空" not in warnings, f"不应该有空数据警告，但得到: {warnings}"
    assert "数据点不足" not in warnings, f"不应该有数据不足警告，但得到: {warnings}"
    
    print("✅ 正常数据测试通过")
    print(f"计算得到的权重: {result}")
    print(f"权重总和: {weight_sum:.6f}")
    print(f"权重绝对值之和: {abs_weight_sum:.6f}")
    if warnings.strip():
        print(f"ℹ️ 日志信息: {warnings.strip()}")
    else:
        print("ℹ️ 无警告信息（正常情况）")
    print()

def test_negative_weights_mixed_correlation():
    """测试 9: 负权重场景 - 混合相关性（正相关 + 负相关）"""
    print("=" * 60)
    print("测试 9: 负权重场景 - 混合相关性")
    print("=" * 60)
    
    log_capture = setup_logging()
    
    factor_names = ['factor_a', 'factor_b']
    calculator = RollingRegressionCalculator(
        factor_names=factor_names,
        target_return_period=5,
        rolling_window_days=20
    )
    
    current_date = pd.Timestamp('2021-01-01')
    # 创建混合相关性数据：factor_a与收益率正相关，factor_b与收益率负相关
    test_data = create_test_data(
        current_date, 
        factor_names, 
        num_stocks=50, 
        signal_strength=4.0, 
        correlation_type='mixed'
    )
    
    result = calculator._calculate_payload_for_day(test_data, current_date)
    
    # 验证结果：权重绝对值总和应该为1（使用绝对值归一化）
    weight_sum = sum(result.values())
    
    # 验证归一化后的权重绝对值之和是否为1.0
    abs_weight_sum = sum(abs(v) for v in result.values())
    assert abs(abs_weight_sum - 1.0) < 1e-6, f"归一化后的权重绝对值之和应为1.0，但得到 {abs_weight_sum}"
    
    # 检查没有"权重归一化异常"警告
    warnings = get_logged_warnings(log_capture)
    assert "权重归一化异常" not in warnings, f"不应该有'权重归一化异常'警告，但得到: {warnings}"
    
    # 检查没有其他错误性警告
    assert "历史数据窗口为空" not in warnings, f"不应该有空数据警告，但得到: {warnings}"
    assert "数据点不足" not in warnings, f"不应该有数据不足警告，但得到: {warnings}"
    
    print("✅ 负权重场景 - 混合相关性测试通过")
    print(f"计算得到的权重: {result}")
    print(f"权重总和: {weight_sum:.6f}")
    print(f"权重绝对值之和: {abs_weight_sum:.6f}")
    if warnings.strip():
        print(f"ℹ️ 日志信息: {warnings.strip()}")
    else:
        print("ℹ️ 无警告信息（正常情况）")
    print()

def test_negative_weights_pure_negative():
    """测试 10: 负权重场景 - 纯负相关性"""
    print("=" * 60)
    print("测试 10: 负权重场景 - 纯负相关性")
    print("=" * 60)
    
    log_capture = setup_logging()
    
    factor_names = ['factor_a', 'factor_b']
    calculator = RollingRegressionCalculator(
        factor_names=factor_names,
        target_return_period=5,
        rolling_window_days=20
    )
    
    current_date = pd.Timestamp('2021-01-01')
    # 创建纯负相关性数据：所有因子都与收益率负相关
    test_data = create_test_data(
        current_date, 
        factor_names, 
        num_stocks=50, 
        signal_strength=3.0, 
        correlation_type='negative'
    )
    
    result = calculator._calculate_payload_for_day(test_data, current_date)
    
    # 验证结果：权重绝对值总和应该为1（使用绝对值归一化）
    weight_sum = sum(result.values())
    
    # 验证归一化后的权重绝对值之和是否为1.0
    abs_weight_sum = sum(abs(v) for v in result.values())
    assert abs(abs_weight_sum - 1.0) < 1e-6, f"归一化后的权重绝对值之和应为1.0，但得到 {abs_weight_sum}"
    
    # 检查没有"权重归一化异常"警告
    warnings = get_logged_warnings(log_capture)
    assert "权重归一化异常" not in warnings, f"不应该有'权重归一化异常'警告，但得到: {warnings}"
    
    # 检查没有其他错误性警告
    assert "历史数据窗口为空" not in warnings, f"不应该有空数据警告，但得到: {warnings}"
    assert "数据点不足" not in warnings, f"不应该有数据不足警告，但得到: {warnings}"
    
    print("✅ 负权重场景 - 纯负相关性测试通过")
    print(f"计算得到的权重: {result}")
    print(f"权重总和: {weight_sum:.6f}")
    print(f"权重绝对值之和: {abs_weight_sum:.6f}")
    if warnings.strip():
        print(f"ℹ️ 日志信息: {warnings.strip()}")
    else:
        print("ℹ️ 无警告信息（正常情况）")
    print()

def test_three_factors_with_two_constant():
    """测试 6: 三个因子中两个为常量（只有1个有效因子）"""
    print("=" * 60)
    print("测试 6: 三个因子中两个为常量（只有1个有效因子）")
    print("=" * 60)
    
    log_capture = setup_logging()
    
    factor_names = ['factor_a', 'factor_b', 'factor_c']
    calculator = RollingRegressionCalculator(
        factor_names=factor_names,
        target_return_period=5,
        rolling_window_days=20
    )
    
    current_date = pd.Timestamp('2021-01-01')
    test_data = create_test_data(current_date, factor_names, num_stocks=25, signal_strength=1.5)
    
    # 设置 factor_a 和 factor_b 为常量（只有 factor_c 有效）
    test_data['factor_a'] = 5.0
    test_data['factor_b'] = 3.0
    
    result = calculator._calculate_payload_for_day(test_data, current_date)
    
    # 验证结果：当只有1个有效因子时，返回None跳过计算（单因子回归没有意义）
    assert result is None, f"期望返回None，但得到 {result}"
    
    # 检查日志
    warnings = get_logged_warnings(log_capture)
    assert "因子 factor_a 值为常量" in warnings, f"期望包含'因子 factor_a 值为常量'的警告，但得到: {warnings}"
    assert "因子 factor_b 值为常量" in warnings, f"期望包含'因子 factor_b 值为常量'的警告，但得到: {warnings}"
    # 系统检测到有效因子数量不足，返回None
    assert "有效因子数量不足" in warnings, f"期望包含'有效因子数量不足'的警告，但得到: {warnings}"
    
    print("✅ 三个因子中两个为常量测试通过")
    print(f"计算得到的权重: {result}")
    print(f"⚠️ 警告信息: {warnings.strip()}")
    print()

def test_four_factors_with_three_constant():
    """测试 7: 四个因子中三个为常量（有1个有效因子）"""
    print("=" * 60)
    print("测试 7: 四个因子中三个为常量（有1个有效因子）")
    print("=" * 60)
    
    log_capture = setup_logging()
    
    factor_names = ['factor_a', 'factor_b', 'factor_c', 'factor_d']
    calculator = RollingRegressionCalculator(
        factor_names=factor_names,
        target_return_period=5,
        rolling_window_days=20
    )
    
    current_date = pd.Timestamp('2021-01-01')
    test_data = create_test_data(current_date, factor_names, num_stocks=25, signal_strength=1.5)
    
    # 设置 factor_a, factor_b, factor_c 为常量（只有 factor_d 有效）
    test_data['factor_a'] = 5.0
    test_data['factor_b'] = 3.0
    test_data['factor_c'] = 2.0
    
    result = calculator._calculate_payload_for_day(test_data, current_date)
    
    # 验证结果：有1个有效因子，无法进行多变量回归，返回None
    assert result is None, f"期望返回None，但得到 {result}"
    
    # 检查日志
    warnings = get_logged_warnings(log_capture)
    assert "因子 factor_a 值为常量" in warnings, f"期望包含'因子 factor_a 值为常量'的警告，但得到: {warnings}"
    assert "因子 factor_b 值为常量" in warnings, f"期望包含'因子 factor_b 值为常量'的警告，但得到: {warnings}"
    assert "因子 factor_c 值为常量" in warnings, f"期望包含'因子 factor_c 值为常量'的警告，但得到: {warnings}"
    assert "有效因子数量不足" in warnings, f"期望包含'有效因子数量不足'的警告，但得到: {warnings}"
    
    print("✅ 四个因子中三个为常量测试通过")
    print(f"计算得到的权重: {result}")
    print(f"⚠️ 警告信息: {warnings.strip()}")
    print()

def test_data_with_nan():
    """测试 8: 包含NaN数据"""
    print("=" * 60)
    print("测试 8: 包含NaN数据")
    print("=" * 60)
    
    log_capture = setup_logging()
    
    factor_names = ['factor_a', 'factor_b']
    calculator = RollingRegressionCalculator(
        factor_names=factor_names,
        target_return_period=5,
        rolling_window_days=20
    )
    
    current_date = pd.Timestamp('2021-01-01')
    test_data = create_test_data(current_date, factor_names, num_stocks=30, signal_strength=2.0)
    
    # 在数据中插入一些NaN值
    test_data.loc[test_data.index[0:5], 'factor_a'] = np.nan
    test_data.loc[test_data.index[10:15], 'forward_return_5d'] = np.nan
    
    result = calculator._calculate_payload_for_day(test_data, current_date)
    
    # 验证结果：dropna()后应该能正常计算
    weight_sum = sum(result.values())
    abs_weight_sum = sum(abs(v) for v in result.values())
    assert abs(abs_weight_sum - 1.0) < 1e-6, f"权重绝对值总和应为1，但得到 {abs_weight_sum}"
    assert all(v >= 0 for v in result.values()), f"所有权重应该非负: {result}"
    
    # 检查没有错误性警告
    warnings = get_logged_warnings(log_capture)
    assert "历史数据窗口为空" not in warnings, f"不应该有空数据警告，但得到: {warnings}"
    assert "数据点不足" not in warnings, f"不应该有数据不足警告，但得到: {warnings}"
    
    print("✅ 包含NaN数据测试通过")
    print(f"计算得到的权重: {result}")
    print(f"权重总和: {weight_sum:.6f}")
    print(f"权重绝对值之和: {abs_weight_sum:.6f}")
    if warnings.strip():
        print(f"ℹ️ 日志信息: {warnings.strip()}")
    else:
        print("ℹ️ 无警告信息（正常情况）")
    print()

def run_all_tests():
    """运行所有测试"""
    print("🚀 开始运行 RollingRegressionCalculator 异常处理验证测试")
    print()
    
    try:
        test_empty_data_window()
        test_insufficient_data()
        test_constant_factor_values()
        test_constant_return_values()
        test_normal_data()
        test_negative_weights_mixed_correlation()
        test_negative_weights_pure_negative()
        test_three_factors_with_two_constant()
        test_four_factors_with_three_constant()
        test_data_with_nan()
        
        print("=" * 60)
        print("🎉 所有测试通过！")
        print("✅ RollingRegressionCalculator 在各种异常情况下都能正确:")
        print("   1. 触发适当的警告信息")
        print("   2. 返回None跳过异常日期的计算")
        print("   3. 保证回测结果的纯净性，不掺杂其他合成方法")
        print("   4. 确保程序稳定性")
        print("   5. 正确处理负权重场景，不再抛出'权重归一化异常'警告")
        print("   6. 验证归一化后的权重绝对值之和为1.0")
        print("=" * 60)
        
        return True
        
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试出现异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)