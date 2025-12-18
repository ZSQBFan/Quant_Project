#!/usr/bin/env python3
"""
RSI因子修复验证测试脚本

验证factor_analyze模式下RSI因子的修复是否有效。
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_rsi_factor_fix.log')
    ]
)

logger = logging.getLogger(__name__)

def create_test_data(symbol: str = "600000", days: int = 30) -> pd.DataFrame:
    """
    创建测试用的股票数据
    
    Args:
        symbol: 股票代码
        days: 天数
    
    Returns:
        包含OHLCV数据的DataFrame
    """
    # 生成日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 创建模拟价格数据
    np.random.seed(42)  # 确保结果可重现
    
    # 生成基础价格序列
    base_price = 10.0
    returns = np.random.normal(0.001, 0.02, len(dates))  # 日收益率
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 创建OHLCV数据
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # 生成日内波动
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else close
        
        # 生成成交量
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    logger.info(f"✅ 创建测试数据: {symbol}, {len(df)} 天数据")
    return df

def test_rsi_factor_basic():
    """
    基础RSI因子测试 - 直接测试因子类
    """
    logger.info("🧪 开始基础RSI因子测试...")
    
    try:
        # 导入RSI因子类
        from factors.library.rsi import RSIFactor
        
        # 创建测试数据
        test_df = create_test_data()
        
        # 创建RSI因子实例
        factor = RSIFactor(params={'rsi_period': 14})
        
        # 计算因子值
        factor_values = factor.calculate(test_df)
        
        # 验证结果
        assert isinstance(factor_values, pd.Series), "因子计算应返回Series"
        assert len(factor_values) == len(test_df), "因子值数量应与输入数据一致"
        assert not factor_values.isna().all(), "不应全部为NaN值"
        assert factor_values.min() >= -50 and factor_values.max() <= 50, "RSI因子值应在合理范围内"
        
        logger.info(f"✅ 基础RSI因子测试通过")
        logger.info(f"   因子值范围: [{factor_values.min():.2f}, {factor_values.max():.2f}]")
        logger.info(f"   有效数据点: {len(factor_values.dropna())}/{len(factor_values)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 基础RSI因子测试失败: {e}", exc_info=True)
        return False

def test_rsi_factor_calculator():
    """
    RSI因子计算器测试 - 测试完整的factor_analyze流程
    """
    logger.info("🧪 开始RSI因子计算器测试...")
    
    try:
        from factors.analysis.calculator import FactorCalculator
        
        # 创建测试数据并保存到临时文件
        test_df = create_test_data()
        temp_db_path = "/tmp/test_rsi_data.db"
        
        # 保存测试数据到临时CSV文件
        test_df.to_csv(f'{temp_db_path}.csv')
        
        # 配置参数
        provider_configs = {}
        
        universe = ['600000']
        start_date = test_df.index.min().strftime('%Y-%m-%d')
        end_date = test_df.index.max().strftime('%Y-%m-%d')
        factor_name = 'RSI'
        factor_params = {'rsi_period': 14}
        required_columns = ['close']
        
        # 由于DataProviderManager需要真实的数据库，我们直接测试_process_single_symbol方法
        class MockDataManager:
            def __init__(self, df):
                self.test_df = df
            
            def get_dataframe(self, symbol, columns=None):
                return self.test_df
        
        # 创建因子计算器实例（简化版）
        calculator = FactorCalculator(
            provider_configs=provider_configs,
            db_path=temp_db_path,
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            factor_name=factor_name,
            factor_params=factor_params,
            num_threads=1,
            required_columns=required_columns
        )
        
        # 替换data_manager为模拟实例
        calculator.data_manager = MockDataManager(test_df)
        
        # 直接测试_process_single_symbol方法，避免多进程问题
        logger.info("🔄 执行_process_single_symbol测试...")
        result_df = calculator._process_single_symbol('600000')
        
        # 验证结果
        assert result_df is not None, "计算结果不应为None"
        assert isinstance(result_df, pd.DataFrame), "计算结果应为DataFrame"
        assert not result_df.empty, "计算结果不应为空"
        assert 'factor_value' in result_df.columns, "结果应包含factor_value列"
        assert 'asset' in result_df.columns, "结果应包含asset列"
        
        # 检查因子值
        factor_values = result_df['factor_value']
        assert not factor_values.isna().all(), "不应全部为NaN值"
        assert factor_values.min() >= -50 and factor_values.max() <= 50, "RSI因子值应在合理范围内"
        
        logger.info(f"✅ RSI因子计算器测试通过")
        logger.info(f"   计算了 {len(result_df)} 个数据点")
        logger.info(f"   股票代码: {result_df['asset'].unique()}")
        logger.info(f"   因子值范围: [{factor_values.min():.2f}, {factor_values.max():.2f}]")
        
        # 清理临时文件
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ RSI因子计算器测试失败: {e}", exc_info=True)
        # 清理临时文件
        temp_db_path = "/tmp/test_rsi_data.db"
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
        return False

def test_rsi_registry():
    """
    测试RSI因子注册表功能
    """
    logger.info("🧪 开始RSI因子注册表测试...")
    
    try:
        from core.registry import get_factor
        
        # 获取RSI因子类
        rsi_cls = get_factor('RSI')
        
        assert rsi_cls is not None, "应能获取到RSI因子类"
        assert hasattr(rsi_cls, 'calculate'), "RSI因子类应有calculate方法"
        
        # 创建实例并测试
        factor = rsi_cls(params={'rsi_period': 14})
        test_df = create_test_data()
        factor_values = factor.calculate(test_df)
        
        assert isinstance(factor_values, pd.Series), "计算结果应为Series"
        
        logger.info("✅ RSI因子注册表测试通过")
        logger.info(f"   因子类: {rsi_cls.__name__}")
        logger.info(f"   因子描述: {factor}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ RSI因子注册表测试失败: {e}", exc_info=True)
        return False

def main():
    """
    主测试函数
    """
    logger.info("🚀 开始RSI因子修复验证测试")
    logger.info("=" * 60)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(("基础RSI因子测试", test_rsi_factor_basic()))
    test_results.append(("RSI因子注册表测试", test_rsi_registry()))
    test_results.append(("RSI因子计算器测试", test_rsi_factor_calculator()))
    
    # 汇总结果
    logger.info("=" * 60)
    logger.info("📊 测试结果汇总:")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 60)
    logger.info(f"总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！RSI因子修复验证成功！")
        return True
    else:
        logger.error(f"⚠️  {total - passed} 个测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)