#!/usr/bin/env python3
"""测试所有标准化器的实现和注册"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from core.registry import get_standardizer, print_registry_stats
from factors.pipeline.standardizers import *

def test_all_standardizers():
    """测试所有标准化器"""
    print("🔧 开始测试所有标准化器...")
    
    # 获取所有注册的标准化器
    from core.registry import registry
    standardizers = registry.get_all('standardizers')
    
    if not standardizers:
        print("❌ 没有找到任何注册的标准化器")
        return False
    
    print(f"✅ 找到 {len(standardizers)} 个注册的标准化器: {list(standardizers.keys())}")
    
    # 创建测试数据
    np.random.seed(42)
    test_data = pd.DataFrame({
        'stock1': np.random.normal(0, 1, 100),
        'stock2': np.random.normal(0.5, 1.2, 100),
        'stock3': np.random.normal(-0.2, 0.8, 100),
    })
    
    success_count = 0
    failed_count = 0
    
    for name, cls in standardizers.items():
        try:
            print(f"\n🧪 测试 {name} 标准化器...")
            
            # 尝试实例化
            if name == 'ZScore':
                instance = cls(winsorize=True, winsorize_limits=[0.01, 0.99])
            elif name == 'Quantile':
                instance = cls(n_quantiles=100)
            elif name == 'MinMax':
                instance = cls(feature_range=[0, 1])
            elif name == 'Rank':
                instance = cls(ascending=True, pct=True)
            elif name == 'MAD':
                instance = cls(scale_factor=1.4826)
            else:
                instance = cls()
            
            print(f"  ✅ 实例化成功: {type(instance).__name__}")
            
            # 测试标准化
            result = instance.standardize(test_data)
            print(f"  ✅ 标准化成功，结果形状: {result.shape}")
            print(f"  📊 结果统计: min={result.min().min():.4f}, max={result.max().max():.4f}, mean={result.mean().mean():.4f}")
            
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 测试失败: {str(e)}")
            failed_count += 1
    
    print(f"\n📈 测试结果汇总:")
    print(f"  ✅ 成功: {success_count}")
    print(f"  ❌ 失败: {failed_count}")
    print(f"  📊 总计: {success_count + failed_count}")
    
    return failed_count == 0

def test_specific_standardizers():
    """测试具体的Rank和MAD标准化器"""
    print("\n🎯 专门测试Rank和MAD标准化器...")
    
    try:
        # 测试Rank标准化器
        rank_cls = get_standardizer('Rank')
        print(f"✅ 成功获取Rank标准化器: {rank_cls}")
        
        rank_instance = rank_cls(ascending=True, pct=True)
        print(f"✅ Rank标准化器实例化成功")
        
        # 创建简单测试数据
        test_data = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
        rank_result = rank_instance.standardize(test_data)
        print(f"✅ Rank标准化结果:\n{rank_result}")
        
    except Exception as e:
        print(f"❌ Rank标准化器测试失败: {str(e)}")
        return False
    
    try:
        # 测试MAD标准化器
        mad_cls = get_standardizer('MAD')
        print(f"✅ 成功获取MAD标准化器: {mad_cls}")
        
        mad_instance = mad_cls(scale_factor=1.4826)
        print(f"✅ MAD标准化器实例化成功")
        
        mad_result = mad_instance.standardize(test_data)
        print(f"✅ MAD标准化结果:\n{mad_result}")
        
    except Exception as e:
        print(f"❌ MAD标准化器测试失败: {str(e)}")
        return False
    
    return True

def main():
    """主函数"""
    print("🚀 标准化器实现验证程序")
    print("=" * 50)
    
    # 打印注册统计
    print_registry_stats()
    
    # 测试所有标准化器
    all_success = test_all_standardizers()
    
    # 专门测试Rank和MAD
    specific_success = test_specific_standardizers()
    
    print("\n" + "=" * 50)
    if all_success and specific_success:
        print("🎉 所有测试通过！Rank和MAD标准化器已成功实现并注册")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查实现")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)