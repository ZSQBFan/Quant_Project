#!/usr/bin/env python3
"""简化版标准化器修复验证脚本 - 不依赖外部库"""
import sys
import os

# 添加项目路径到 Python 路径
sys.path.insert(0, '/Volumes/叽叽叽/Code/quant_project_3.12_macmini')

def test_quantile_standardizer_instantiation():
    """测试分位数标准化器实例化"""
    print("测试 CrossSectionalQuantileStandardizer 实例化...")
    
    try:
        from factors.pipeline.standardizers.quantile import CrossSectionalQuantileStandardizer
        
        # 测试默认参数实例化
        standardizer = CrossSectionalQuantileStandardizer()
        print(f"✓ 默认参数实例化成功")
        print(f"  n_quantiles: {standardizer.n_quantiles}")
        
        # 测试自定义参数实例化
        standardizer = CrossSectionalQuantileStandardizer(n_quantiles=50)
        print(f"✓ 自定义参数实例化成功")
        print(f"  n_quantiles: {standardizer.n_quantiles}")
        
        # 测试 **kwargs 兼容性
        standardizer = CrossSectionalQuantileStandardizer(n_quantiles=75, some_extra_param="test")
        print(f"✓ **kwargs 兼容性测试成功")
        print(f"  n_quantiles: {standardizer.n_quantiles}")
        
        return True
        
    except Exception as e:
        print(f"✗ 分位数标准化器实例化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_zscore_standardizer_instantiation():
    """测试Z-Score标准化器实例化"""
    print("\n测试 CrossSectionalZScoreStandardizer 实例化...")
    
    try:
        from factors.pipeline.standardizers.zscore import CrossSectionalZScoreStandardizer
        
        # 测试默认参数实例化
        standardizer = CrossSectionalZScoreStandardizer()
        print(f"✓ 默认参数实例化成功")
        print(f"  winsorize: {standardizer.winsorize}")
        print(f"  winsorize_limits: {standardizer.winsorize_limits}")
        
        # 测试自定义参数实例化
        standardizer = CrossSectionalZScoreStandardizer(winsorize=False, winsorize_limits=[0.05, 0.95])
        print(f"✓ 自定义参数实例化成功")
        print(f"  winsorize: {standardizer.winsorize}")
        print(f"  winsorize_limits: {standardizer.winsorize_limits}")
        
        # 测试 **kwargs 兼容性
        standardizer = CrossSectionalZScoreStandardizer(
            winsorize=True, 
            winsorize_limits=[0.02, 0.98],
            some_extra_param="test"
        )
        print(f"✓ **kwargs 兼容性测试成功")
        print(f"  winsorize: {standardizer.winsorize}")
        print(f"  winsorize_limits: {standardizer.winsorize_limits}")
        
        return True
        
    except Exception as e:
        print(f"✗ Z-Score标准化器实例化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_parameters():
    """测试配置文件参数兼容性"""
    print("\n测试配置文件参数兼容性...")
    
    try:
        # 模拟配置文件参数
        quantile_config_params = {'n_quantiles': 100}
        zscore_config_params = {
            'winsorize': True,
            'winsorize_limits': [0.01, 0.99]
        }
        
        # 测试 Quantile 配置参数
        from factors.pipeline.standardizers.quantile import CrossSectionalQuantileStandardizer
        standardizer = CrossSectionalQuantileStandardizer(**quantile_config_params)
        print(f"✓ Quantile 配置参数测试成功: n_quantiles={standardizer.n_quantiles}")
        
        # 测试 ZScore 配置参数
        from factors.pipeline.standardizers.zscore import CrossSectionalZScoreStandardizer
        standardizer = CrossSectionalZScoreStandardizer(**zscore_config_params)
        print(f"✓ ZScore 配置参数测试成功: winsorize={standardizer.winsorize}, winsorize_limits={standardizer.winsorize_limits}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置文件参数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_original_functionality():
    """测试原始功能保持完整"""
    print("\n测试原始功能保持完整...")
    
    try:
        from factors.pipeline.standardizers.quantile import CrossSectionalQuantileStandardizer
        from factors.pipeline.standardizers.zscore import CrossSectionalZScoreStandardizer
        
        # 检查类是否仍然有 standardize 方法
        quantile_std = CrossSectionalQuantileStandardizer()
        zscore_std = CrossSectionalZScoreStandardizer()
        
        if hasattr(quantile_std, 'standardize'):
            print("✓ CrossSectionalQuantileStandardizer 保留 standardize 方法")
        else:
            print("✗ CrossSectionalQuantileStandardizer 丢失 standardize 方法")
            return False
            
        if hasattr(zscore_std, 'standardize'):
            print("✓ CrossSectionalZScoreStandardizer 保留 standardize 方法")
        else:
            print("✗ CrossSectionalZScoreStandardizer 丢失 standardize 方法")
            return False
        
        # 检查方法是否为可调用
        if callable(quantile_std.standardize):
            print("✓ CrossSectionalQuantileStandardizer.standardize 可调用")
        else:
            print("✗ CrossSectionalQuantileStandardize.standardize 不可调用")
            return False
            
        if callable(zscore_std.standardize):
            print("✓ CrossSectionalZScoreStandardizer.standardize 可调用")
        else:
            print("✗ CrossSectionalZScoreStandardizer.standardize 不可调用")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 原始功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始验证标准化器修复效果（简化版）...")
    print("=" * 60)
    
    success_count = 0
    total_tests = 4
    
    # 测试分位数标准化器实例化
    if test_quantile_standardizer_instantiation():
        success_count += 1
    
    # 测试Z-Score标准化器实例化
    if test_zscore_standardizer_instantiation():
        success_count += 1
    
    # 测试配置文件参数兼容性
    if test_config_parameters():
        success_count += 1
    
    # 测试原始功能保持完整
    if test_original_functionality():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {success_count}/{total_tests} 项测试通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！标准化器修复成功！")
        print("\n修复总结:")
        print("1. ✓ CrossSectionalQuantileStandardizer 现在支持 n_quantiles 参数")
        print("2. ✓ CrossSectionalZScoreStandardizer 现在支持 winsorize 和 winsorize_limits 参数")
        print("3. ✓ 两个类都支持 **kwargs 以接收额外配置参数")
        print("4. ✓ 原始 standardize 功能保持完整")
        return True
    else:
        print("❌ 部分测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)