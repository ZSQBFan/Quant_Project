#!/usr/bin/env python3
"""验证标准化器修复效果的测试脚本"""
import pandas as pd
import sys
import os

# 添加项目路径到 Python 路径
sys.path.insert(0, '/Volumes/叽叽叽/Code/quant_project_3.12_macmini')

def test_quantile_standardizer():
    """测试分位数标准化器"""
    print("测试 CrossSectionalQuantileStandardizer...")
    
    try:
        from factors.pipeline.standardizers.quantile import CrossSectionalQuantileStandardizer
        
        # 测试默认参数实例化
        standardizer = CrossSectionalQuantileStandardizer()
        print(f"✓ 默认参数实例化成功: n_quantiles={standardizer.n_quantiles}")
        
        # 测试自定义参数实例化
        standardizer = CrossSectionalQuantileStandardizer(n_quantiles=50)
        print(f"✓ 自定义参数实例化成功: n_quantiles={standardizer.n_quantiles}")
        
        # 测试标准化功能
        test_data = pd.DataFrame({
            'stock1': [1, 2, 3, 4, 5],
            'stock2': [2, 4, 6, 8, 10],
            'stock3': [0.5, 1.5, 2.5, 3.5, 4.5]
        })
        
        result = standardizer.standardize(test_data)
        print(f"✓ 标准化功能正常，结果形状: {result.shape}")
        print(f"  前3行结果: \n{result.head(3)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 分位数标准化器测试失败: {e}")
        return False

def test_zscore_standardizer():
    """测试Z-Score标准化器"""
    print("\n测试 CrossSectionalZScoreStandardizer...")
    
    try:
        from factors.pipeline.standardizers.zscore import CrossSectionalZScoreStandardizer
        
        # 测试默认参数实例化
        standardizer = CrossSectionalZScoreStandardizer()
        print(f"✓ 默认参数实例化成功: winsorize={standardizer.winsorize}, winsorize_limits={standardizer.winsorize_limits}")
        
        # 测试自定义参数实例化
        standardizer = CrossSectionalZScoreStandardizer(winsorize=False, winsorize_limits=[0.05, 0.95])
        print(f"✓ 自定义参数实例化成功: winsorize={standardizer.winsorize}, winsorize_limits={standardizer.winsorize_limits}")
        
        # 测试标准化功能
        test_data = pd.DataFrame({
            'stock1': [1, 2, 3, 4, 5],
            'stock2': [2, 4, 6, 8, 10],
            'stock3': [0.5, 1.5, 2.5, 3.5, 4.5]
        })
        
        result = standardizer.standardize(test_data)
        print(f"✓ 标准化功能正常，结果形状: {result.shape}")
        print(f"  前3行结果: \n{result.head(3)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Z-Score标准化器测试失败: {e}")
        return False

def test_with_config():
    """测试使用配置文件参数"""
    print("\n测试使用配置文件参数...")
    
    try:
        import yaml
        
        # 读取配置文件
        config_path = '/Volumes/叽叽叽/Code/quant_project_3.12_macmini/configs/factors/pipeline/standardizers.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 测试 Quantile 配置
        if 'Quantile' in config:
            quantile_config = config['Quantile']
            print(f"✓ 读取 Quantile 配置: {quantile_config}")
            
            from factors.pipeline.standardizers.quantile import CrossSectionalQuantileStandardizer
            standardizer = CrossSectionalQuantileStandardizer(**quantile_config.get('params', {}))
            print(f"✓ 使用配置参数实例化成功: n_quantiles={standardizer.n_quantiles}")
        
        # 测试 ZScore 配置
        if 'ZScore' in config:
            zscore_config = config['ZScore']
            print(f"✓ 读取 ZScore 配置: {zscore_config}")
            
            from factors.pipeline.standardizers.zscore import CrossSectionalZScoreStandardizer
            standardizer = CrossSectionalZScoreStandardizer(**zscore_config.get('params', {}))
            print(f"✓ 使用配置参数实例化成功: winsorize={standardizer.winsorize}, winsorize_limits={standardizer.winsorize_limits}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置文件测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始验证标准化器修复效果...")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # 测试分位数标准化器
    if test_quantile_standardizer():
        success_count += 1
    
    # 测试Z-Score标准化器
    if test_zscore_standardizer():
        success_count += 1
    
    # 测试配置文件集成
    if test_with_config():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"测试结果: {success_count}/{total_tests} 项测试通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！标准化器修复成功！")
        return True
    else:
        print("❌ 部分测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)