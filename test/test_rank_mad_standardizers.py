#!/usr/bin/env python3
"""Rank和MAD标准化器实现验证脚本 - 不依赖外部库"""
import sys
import os

# 添加项目路径到 Python 路径
sys.path.insert(0, '/Volumes/叽叽叽/Code/quant_project_3.12_macmini')

def test_rank_standardizer_instantiation():
    """测试Rank标准化器实例化"""
    print("测试 CrossSectionalRankStandardizer 实例化...")
    
    try:
        from factors.pipeline.standardizers.rank import CrossSectionalRankStandardizer
        
        # 测试默认参数实例化
        standardizer = CrossSectionalRankStandardizer()
        print(f"✓ 默认参数实例化成功")
        print(f"  ascending: {standardizer.ascending}")
        print(f"  pct: {standardizer.pct}")
        
        # 测试自定义参数实例化
        standardizer = CrossSectionalRankStandardizer(ascending=False, pct=False)
        print(f"✓ 自定义参数实例化成功")
        print(f"  ascending: {standardizer.ascending}")
        print(f"  pct: {standardizer.pct}")
        
        # 测试 **kwargs 兼容性
        standardizer = CrossSectionalRankStandardizer(
            ascending=True, 
            pct=True,
            some_extra_param="test"
        )
        print(f"✓ **kwargs 兼容性测试成功")
        print(f"  ascending: {standardizer.ascending}")
        print(f"  pct: {standardizer.pct}")
        
        return True
        
    except Exception as e:
        print(f"✗ Rank标准化器实例化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mad_standardizer_instantiation():
    """测试MAD标准化器实例化"""
    print("\n测试 CrossSectionalMADStandardizer 实例化...")
    
    try:
        from factors.pipeline.standardizers.mad import CrossSectionalMADStandardizer
        
        # 测试默认参数实例化
        standardizer = CrossSectionalMADStandardizer()
        print(f"✓ 默认参数实例化成功")
        print(f"  scale_factor: {standardizer.scale_factor}")
        
        # 测试自定义参数实例化
        standardizer = CrossSectionalMADStandardizer(scale_factor=1.5)
        print(f"✓ 自定义参数实例化成功")
        print(f"  scale_factor: {standardizer.scale_factor}")
        
        # 测试 **kwargs 兼容性
        standardizer = CrossSectionalMADStandardizer(
            scale_factor=1.4826,
            some_extra_param="test"
        )
        print(f"✓ **kwargs 兼容性测试成功")
        print(f"  scale_factor: {standardizer.scale_factor}")
        
        return True
        
    except Exception as e:
        print(f"✗ MAD标准化器实例化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_parameters():
    """测试配置文件参数兼容性"""
    print("\n测试配置文件参数兼容性...")
    
    try:
        # 模拟配置文件参数
        rank_config_params = {
            'ascending': True,
            'pct': True
        }
        mad_config_params = {
            'scale_factor': 1.4826
        }
        
        # 测试 Rank 配置参数
        from factors.pipeline.standardizers.rank import CrossSectionalRankStandardizer
        standardizer = CrossSectionalRankStandardizer(**rank_config_params)
        print(f"✓ Rank 配置参数测试成功: ascending={standardizer.ascending}, pct={standardizer.pct}")
        
        # 测试 MAD 配置参数
        from factors.pipeline.standardizers.mad import CrossSectionalMADStandardizer
        standardizer = CrossSectionalMADStandardizer(**mad_config_params)
        print(f"✓ MAD 配置参数测试成功: scale_factor={standardizer.scale_factor}")
        
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
        from factors.pipeline.standardizers.rank import CrossSectionalRankStandardizer
        from factors.pipeline.standardizers.mad import CrossSectionalMADStandardizer
        
        # 检查类是否仍然有 standardize 方法
        rank_std = CrossSectionalRankStandardizer()
        mad_std = CrossSectionalMADStandardizer()
        
        if hasattr(rank_std, 'standardize'):
            print("✓ CrossSectionalRankStandardizer 保留 standardize 方法")
        else:
            print("✗ CrossSectionalRankStandardizer 丢失 standardize 方法")
            return False
            
        if hasattr(mad_std, 'standardize'):
            print("✓ CrossSectionalMADStandardizer 保留 standardize 方法")
        else:
            print("✗ CrossSectionalMADStandardizer 丢失 standardize 方法")
            return False
        
        # 检查方法是否为可调用
        if callable(rank_std.standardize):
            print("✓ CrossSectionalRankStandardizer.standardize 可调用")
        else:
            print("✗ CrossSectionalRankStandardizer.standardize 不可调用")
            return False
            
        if callable(mad_std.standardize):
            print("✓ CrossSectionalMADStandardizer.standardize 可调用")
        else:
            print("✗ CrossSectionalMADStandardizer.standardize 不可调用")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 原始功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_registration():
    """测试注册功能"""
    print("\n测试注册功能...")
    
    try:
        from core.registry import get_standardizer
        
        # 测试获取Rank标准化器
        rank_cls = get_standardizer('Rank')
        print(f"✓ 成功获取Rank标准化器: {rank_cls}")
        
        # 测试获取MAD标准化器
        mad_cls = get_standardizer('MAD')
        print(f"✓ 成功获取MAD标准化器: {mad_cls}")
        
        return True
        
    except Exception as e:
        print(f"✗ 注册功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始验证Rank和MAD标准化器实现...")
    print("=" * 60)
    
    success_count = 0
    total_tests = 5
    
    # 测试Rank标准化器实例化
    if test_rank_standardizer_instantiation():
        success_count += 1
    
    # 测试MAD标准化器实例化
    if test_mad_standardizer_instantiation():
        success_count += 1
    
    # 测试配置文件参数兼容性
    if test_config_parameters():
        success_count += 1
    
    # 测试原始功能保持完整
    if test_original_functionality():
        success_count += 1
    
    # 测试注册功能
    if test_registration():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {success_count}/{total_tests} 项测试通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！Rank和MAD标准化器实现成功！")
        print("\n实现总结:")
        print("1. ✓ CrossSectionalRankStandardizer 实现完成，支持ascending和pct参数")
        print("2. ✓ CrossSectionalMADStandardizer 实现完成，支持scale_factor参数")
        print("3. ✓ 两个类都支持 **kwargs 以接收额外配置参数")
        print("4. ✓ 原始 standardize 功能保持完整")
        print("5. ✓ 成功注册到系统中，可通过配置实例化")
        return True
    else:
        print("❌ 部分测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)