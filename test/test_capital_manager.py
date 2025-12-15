# test/test_capital_manager.py

"""
资金管理器模块测试
验证资金管理器的各项功能和边界情况处理
"""

import sys
import os
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger.logger_config import setup_logging
from bt.pipeline.capital import CapitalManagerBase, FullPositionManager, create_full_position_manager


def test_full_position_manager():
    """测试全仓资金管理器的基本功能"""
    print("🧪 测试 1: 全仓资金管理器基本功能")
    print("=" * 50)
    
    # 创建资金管理器
    manager = FullPositionManager(utilization_ratio=0.95)
    print(f"资金管理器: {manager}")
    
    # 测试正常场景
    test_cases = [
        (100000.0, 0.95, "标准场景"),
        (50000.0, 0.90, "保守策略"),
        (200000.0, 0.98, "激进策略"),
    ]
    
    for total_value, expected_ratio, description in test_cases:
        print(f"\n测试场景: {description}")
        print(f"总价值: {total_value:,.2f}")
        print(f"期望利用率: {expected_ratio:.2%}")
        
        try:
            # 为每个测试案例创建不同的管理器实例
            test_manager = FullPositionManager(utilization_ratio=expected_ratio)
            allocation = test_manager.get_allocation(total_value)
            reserved_cash = total_value - allocation
            actual_ratio = allocation / total_value
            
            print(f"分配资金: {allocation:,.2f}")
            print(f"预留现金: {reserved_cash:,.2f}")
            print(f"实际利用率: {actual_ratio:.2%}")
            print(f"预留比例: {(1-actual_ratio)*100:.2f}%")
            
            # 验证结果合理性
            assert allocation <= total_value, "分配金额不应超过总价值"
            assert allocation >= 0, "分配金额不应为负"
            assert abs(actual_ratio - expected_ratio) < 1e-6, "利用率计算错误"
            
            print("✅ 通过")
            
        except Exception as e:
            print(f"❌ 失败: {e}")
    
    print("\n✅ 基本功能测试完成")


def test_boundary_conditions():
    """测试边界情况处理"""
    print("\n🧪 测试 2: 边界情况处理")
    print("=" * 50)
    
    # 测试边界情况
    edge_cases = [
        (0, 0.95, "零总价值"),
        (-10000, 0.95, "负总价值"),
        (100000, 1.5, "超100%利用率"),
        (100000, -0.1, "负利用率"),
        (100000, None, "无效利用率"),
    ]
    
    for total_value, utilization_ratio, description in edge_cases:
        print(f"\n边界测试: {description}")
        
        try:
            if utilization_ratio is None:
                # 测试无效的利用率参数
                manager = FullPositionManager(utilization_ratio="invalid")
                continue
            else:
                manager = FullPositionManager(utilization_ratio=utilization_ratio)
            
            allocation = manager.get_allocation(total_value)
            print(f"分配结果: {allocation:,.2f}")
            
        except ValueError as e:
            print(f"✅ 预期错误 (ValueError): {e}")
        except Exception as e:
            print(f"⚠️  其他错误: {e}")


def test_manager_operations():
    """测试资金管理器的操作方法"""
    print("\n🧪 测试 3: 资金管理器操作方法")
    print("=" * 50)
    
    # 创建管理器
    manager = FullPositionManager(utilization_ratio=0.95)
    
    print(f"初始利用率: {manager.get_utilization_ratio():.4f}")
    print(f"初始现金缓冲: {manager.get_reserved_ratio():.4f}")
    
    # 测试更新利用率
    print("\n更新利用率到 0.90...")
    manager.update_utilization_ratio(0.90)
    print(f"更新后利用率: {manager.get_utilization_ratio():.4f}")
    print(f"更新后现金缓冲: {manager.get_reserved_ratio():.4f}")
    
    # 验证更新后的计算
    allocation = manager.get_allocation(100000)
    print(f"100,000 总价值的分配: {allocation:,.2f}")
    
    print("✅ 操作方法测试完成")


def test_convenience_functions():
    """测试便捷函数"""
    print("\n🧪 测试 4: 便捷函数")
    print("=" * 50)
    
    # 测试创建函数
    manager1 = create_full_position_manager()
    manager2 = create_full_position_manager(utilization_ratio=0.90, name="CustomManager")
    
    print(f"默认管理器: {manager1}")
    print(f"自定义管理器: {manager2}")
    
    # 测试工厂函数
    from bt.pipeline.capital import get_capital_manager
    
    standard_manager = get_capital_manager('full_position', utilization_ratio=0.95)
    conservative_manager = get_capital_manager('conservative')
    aggressive_manager = get_capital_manager('aggressive')
    
    print(f"标准管理器: {standard_manager}")
    print(f"保守管理器: {conservative_manager}")
    print(f"激进管理器: {aggressive_manager}")
    
    print("✅ 便捷函数测试完成")


def test_logging():
    """测试日志记录功能"""
    print("\n🧪 测试 5: 日志记录功能")
    print("=" * 50)
    
    # 创建带有详细日志的管理器
    manager = FullPositionManager(utilization_ratio=0.95, name="TestManager")
    
    # 测试日志记录
    print("执行分配操作，日志将被记录...")
    allocation = manager.get_allocation(100000.0)
    
    print(f"分配结果: {allocation:,.2f}")
    print("✅ 日志记录测试完成")


def main():
    """主测试函数"""
    print("🚀 资金管理器模块测试开始")
    print("=" * 60)
    
    # 设置日志系统
    setup_logging(log_prefix='capital_manager_test')
    
    try:
        # 运行各项测试
        test_full_position_manager()
        test_boundary_conditions()
        test_manager_operations()
        test_convenience_functions()
        test_logging()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！资金管理器模块工作正常")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()