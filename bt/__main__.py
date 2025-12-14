"""
Backtrader 回测框架验证脚本
运行: python -m bt
"""

import sys

def verify_structure():
    """验证目录结构是否正确创建"""
    try:
        # 尝试导入所有新创建的模块
        import config
        import bt
        import bt.core
        import bt.triggers
        import bt.pipeline
        import bt.utils
        
        print("✅ Backtrader 回测框架目录结构验证成功！")
        print("=" * 60)
        print(f"bt 框架版本: {bt.__version__ if hasattr(bt, '__version__') else '0.1.0'}")
        print()
        
        print("已就绪的模块:")
        print("  ✓ config         - 配置管理")
        print("  ✓ bt.core        - 核心组件")
        print("  ✓ bt.triggers    - 触发器系统")
        print("  ✓ bt.pipeline    - 数据流水线")
        print("  ✓ bt.utils       - 工具函数")
        print()
        
        print("下一步开发任务:")
        print("  1. 实现 bt/pipeline/pandas_feed.py")
        print("  2. 创建 bt/core/base_strategy.py")
        print("  3. 实现 bt/triggers/risk_manager.py")
        print("  4. 查看 BACKTRADER_SETUP.md 获取详细说明")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

if __name__ == "__main__":
    success = verify_structure()
    sys.exit(0 if success else 1)
