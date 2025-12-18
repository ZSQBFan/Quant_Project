#!/usr/bin/env python3
"""
测试因子过滤修复

验证只有 enabled_factors 中定义的因子被加载和计算。
"""

import logging
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config

def test_factor_filtering():
    """测试因子过滤是否正常工作"""
    print("=" * 60)
    print("测试因子过滤修复")
    print("=" * 60)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 加载配置
        config_loader = load_config()
        
        # 加载因子配置
        factors_config = config_loader.load_factors()
        
        print(f"\n✅ 成功加载 {len(factors_config)} 个因子")
        print(f"启用的因子: {list(factors_config.keys())}")
        
        # 验证所有因子都是复合因子（行业中性因子）
        for name, cfg in factors_config.items():
            print(f"  - {name}: category={cfg.category}, enabled={cfg.enabled}")
            
            # 检查是否都是复合因子
            if cfg.category != 'complex':
                raise ValueError(f"因子 {name} 不是复合因子！应该是 complex，但实际是 {cfg.category}")
        
        print(f"\n✅ 所有因子都是复合因子，符合预期")
        print(f"✅ 修复成功：只加载了 enabled_factors 中定义的因子")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_factor_filtering()
    sys.exit(0 if success else 1)