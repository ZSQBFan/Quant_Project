# test/test_debug_logging.py

"""
测试调试级别日志记录
验证资金管理器的DEBUG级别日志功能
"""

import sys
import os
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger.logger_config import setup_logging
from bt.pipeline.capital import FullPositionManager

def test_debug_logging():
    """测试DEBUG级别日志记录"""
    print("🧪 测试 DEBUG 级别日志记录")
    print("=" * 50)
    
    # 设置DEBUG级别的日志
    setup_logging(log_prefix='debug_capital_test')
    
    # 创建资金管理器
    manager = FullPositionManager(utilization_ratio=0.95, name="DebugTestManager")
    
    print("执行分配操作，查看DEBUG级别日志...")
    
    # 执行分配操作
    allocation = manager.get_allocation(100000.0)
    
    print(f"分配结果: {allocation:,.2f}")
    print("请查看日志文件中的DEBUG级别记录")

if __name__ == "__main__":
    test_debug_logging()