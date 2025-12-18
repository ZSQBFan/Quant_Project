#!/usr/bin/env python3
"""
测试日志系统重复初始化修复
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, '/Volumes/叽叽叽/Code/quant_project_3.12_macmini')

from utils.logger import setup_logging
import logging

def test_logging_initialization():
    """测试日志系统重复初始化"""
    print("测试日志系统重复初始化...")
    
    # 清理之前的日志文件
    log_dir = 'output/logs'
    if os.path.exists(log_dir):
        old_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        print(f"清理前日志文件数量: {len(old_files)}")
    
    print("\n第1次调用 setup_logging:")
    setup_logging(log_dir='output/logs', log_prefix='test1')
    
    logging.info("第1次日志记录")
    
    print("\n第2次调用 setup_logging (应该跳过):")
    setup_logging(log_dir='output/logs', log_prefix='test2')
    
    logging.info("第2次日志记录")
    
    print("\n第3次调用 setup_logging (应该跳过):")
    setup_logging(log_dir='output/logs', log_prefix='test3')
    
    logging.info("第3次日志记录")
    
    # 检查生成的日志文件数量
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        log_files = [f for f in log_files if 'test' in f]
        print(f"\n测试后日志文件数量: {len(log_files)}")
        print(f"生成的日志文件: {log_files}")
        
        # 验证只生成了一个文件
        if len(log_files) == 1:
            print("✅ 成功！只生成了一个日志文件，重复初始化被阻止")
            
            # 检查日志内容
            log_file_path = os.path.join(log_dir, log_files[0])
            with open(log_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "第1次日志记录" in content and "第2次日志记录" in content and "第3次日志记录" in content:
                    print("✅ 日志记录正常，所有消息都被记录")
                    return True
                else:
                    print("⚠️  日志记录不完整")
                    return False
        else:
            print(f"❌ 失败！期望1个日志文件，实际生成了{len(log_files)}个")
            return False
    else:
        print("❌ 日志目录不存在")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("开始测试日志系统重复初始化修复")
    print("=" * 60)
    
    if test_logging_initialization():
        print("\n🎉 日志系统修复测试通过！")
        sys.exit(0)
    else:
        print("\n⚠️  日志系统修复测试失败")
        sys.exit(1)