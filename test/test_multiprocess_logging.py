#!/usr/bin/env python3
"""
测试多进程环境下的日志系统修复
"""

import sys
import os
import multiprocessing
import time

# 添加项目路径
sys.path.insert(0, '/Volumes/叽叽叽/Code/quant_project_3.12_macmini')

from utils.logger import setup_logging
import logging

def worker_function(worker_id):
    """工作函数，模拟子进程中的日志使用"""
    print(f"Worker {worker_id} 开始执行")
    
    # 在子进程中尝试初始化日志
    setup_logging(log_dir='output/logs', log_prefix=f'worker_{worker_id}')
    
    # 记录一些日志
    logging.info(f"Worker {worker_id} 记录日志消息 1")
    time.sleep(0.1)
    logging.info(f"Worker {worker_id} 记录日志消息 2")
    
    print(f"Worker {worker_id} 执行完成")
    return worker_id

def test_multiprocess_logging():
    """测试多进程日志系统"""
    print("测试多进程环境下的日志系统...")
    
    # 清理之前的日志文件
    log_dir = 'output/logs'
    if os.path.exists(log_dir):
        old_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        print(f"清理前日志文件数量: {len(old_files)}")
    
    print("\n主进程初始化日志系统:")
    setup_logging(log_dir='output/logs', log_prefix='main')
    
    logging.info("主进程记录的第一条日志")
    
    # 创建多个子进程
    print("\n启动多个子进程...")
    with multiprocessing.Pool(processes=3) as pool:
        results = pool.map(worker_function, range(3))
    
    print(f"\n子进程执行结果: {results}")
    
    # 检查生成的日志文件数量
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        print(f"\n测试后日志文件数量: {len(log_files)}")
        print(f"生成的日志文件: {sorted(log_files)}")
        
        # 验证只有主进程日志文件
        main_logs = [f for f in log_files if 'main_' in f]
        worker_logs = [f for f in log_files if 'worker_' in f]
        
        print(f"主进程日志文件: {len(main_logs)}")
        print(f"子进程日志文件: {len(worker_logs)}")
        
        if len(main_logs) == 1 and len(worker_logs) == 0:
            print("✅ 成功！只有主进程日志文件，子进程没有创建额外文件")
            return True
        else:
            print("❌ 失败！子进程创建了额外的日志文件")
            return False
    else:
        print("❌ 日志目录不存在")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("开始测试多进程日志系统修复")
    print("=" * 60)
    
    if test_multiprocess_logging():
        print("\n🎉 多进程日志系统修复测试通过！")
        sys.exit(0)
    else:
        print("\n⚠️  多进程日志系统修复测试失败")
        sys.exit(1)