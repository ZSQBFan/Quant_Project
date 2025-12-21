"""
日志配置模块

提供统一的日志系统配置，支持 tqdm 进度条兼容。
"""

import logging
import sys
import os
from datetime import datetime
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """
    自定义日志处理器，将日志记录重定向到 tqdm.write()。
    这确保了日志输出不会与 tqdm 进度条的显示发生冲突。
    """

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stdout)
            self.flush()
        except Exception:
            self.handleError(record)


# 全局标志：防止重复初始化日志系统
_logging_initialized = False
_initialized_log_path = None

def _is_main_process():
    """
    检测是否在主进程中运行。

    使用 multiprocessing.current_process().name 来判断，
    这是最可靠的方式来检测 ProcessPoolExecutor/multiprocessing 创建的子进程。
    """
    try:
        import multiprocessing
        return multiprocessing.current_process().name == 'MainProcess'
    except Exception:
        return True  # 如果无法检测，假设是主进程


def setup_logging(log_dir='output/logs', log_prefix='run', log_level=None, force=False):
    """
    配置全局日志系统，生成带时间戳的唯一日志文件。
    在多进程环境中，子进程只会使用控制台日志，不创建文件。

    Args:
        log_dir (str, optional): 存放日志文件的目录。默认为 'output/logs'。
        log_prefix (str, optional): 日志文件名的前缀。默认为 'run'。
        log_level (int, optional): 日志级别。默认为 logging.INFO。
        force (bool, optional): 是否强制重新初始化。默认为 False。
    """
    global _logging_initialized, _initialized_log_path

    # 如果已经初始化过，且不强制重新初始化
    if _logging_initialized and not force:
        return

    # 日志级别配置
    if log_level is None:
        log_level = logging.INFO

    # 检测是否在主进程中
    is_main = _is_main_process()

    # 在子进程中，禁用日志或使用最小化配置
    if not is_main:
        # 子进程：设置 WARNING 级别，减少输出
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)
        # 清除所有 handler，避免重复配置
        logger.handlers.clear()
        _logging_initialized = True
        return

    # 1. 创建日志目录（如果不存在）
    os.makedirs(log_dir, exist_ok=True)

    # 2. 生成基于当前时间的文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{log_prefix}_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_filename)

    # 3. 获取根日志记录器并配置
    logger = logging.getLogger()
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # 4. 定义日志格式
    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - PID:%(process)d-TID:%(thread)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # 5. 创建文件处理器 (FileHandler)
    file_handler = logging.FileHandler(log_file_path,
                                       mode='w',
                                       encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 6. 创建 Tqdm 处理器，用于在控制台输出
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setLevel(log_level)
    tqdm_handler.setFormatter(formatter)
    logger.addHandler(tqdm_handler)

    # 在控制台打印日志文件的实际位置
    print(f"日志系统已启动，日志文件: {log_file_path}")
    
    # 标记为已初始化
    _logging_initialized = True
    _initialized_log_path = log_file_path
