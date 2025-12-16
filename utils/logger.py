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


def setup_logging(log_dir='output/logs', log_prefix='run', log_level=None):
    """
    配置全局日志系统，生成带时间戳的唯一日志文件。

    Args:
        log_dir (str, optional): 存放日志文件的目录。默认为 'output/logs'。
        log_prefix (str, optional): 日志文件名的前缀。默认为 'run'。
        log_level (int, optional): 日志级别。默认为 logging.INFO。
    """

    # 日志级别配置
    if log_level is None:
        log_level = logging.INFO

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
        '%(asctime)s - [%(levelname)s] - (%(thread)d) - %(message)s',
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
