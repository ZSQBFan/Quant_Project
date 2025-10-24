# logger_config.py
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


def setup_logging(log_dir='logs', log_prefix='run'):
    """
    配置全局日志系统，生成带时间戳的唯一日志文件。

    Args:
        log_dir (str, optional): 存放日志文件的目录。默认为 'logs'。
        log_prefix (str, optional): 日志文件名的前缀。默认为 'run'。
    """
    # 1. 创建日志目录（如果不存在）
    os.makedirs(log_dir, exist_ok=True)

    # 2. 生成基于当前时间的文件名
    # 文件名格式: 前缀_YYYYMMDD_HHMMSS.log, 例如: backtest_20251016_213000.log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{log_prefix}_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_filename)

    # 3. 获取根日志记录器并配置
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # 4. 定义日志格式
    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - (%(thread)d) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # 5. 创建文件处理器 (FileHandler)
    # 使用 'w' 模式，确保每次运行都创建一个新的日志文件，而不是追加
    file_handler = logging.FileHandler(log_file_path,
                                       mode='w',
                                       encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 6. 创建 Tqdm 处理器，用于在控制台输出
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setLevel(logging.INFO)
    tqdm_handler.setFormatter(formatter)
    logger.addHandler(tqdm_handler)

    # 在控制台打印日志文件的实际位置，方便查找
    print(f"日志系统已启动，本次运行的日志将被记录到: {log_file_path}")
