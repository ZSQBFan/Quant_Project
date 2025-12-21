"""
主程序 - 量化交易系统统一入口

使用方法：
    python main.py --mode factor_analysis  # 因子分析
    python main.py --mode backtest        # 回测
    python main.py --mode download_data   # 数据下载
"""

import argparse
import logging
from pathlib import Path

# 初始化框架
from core import auto_load_all, load_config, print_registry_stats
from utils.logger import setup_logging

# 设置日志
setup_logging(log_dir='output/logs', log_prefix='main')
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='量化交易系统')
    parser.add_argument(
        '--mode',
        choices=['factor_analysis', 'backtest', 'download_data', 'list_components'],
        default='factor_analysis',
        help='运行模式'
    )
    parser.add_argument('--config-dir', type=str, default='configs', help='配置文件目录')

    args = parser.parse_args()

    # 1. 自动加载所有组件
    logger.info("="*60)
    logger.info("🚀 正在启动量化交易系统...")
    logger.info("="*60)

    logger.info("📦 正在加载组件...")
    auto_load_all()

    # 2. 显示注册表统计
    logger.info("📊 组件加载完成，显示注册表统计：")
    print_registry_stats()

    # 3. 加载配置
    logger.info(f"📁 正在加载配置文件（目录: {args.config_dir}）...")
    config_loader = load_config(args.config_dir)

    # 3.1 动态调整日志等级
    logging_config = config_loader.load_logging()
    if logging_config:
        new_level = logging_config.get('level', 'INFO')
        logger.info(f"⚙️ 正在根据配置文件调整日志等级为: {new_level}")
        setup_logging(
            log_dir=logging_config.get('log_dir', 'output/logs'),
            log_prefix=logging_config.get('log_prefix', 'main'),
            log_level=new_level,
            force=True
        )

    # 4. 根据模式执行
    if args.mode == 'list_components':
        logger.info("📋 列出所有已注册组件")
        print_registry_stats()

    elif args.mode == 'download_data':
        logger.info("📥 数据下载模式")
        from scripts.download_data import download_data
        download_data()

    elif args.mode == 'factor_analysis':
        logger.info("📊 因子分析模式")
        from scripts.run_factor_analysis import run_factor_analysis
        run_factor_analysis(config_loader)

    elif args.mode == 'backtest':
        logger.info("🔄 回测模式")
        from scripts.run_backtest import run_backtest
        run_backtest(config_loader)

    logger.info("="*60)
    logger.info("✅ 程序执行完成")
    logger.info("="*60)


if __name__ == '__main__':
    main()
