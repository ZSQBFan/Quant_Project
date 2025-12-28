"""
测试 Backtrader 数据导出和集成功能

这个脚本验证：
1. BTDataExporter 能否正确导出数据
2. FactorPandasData 能否正确加载数据到 Backtrader
3. 数据流是否完整（从 DataManager 到 Backtrader）
"""

import sys
import os
import pandas as pd
import numpy as np
import backtrader as bt
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from logger.logger_config import setup_logging
from data.data_manager import DataProviderManager
from data.data_providers import SQLiteDataProvider
from bt.data import BTDataExporter, FactorPandasData


# 设置日志
setup_logging(log_prefix='bt_data_test')

# 简单的测试策略，只打印数据而不实际交易
class TestStrategy(bt.Strategy):
    """
    简单测试策略，验证数据加载和因子信号
    """
    
    def __init__(self):
        # 记录数据开始
        self.dataclose = self.datas[0].close
        self.data_volume = self.datas[0].volume
        self.data_signal = getattr(self.datas[0], 'combined_signal', None)
        self.data_suspended = getattr(self.datas[0], 'suspended', None)
        
        self.day_count = 0
        self.signal_count = 0
        self.suspended_count = 0
        
    def next(self):
        self.day_count += 1
        
        # 获取当前日期
        current_date = self.datas[0].datetime.date(0)
        
        # 检查是否为停牌日
        is_suspended = self.data_suspended is not None and self.data_suspended[0]
        
        # 每5天打印一次数据状态，或者遇到停牌日时打印
        if self.day_count % 5 == 0 or is_suspended:
            if is_suspended:
                info_msg = f"🔴 停牌日 - 日期: {current_date}, 收盘价: NaN, 成交量: NaN"
                self.suspended_count += 1
            else:
                info_msg = f"🟢 交易日 - 日期: {current_date}, 收盘价: {self.dataclose[0]:.2f}, 成交量: {self.data_volume[0]}"
            
            # 检查因子信号
            if self.data_signal is not None and not np.isnan(self.data_signal[0]):
                info_msg += f", 信号: {self.data_signal[0]:.4f}"
                self.signal_count += 1
            else:
                info_msg += ", 信号: NaN"
            
            # 检查停牌状态
            if self.data_suspended is not None:
                info_msg += f", 停牌标记: {self.data_suspended[0]}"
            
            logging.info(f"[策略] {info_msg}")
    
    def stop(self):
        trading_days = self.day_count - self.suspended_count
        logging.info(f"[策略] 回测结束，共处理 {self.day_count} 天，其中 {trading_days} 个交易日，{self.suspended_count} 个停牌日")
        logging.info(f"[策略] 其中 {self.signal_count} 天有有效信号")


def create_test_factor_series(universe, start_date, end_date):
    """
    创建测试用的因子序列
    
    返回:
        pd.Series: 索引为 (date, asset) 的 MultiIndex，值为随机因子值
    """
    logging.info("创建测试因子序列...")
    
    # 创建日期范围
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    # 只保留工作日（简化处理）
    date_range = date_range[date_range.weekday < 5]
    
    # 创建 MultiIndex
    index = pd.MultiIndex.from_product(
        [date_range, universe],
        names=['date', 'asset']
    )
    
    # 创建随机因子值（部分设为 NaN 模拟缺失）
    np.random.seed(42)  # 固定随机种子以便复现
    factor_values = np.random.normal(0, 1, size=len(index))
    
    # 随机将 20% 的值设为 NaN
    nan_mask = np.random.random(len(index)) < 0.2
    factor_values[nan_mask] = np.nan
    
    factor_series = pd.Series(factor_values, index=index, name='combined_signal')
    
    logging.info(f"创建因子序列完成，形状: {factor_series.shape}")
    return factor_series


def add_suspended_days_to_test_data(exported_files, universe, start_date, end_date):
    """
    为测试数据添加一些停牌日
    
    参数:
        exported_files: 已导出的文件路径列表
        universe: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
    """
    logging.info("开始为测试数据添加停牌日...")
    
    # 为每只股票添加一些随机的停牌日
    np.random.seed(123)  # 固定随机种子以便复现
    
    for file_path in exported_files:
        try:
            # 读取导出的数据
            df = pd.read_parquet(file_path)
            
            # 从文件名获取股票代码
            asset = os.path.basename(file_path).replace('.parquet', '')
            
            if asset not in universe:
                continue
                
            # 随机选择2-5个停牌日
            num_suspended_days = np.random.randint(2, 6)
            
            # 获取所有交易日
            all_dates = df.index.tolist()
            
            # 随机选择停牌日
            suspended_indices = np.random.choice(
                len(all_dates),
                size=min(num_suspended_days, len(all_dates)),
                replace=False
            )
            suspended_dates = [all_dates[i] for i in suspended_indices]
            
            # 为停牌日设置 NaN 值
            for date in suspended_dates:
                # 将 OHLCV 设为 NaN，模拟停牌
                df.loc[date, ['open', 'high', 'low', 'close', 'volume']] = np.nan
                # 更新停牌标记
                df.loc[date, 'suspended'] = True
                
            # 重新保存文件
            df.to_parquet(file_path)
            
            logging.info(f"  > 为 {asset} 添加了 {len(suspended_dates)} 个停牌日: {suspended_dates}")
            
        except Exception as e:
            logging.error(f"  > 为 {file_path} 添加停牌日失败: {e}")


def main():
    """
    主测试函数
    """
    logging.info("=== 开始 Backtrader 数据导出和集成测试 ===")
    
    # 1. 初始化 DataManager
    logging.info("1. 初始化 DataManager...")
    
    # 设置详细日志以便调试
    logging.getLogger().setLevel(logging.DEBUG)
    
    # 使用 SQLite 数据提供者（假设已有数据库）
    provider_configs = [
        (SQLiteDataProvider, {'db_path': 'test/test.db'})
    ]
    
    # 测试股票池（选择几只常见的股票）
    test_universe = ['000001', '000002', '600519', '600036']
    
    # 测试日期范围
    start_date = '2023-01-01'
    end_date = '2023-06-30'
    
    try:
        dm = DataProviderManager(
            provider_configs=provider_configs,
            symbols=test_universe,
            start_date=start_date,
            end_date=end_date,
            db_path='./database/quant_data.db',
            auto_detect_universe=False
        )
        logging.info("✅ DataManager 初始化成功")
    except Exception as e:
        logging.error(f"❌ DataManager 初始化失败: {e}")
        return
    
    # 2. 创建测试因子序列
    logging.info("2. 创建测试因子序列...")
    factor_series = create_test_factor_series(test_universe, start_date, end_date)
    
    # 3. 初始化数据导出器
    logging.info("3. 初始化数据导出器...")
    exporter = BTDataExporter(
        data_manager=dm,
        output_dir='temp/test_bt_data/'
    )
    
    # 清空导出目录
    exporter.clear_export_dir()
    
    # 4. 导出数据
    logging.info("4. 导出数据...")
    try:
        exported_files = exporter.export(
            universe=test_universe,
            start_date=start_date,
            end_date=end_date,
            factor_series=factor_series
        )
        logging.info(f"✅ 数据导出成功，共导出 {len(exported_files)} 个文件")
        
        # 打印导出的文件列表
        for file_path in exported_files:
            logging.info(f"  > {file_path}")
            
    except Exception as e:
        logging.error(f"❌ 数据导出失败: {e}")
        return
    
    # 4.5. 为测试数据添加停牌日
    logging.info("4.5. 为测试数据添加停牌日...")
    try:
        add_suspended_days_to_test_data(exported_files, test_universe, start_date, end_date)
        logging.info("✅ 成功添加停牌日到测试数据")
    except Exception as e:
        logging.error(f"❌ 添加停牌日失败: {e}")
        import traceback
        logging.error(f"详细错误: {traceback.format_exc()}")
    
    # 5. 验证导出的数据
    logging.info("5. 验证导出的数据...")
    try:
        # 选择第一个文件进行验证
        test_file = exported_files[0]
        df = pd.read_parquet(test_file)
        
        logging.info(f"验证文件: {test_file}")
        logging.info(f"数据形状: {df.shape}")
        logging.info(f"列名: {df.columns.tolist()}")
        logging.info(f"索引类型: {type(df.index)}")
        logging.info(f"前5行数据:\n{df.head()}")
        
        # 检查必要列是否存在
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'openinterest', 'combined_signal', 'suspended']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logging.error(f"❌ 缺少必要列: {missing_columns}")
        else:
            logging.info("✅ 所有必要列都存在")
            
        # 检查停牌日数据
        suspended_days = df[df['suspended'] == True]
        if not suspended_days.empty:
            logging.info(f"✅ 发现 {len(suspended_days)} 个停牌日")
            logging.info(f"停牌日示例:\n{suspended_days.head()}")
            
            # 验证停牌日的价格数据是否为 NaN
            price_cols = ['open', 'high', 'low', 'close', 'volume']
            for date, row in suspended_days.iterrows():
                nan_prices = [col for col in price_cols if pd.isna(row[col])]
                if len(nan_prices) == len(price_cols):
                    logging.debug(f"  > {date}: 所有价格数据正确为 NaN")
                else:
                    logging.warning(f"  > ⚠️ {date}: 部分价格数据不为 NaN: {nan_prices}")
        else:
            logging.warning("⚠️ 未发现停牌日数据")
            
    except Exception as e:
        logging.error(f"❌ 数据验证失败: {e}")
        return
    
    # 6. 测试 Backtrader 数据加载
    logging.info("6. 测试 Backtrader 数据加载...")
    try:
        # 创建 Cerebro 引擎
        cerebro = bt.Cerebro()
        
        # 加载第一个导出的文件
        data = FactorPandasData(dataname=df)
        cerebro.adddata(data)
        
        # 添加测试策略
        cerebro.addstrategy(TestStrategy)
        
        # 设置初始资金
        cerebro.broker.setcash(100000.0)
        
        # 设置手续费
        cerebro.broker.setcommission(commission=0.001)
        
        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # 运行回测
        logging.info("开始运行 Backtrader 回测...")
        start_value = cerebro.broker.getvalue()
        results = cerebro.run()
        end_value = cerebro.broker.getvalue()
        
        # 打印结果
        logging.info(f"初始资金: {start_value:,.2f}")
        logging.info(f"最终资金: {end_value:,.2f}")
        logging.info(f"总收益率: {(end_value/start_value - 1):.2%}")
        
        # 获取分析结果
        strat = results[0]
        returns_analysis = strat.analyzers.returns.get_analysis()
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        
        annual_return = returns_analysis.get('rnorm100', 0)
        sharpe_ratio = sharpe_analysis.get('sharperatio', 0)
        
        # 处理 None 值
        annual_return = annual_return if annual_return is not None else 0
        sharpe_ratio = sharpe_ratio if sharpe_ratio is not None else 0
        
        logging.info(f"年化收益率: {annual_return:.2f}%")
        logging.info(f"夏普比率: {sharpe_ratio:.2f}")
        
        logging.info("✅ Backtrader 数据加载和回测测试成功")
        
    except Exception as e:
        logging.error(f"❌ Backtrader 测试失败: {e}")
        import traceback
        logging.error(f"详细错误: {traceback.format_exc()}")
        return
    
    logging.info("=== 测试完成 ===")


if __name__ == '__main__':
    main()