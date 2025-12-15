# test/test_backtrader_with_real_db.py

"""
使用真实数据库数据的Backtrader集成测试

此测试使用quant_data.db中的真实数据验证：
1. 数据库连接和数据获取是否正常
2. 数据桥接器能否处理真实数据
3. 策略框架能否在真实数据上正常运行

前置条件：
- quant_data.db 数据库文件存在且包含股票数据
- 数据库中有stock_daily_prices表和相关数据
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtrader as bt
from logger.logger_config import setup_logging
from bt.core.base_strategy import BacktestStrategy
from bt.core.constants import ActionPriority, ActionType
from bt.data.feeds import FactorPandasData
from bt.data.exporter import BTDataExporter
from data.data_manager import DataProviderManager
from data.database_handler import DatabaseHandler


class DatabaseBacktraderTest:
    """
    使用真实数据库数据的Backtrader测试类
    """
    
    def __init__(self, db_path='quant_data.db'):
        self.db_path = db_path
        self.temp_dir = None
        
        # 检查数据库是否存在
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
        
        print(f"✅ 找到数据库文件: {db_path}")
        
        # 测试数据库连接
        try:
            self.db_handler = DatabaseHandler(db_path)
            print("✅ 数据库连接成功")
        except Exception as e:
            raise Exception(f"数据库连接失败: {e}")
    
    def check_database_content(self):
        """检查数据库内容"""
        print("\n" + "=" * 60)
        print("检查数据库内容")
        print("=" * 60)
        
        try:
            # 检查stock_daily_prices表
            query = "SELECT COUNT(*) as count FROM stock_daily_prices"
            result = self.db_handler.query_data(query)
            total_records = result.iloc[0]['count'] if not result.empty else 0
            print(f"📊 stock_daily_prices表总记录数: {total_records}")
            
            # 检查股票数量
            query = "SELECT COUNT(DISTINCT code) as count FROM stock_daily_prices"
            result = self.db_handler.query_data(query)
            stock_count = result.iloc[0]['count'] if not result.empty else 0
            print(f"📈 数据库中的股票数量: {stock_count}")
            
            # 获取股票列表（前10只）
            query = "SELECT DISTINCT code FROM stock_daily_prices LIMIT 10"
            result = self.db_handler.query_data(query)
            if not result.empty:
                stock_list = result['code'].tolist()
                print(f"📋 股票列表（前10只）: {stock_list}")
                return stock_list
            else:
                print("❌ 数据库中没有找到股票数据")
                return []
                
        except Exception as e:
            print(f"❌ 检查数据库内容失败: {e}")
            return []
    
    def get_sample_stocks(self, limit=5):
        """获取样本股票列表"""
        try:
            # 获取有足够数据的股票
            query = """
            SELECT code, COUNT(*) as record_count 
            FROM stock_daily_prices 
            GROUP BY code 
            HAVING record_count >= 30  -- 至少有30天数据
            ORDER BY record_count DESC 
            LIMIT ?
            """
            result = self.db_handler.query_data(query, (limit,))
            
            if not result.empty:
                stock_list = result['code'].tolist()
                print(f"✅ 筛选出 {len(stock_list)} 只有足够数据的股票: {stock_list}")
                return stock_list
            else:
                print("❌ 没有找到有足够数据的股票")
                return []
                
        except Exception as e:
            print(f"❌ 获取样本股票失败: {e}")
            return []
    
    def create_data_manager(self, symbols, start_date, end_date):
        """创建数据管理器"""
        try:
            # 配置空的提供者列表，因为我们只从数据库读取
            provider_configs = []
            
            data_manager = DataProviderManager(
                provider_configs=provider_configs,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                db_path=self.db_path,
                auto_detect_universe=False  # 不自动检测，使用指定的symbols
            )
            
            print(f"✅ 数据管理器创建成功")
            return data_manager
            
        except Exception as e:
            print(f"❌ 数据管理器创建失败: {e}")
            return None
    
    def test_data_bridge_with_real_data(self, data_manager, symbols, start_date, end_date):
        """使用真实数据测试数据桥接器"""
        print("\n" + "=" * 60)
        print("测试数据桥接器（真实数据）")
        print("=" * 60)
        
        try:
            # 创建临时导出目录
            self.temp_dir = tempfile.mkdtemp(prefix='bt_real_test_')
            print(f"✅ 创建临时导出目录: {self.temp_dir}")
            
            # 创建导出器
            exporter = BTDataExporter(data_manager, self.temp_dir)
            
            # 生成模拟因子信号（基于真实价格数据）
            print("📊 生成基于真实数据的因子信号...")
            combined_signal = self._generate_real_based_signals(data_manager, symbols, start_date, end_date)
            
            if combined_signal is None or combined_signal.empty:
                print("❌ 生成因子信号失败")
                return False, []
            
            print(f"✅ 生成因子信号: {len(combined_signal)} 个数据点")
            
            # 导出数据
            print("📤 导出数据到Backtrader格式...")
            exported_files = exporter.export(
                universe=symbols,
                start_date=start_date,
                end_date=end_date,
                factor_series=combined_signal
            )
            
            print(f"✅ 数据导出完成: {len(exported_files)} 个文件")
            
            # 验证导出文件
            for file_path in exported_files:
                if os.path.exists(file_path):
                    df = pd.read_parquet(file_path)
                    print(f"   📁 {os.path.basename(file_path)}: {len(df)} 行数据")
                    
                    # 验证必要列
                    required_cols = ['open', 'high', 'low', 'close', 'volume', 
                                   'combined_signal', 'suspended']
                    missing_cols = [c for c in required_cols if c not in df.columns]
                    
                    if missing_cols:
                        print(f"   ❌ 缺少列: {missing_cols}")
                        return False, []
                    else:
                        print(f"   ✅ 所有必要列都存在")
                    
                    # 验证数据质量
                    non_nan_signals = df['combined_signal'].notna().sum()
                    print(f"   📊 有效信号数量: {non_nan_signals}/{len(df)}")
                    
                    # 检查价格数据
                    price_valid = df[['open', 'high', 'low', 'close']].notna().all().all()
                    print(f"   💰 价格数据完整性: {'✅' if price_valid else '❌'}")
                else:
                    print(f"   ❌ 文件不存在: {file_path}")
                    return False, []
            
            return True, exported_files
            
        except Exception as e:
            print(f"❌ 数据桥接器测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False, []
    
    def _generate_real_based_signals(self, data_manager, symbols, start_date, end_date):
        """基于真实价格数据生成因子信号"""
        try:
            print("🔄 正在基于真实价格数据生成因子信号...")
            
            # 获取所有股票的价格数据
            all_data = data_manager.get_all_data_for_universe(symbols, required_columns=['close'])
            
            if all_data is None or all_data.empty:
                print("❌ 无法获取价格数据")
                return None
            
            print(f"📈 获取到 {all_data.shape[0]} 行价格数据")
            
            # 计算简单的动量因子作为示例
            # 动量因子：近期收益率
            momentum_signals = []
            
            for asset in symbols:
                try:
                    # 获取单个股票数据
                    asset_data = all_data[all_data.index.get_level_values('asset') == asset]
                    
                    if asset_data.empty:
                        continue
                    
                    # 计算5日收益率作为信号
                    close_prices = asset_data['close']
                    returns_5d = close_prices.pct_change(5).fillna(0)
                    
                    # 创建信号序列
                    for date_idx, (date, asset_idx) in enumerate(asset_data.index):
                        if date_idx >= 5:  # 跳过前5天（因为需要计算收益率）
                            signal_value = returns_5d.iloc[date_idx]
                            momentum_signals.append({
                                'date': date,
                                'asset': asset,
                                'combined_signal': signal_value
                            })
                            
                except Exception as e:
                    print(f"⚠️ 处理股票 {asset} 时出错: {e}")
                    continue
            
            if not momentum_signals:
                print("❌ 没有生成任何信号")
                return None
            
            # 转换为DataFrame并创建MultiIndex Series
            signals_df = pd.DataFrame(momentum_signals)
            signals_df.set_index(['date', 'asset'], inplace=True)
            signals_series = signals_df['combined_signal']
            
            print(f"✅ 生成 {len(signals_series)} 个基于真实数据的信号")
            return signals_series
            
        except Exception as e:
            print(f"❌ 生成基于真实数据的信号失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_backtrader_with_real_data(self, exported_files):
        """使用真实数据测试Backtrader策略"""
        print("\n" + "=" * 60)
        print("测试Backtrader策略（真实数据）")
        print("=" * 60)
        
        try:
            # 创建Cerebro引擎
            cerebro = bt.Cerebro()
            cerebro.broker.setcash(100000.0)
            
            # 从导出文件加载数据
            print(f"📂 从 {len(exported_files)} 个导出文件加载真实数据...")
            
            for file_path in exported_files:
                if os.path.exists(file_path):
                    # 加载数据
                    df = pd.read_parquet(file_path)
                    asset_name = os.path.basename(file_path).replace('.parquet', '')
                    
                    # 检查数据质量
                    if df.empty or df[['open', 'high', 'low', 'close']].isna().any().any():
                        print(f"⚠️ 跳过质量不高的数据文件: {asset_name}")
                        continue
                    
                    # 创建FactorPandasData
                    data = FactorPandasData(dataname=df, name=asset_name)
                    cerebro.adddata(data)
                    
                    print(f"   ✅ 加载 {asset_name}: {len(df)} 行有效数据")
            
            if len(cerebro.datas) == 0:
                print("❌ 没有加载任何有效数据")
                return False
            
            # 创建动量触发器
            class MomentumTrigger:
                def __init__(self, strategy, momentum_window=5):
                    self.strategy = strategy
                    self.momentum_window = momentum_window
                    self.step_count = 0  # 时间步计数器
                
                def check_and_execute(self):
                    """基于动量的交易策略"""
                    self.step_count += 1
                    
                    # 打印当前时间步
                    try:
                        current_date = self.strategy.datetime.date(0)
                        print(f"📅 时间步 {self.step_count}: {current_date}")
                    except:
                        print(f"📅 时间步 {self.step_count}: [未知日期]")
                    
                    for data in self.strategy.datas:
                        if len(data) < self.momentum_window + 1:
                            continue
                        
                        # 计算动量信号
                        try:
                            recent_prices = []
                            for i in range(self.momentum_window):
                                price_value = data.close.get(ago=-i)
                                if hasattr(price_value, '__len__') and len(price_value) > 0:
                                    price = float(price_value[0])  # 取数组第一个元素
                                else:
                                    price = float(price_value)
                                recent_prices.append(price)
                            
                            current_price_value = data.close[0]
                            if hasattr(current_price_value, '__len__') and len(current_price_value) > 0:
                                current_price = float(current_price_value[0])
                            else:
                                current_price = float(current_price_value)
                            
                            if all(price > 0 for price in recent_prices):
                                # 计算价格趋势
                                momentum = (current_price - recent_prices[-1]) / recent_prices[-1]
                                
                                if momentum > 0.005:  # 降低阈值到0.5%买入
                                    self.strategy.submit_action(
                                        data=data,
                                        action=ActionType.BUY,
                                        size=100,
                                        reason=f"动量买入: {momentum:.3f}",
                                        priority=ActionPriority.REBALANCE
                                    )
                                elif momentum < -0.005:  # 降低阈值到0.5%卖出
                                    self.strategy.submit_action(
                                        data=data,
                                        action=ActionType.SELL,
                                        size=100,
                                        reason=f"动量卖出: {momentum:.3f}",
                                        priority=ActionPriority.REBALANCE
                                    )
                        except Exception as e:
                            # 忽略无法处理的price数据
                            continue
            
            print(f"\n🧠 添加动量策略到引擎...")
            cerebro.addstrategy(
                BacktestStrategy,
                triggers=[lambda s: MomentumTrigger(s, momentum_window=5)]
            )
            
            # 添加分析器
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            
            print("✅ 策略框架初始化完成")
            
            # 运行回测
            print("🚀 运行真实数据回测...")
            initial_value = cerebro.broker.getvalue()
            results = cerebro.run()
            final_value = cerebro.broker.getvalue()
            
            # 输出结果
            print(f"\n📊 真实数据回测结果:")
            print(f"   初始资金: {initial_value:,.2f}")
            print(f"   最终资金: {final_value:,.2f}")
            print(f"   总收益: {final_value - initial_value:,.2f} ({(final_value/initial_value - 1)*100:.2f}%)")
            
            # 分析器结果
            strat = results[0]
            trades = strat.analyzers.trades.get_analysis()
            sharpe = strat.analyzers.sharpe.get_analysis()
            drawdown = strat.analyzers.drawdown.get_analysis()
            
            print(f"\n📊 交易统计:")
            total_trades = trades.get('total', {}).get('total', 0)
            won_trades = trades.get('won', {}).get('total', 0)
            lost_trades = trades.get('lost', {}).get('total', 0)
            
            print(f"   总交易次数: {total_trades}")
            print(f"   盈利交易: {won_trades}")
            print(f"   亏损交易: {lost_trades}")
            
            if total_trades > 0:
                win_rate = won_trades / total_trades * 100
                print(f"   胜率: {win_rate:.1f}%")
            
            print(f"\n⚠️ 风险指标:")
            print(f"   夏普比率: {sharpe.get('sharperatio', 'N/A')}")
            print(f"   最大回撤: {drawdown.max.drawdown:.2f}%")
            
            print("✅ 真实数据Backtrader测试完成")
            return True
            
        except Exception as e:
            print(f"❌ Backtrader真实数据测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup(self):
        """清理临时文件"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 清理临时目录: {self.temp_dir}")


def main():
    """
    主测试函数
    """
    print("🎯 Backtrader真实数据库集成测试")
    print("=" * 80)
    
    # 设置日志
    setup_logging(log_dir='logs', log_prefix='bt_real_db_test')
    
    # 测试参数
    db_path = 'database/quant_data.db'  # 数据库在database目录下
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    tester = None
    try:
        # 初始化测试器
        tester = DatabaseBacktraderTest(db_path)
        
        # 检查数据库内容
        sample_stocks = tester.get_sample_stocks(limit=3)  # 使用3只股票进行测试
        if not sample_stocks:
            print("❌ 无法获取样本股票，测试终止")
            return False
        
        # 创建数据管理器
        data_manager = tester.create_data_manager(sample_stocks, start_date, end_date)
        if not data_manager:
            print("❌ 数据管理器创建失败，测试终止")
            return False
        
        # 测试数据桥接器
        bridge_success, exported_files = tester.test_data_bridge_with_real_data(
            data_manager, sample_stocks, start_date, end_date
        )
        
        if not bridge_success:
            print("❌ 数据桥接器测试失败，测试终止")
            return False
        
        # 测试Backtrader策略
        strategy_success = tester.test_backtrader_with_real_data(exported_files)
        
        if strategy_success:
            print(f"\n🎉 真实数据库集成测试全部通过！")
            print(f"\n✅ 验证结果:")
            print(f"   1. 数据库连接和数据获取 ✅")
            print(f"   2. 数据桥接器处理真实数据 ✅")
            print(f"   3. 策略框架在真实数据上运行 ✅")
            return True
        else:
            print(f"\n❌ 部分测试失败")
            return False
            
    except Exception as e:
        print(f"\n💥 测试过程中发生未处理的错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理
        if tester:
            tester.cleanup()


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)