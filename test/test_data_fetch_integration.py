#!/usr/bin/env python3
"""
多SQLite数据源集成测试脚本

验证CSMR和JY数据库的数据获取功能，以及优先级机制的工作情况。
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_integration.log')
    ]
)
logger = logging.getLogger(__name__)

class MultiSQLiteIntegrationTester:
    """多SQLite数据源集成测试器"""
    
    def __init__(self):
        self.test_symbols = ['000001', '600000', '600519']  # 测试股票代码
        self.test_start_date = '2023-01-01'
        self.test_end_date = '2023-01-31'
        
        # 数据源配置 - 根据实际数据库表结构修正
        self.csmr_config = {
            'db_path': './database/CSMR/CSMR_stock_daily_price.sqlite',
            'table_name': 'daily_price',
            'column_mapping': {
                'TradingDate': 'date',
                'Symbol': 'code',
                'OpenPrice': 'open',
                'HighPrice': 'high',
                'LowPrice': 'low',
                'ClosePrice': 'close',
                'Volume': 'volume',
                'Amount': 'turnover',
                'ChangeRatio': 'pct_change'
            }
        }
        
        self.jy_config = {
            'db_path': './database/JY_database/sqlite/JY_database.sqlite',
            'table_name': 'JY_t_price_daily',
            'column_mapping': {
                '_date': 'date',
                'ticker': 'code',
                '_open': 'open',
                '_high': 'high',
                '_low': 'low',
                '_close': 'close',
                '_volume': 'volume',
                '_value': 'turnover',
                '_return': 'pct_change'
            }
        }
        
        # 测试结果
        self.test_results = {
            'csmr_tests': {},
            'jy_tests': {},
            'priority_tests': {},
            'integration_tests': {},
            'overall_success': True
        }
    
    def setup_test_environment(self):
        """设置测试环境"""
        logger.info("🔧 设置测试环境...")
        
        try:
            # 导入必要的模块
            from data.providers import SQLiteDataProvider
            from data.manager import DataProviderManager
            
            # 检查数据库文件是否存在
            if not os.path.exists(self.csmr_config['db_path']):
                logger.error(f"❌ CSMR数据库文件不存在: {self.csmr_config['db_path']}")
                return False
            
            if not os.path.exists(self.jy_config['db_path']):
                logger.error(f"❌ JY数据库文件不存在: {self.jy_config['db_path']}")
                return False
            
            logger.info("✅ 数据库文件检查通过")
            
            # 创建数据提供者配置
            self.provider_configs = [
                ('sqlite_csmr', SQLiteDataProvider, self.csmr_config),
                ('sqlite_jy', SQLiteDataProvider, self.jy_config)
            ]
            
            # 创建数据提供者管理器
            self.data_manager = DataProviderManager(
                provider_configs=self.provider_configs,
                symbols=self.test_symbols,
                start_date=self.test_start_date,
                end_date=self.test_end_date,
                db_path='./test_integration_quant_data.db',
                auto_detect_universe=False
            )
            
            logger.info("✅ 测试环境设置完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试环境设置失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def test_csmr_data_fetch(self):
        """测试CSMR数据库数据获取功能"""
        logger.info("\n🧪 测试CSMR数据库数据获取功能")
        logger.info("=" * 50)
        
        try:
            from data.providers import SQLiteDataProvider
            
            # 创建CSMR数据提供者
            csmr_provider = SQLiteDataProvider(**self.csmr_config)
            
            csmr_results = {}
            
            for symbol in self.test_symbols:
                logger.info(f"📊 获取{symbol}的CSMR数据...")
                
                # 获取数据
                df = csmr_provider.fetch_data(
                    symbol=symbol,
                    start_date=self.test_start_date,
                    end_date=self.test_end_date
                )
                
                if df is not None and not df.empty:
                    # 数据验证
                    validation_result = self._validate_dataframe(df, symbol, "CSMR")
                    
                    csmr_results[symbol] = {
                        'success': True,
                        'data_shape': df.shape,
                        'date_range': (df.index.min(), df.index.max()) if hasattr(df.index, 'min') else ('N/A', 'N/A'),
                        'columns': list(df.columns),
                        'validation': validation_result,
                        'sample_data': df.head(2).to_dict() if len(df) > 0 else {}
                    }
                    
                    logger.info(f"✅ {symbol}: 成功获取 {len(df)} 条数据")
                    logger.info(f"   日期范围: {csmr_results[symbol]['date_range']}")
                    logger.info(f"   列名: {csmr_results[symbol]['columns']}")
                    
                else:
                    csmr_results[symbol] = {
                        'success': False,
                        'error': 'No data returned'
                    }
                    logger.warning(f"⚠️ {symbol}: 未获取到数据")
            
            self.test_results['csmr_tests'] = csmr_results
            
            # 统计成功率
            success_count = sum(1 for result in csmr_results.values() if result['success'])
            total_count = len(self.test_symbols)
            success_rate = success_count / total_count * 100
            
            logger.info(f"📈 CSMR数据获取成功率: {success_rate:.1f}% ({success_count}/{total_count})")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ CSMR数据获取测试失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def test_jy_data_fetch(self):
        """测试JY数据库数据获取功能"""
        logger.info("\n🧪 测试JY数据库数据获取功能")
        logger.info("=" * 50)
        
        try:
            from data.providers import SQLiteDataProvider
            
            # 创建JY数据提供者
            jy_provider = SQLiteDataProvider(**self.jy_config)
            
            jy_results = {}
            
            for symbol in self.test_symbols:
                logger.info(f"📊 获取{symbol}的JY数据...")
                
                # 获取数据
                df = jy_provider.fetch_data(
                    symbol=symbol,
                    start_date=self.test_start_date,
                    end_date=self.test_end_date
                )
                
                if df is not None and not df.empty:
                    # 数据验证
                    validation_result = self._validate_dataframe(df, symbol, "JY")
                    
                    jy_results[symbol] = {
                        'success': True,
                        'data_shape': df.shape,
                        'date_range': (df.index.min(), df.index.max()) if hasattr(df.index, 'min') else ('N/A', 'N/A'),
                        'columns': list(df.columns),
                        'validation': validation_result,
                        'sample_data': df.head(2).to_dict() if len(df) > 0 else {}
                    }
                    
                    logger.info(f"✅ {symbol}: 成功获取 {len(df)} 条数据")
                    logger.info(f"   日期范围: {jy_results[symbol]['date_range']}")
                    logger.info(f"   列名: {jy_results[symbol]['columns']}")
                    
                else:
                    jy_results[symbol] = {
                        'success': False,
                        'error': 'No data returned'
                    }
                    logger.warning(f"⚠️ {symbol}: 未获取到数据")
            
            self.test_results['jy_tests'] = jy_results
            
            # 统计成功率
            success_count = sum(1 for result in jy_results.values() if result['success'])
            total_count = len(self.test_symbols)
            success_rate = success_count / total_count * 100
            
            logger.info(f"📈 JY数据获取成功率: {success_rate:.1f}% ({success_count}/{total_count})")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ JY数据获取测试失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def test_priority_mechanism(self):
        """测试数据获取优先级机制"""
        logger.info("\n🧪 测试数据获取优先级机制")
        logger.info("=" * 50)
        
        try:
            priority_results = {}
            
            for symbol in self.test_symbols:
                logger.info(f"🎯 测试{symbol}的优先级机制...")
                
                # 测试DataProviderManager的优先级机制
                # 这里模拟DataProviderManager的数据获取流程
                data_sources_tested = []
                data_obtained = None
                
                # 按照优先级顺序尝试获取数据
                for provider_name, provider_class, config in self.provider_configs:
                    logger.info(f"  🔄 尝试使用 {provider_name} 获取数据...")
                    
                    try:
                        provider = provider_class(**config)
                        df = provider.fetch_data(
                            symbol=symbol,
                            start_date=self.test_start_date,
                            end_date=self.test_end_date
                        )
                        
                        data_sources_tested.append(provider_name)
                        
                        if df is not None and not df.empty:
                            data_obtained = df
                            logger.info(f"  ✅ {provider_name} 成功返回数据")
                            break
                        else:
                            logger.info(f"  ⚠️ {provider_name} 未返回数据，继续下一个源")
                            
                    except Exception as e:
                        logger.warning(f"  ❌ {provider_name} 获取数据失败: {e}")
                        data_sources_tested.append(f"{provider_name}_error")
                        continue
                
                # 验证优先级机制结果
                if data_obtained is not None:
                    validation_result = self._validate_dataframe(data_obtained, symbol, "优先级测试")
                    
                    priority_results[symbol] = {
                        'success': True,
                        'data_shape': data_obtained.shape,
                        'sources_tested': data_sources_tested,
                        'first_success_source': data_sources_tested[len([s for s in data_sources_tested if not s.endswith('_error')]) - 1] if data_sources_tested else None,
                        'validation': validation_result
                    }
                    
                    logger.info(f"✅ {symbol}: 优先级机制工作正常")
                    logger.info(f"   测试的数据源: {data_sources_tested}")
                    logger.info(f"   首个成功源: {priority_results[symbol]['first_success_source']}")
                    
                else:
                    priority_results[symbol] = {
                        'success': False,
                        'sources_tested': data_sources_tested,
                        'error': 'All data sources failed'
                    }
                    logger.error(f"❌ {symbol}: 所有数据源都无法获取数据")
            
            self.test_results['priority_tests'] = priority_results
            
            # 统计成功率
            success_count = sum(1 for result in priority_results.values() if result['success'])
            total_count = len(self.test_symbols)
            success_rate = success_count / total_count * 100
            
            logger.info(f"📈 优先级机制测试成功率: {success_rate:.1f}% ({success_count}/{total_count})")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ 优先级机制测试失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def test_global_data_fetch(self):
        """测试全局数据获取场景（模拟实际使用）"""
        logger.info("\n🧪 测试全局数据获取场景")
        logger.info("=" * 50)
        
        try:
            integration_results = {}
            
            # 使用DataProviderManager进行全局测试
            test_symbols_subset = self.test_symbols[:2]  # 取前两个股票进行测试
            
            for symbol in test_symbols_subset:
                logger.info(f"🌐 测试{symbol}的全局数据获取...")
                
                try:
                    # 使用DataProviderManager获取数据
                    df = self.data_manager.get_dataframe(symbol)
                    
                    if df is not None and not df.empty:
                        validation_result = self._validate_dataframe(df, symbol, "全局测试")
                        
                        integration_results[symbol] = {
                            'success': True,
                            'data_shape': df.shape,
                            'date_range': (df.index.min(), df.index.max()),
                            'columns': list(df.columns),
                            'validation': validation_result,
                            'sample_data': df.head(2).to_dict() if len(df) > 0 else {}
                        }
                        
                        logger.info(f"✅ {symbol}: 全局数据获取成功")
                        logger.info(f"   数据形状: {df.shape}")
                        logger.info(f"   日期范围: {integration_results[symbol]['date_range']}")
                        
                    else:
                        integration_results[symbol] = {
                            'success': False,
                            'error': 'No data returned from manager'
                        }
                        logger.warning(f"⚠️ {symbol}: DataProviderManager未返回数据")
                        
                except Exception as e:
                    integration_results[symbol] = {
                        'success': False,
                        'error': str(e)
                    }
                    logger.error(f"❌ {symbol}: 全局数据获取失败 - {e}")
            
            # 测试多股票批量获取
            logger.info(f"📊 测试多股票批量获取...")
            try:
                universe_data = self.data_manager.get_all_data_for_universe(
                    universe=test_symbols_subset,
                    required_columns=['close', 'volume']
                )
                
                if universe_data is not None and not universe_data.empty:
                    logger.info(f"✅ 批量获取成功，数据形状: {universe_data.shape}")
                    integration_results['batch_fetch'] = {
                        'success': True,
                        'data_shape': universe_data.shape,
                        'index_names': universe_data.index.names if hasattr(universe_data, 'index') else None,
                        'columns': list(universe_data.columns)
                    }
                else:
                    integration_results['batch_fetch'] = {
                        'success': False,
                        'error': 'Batch fetch returned empty data'
                    }
                    logger.warning("⚠️ 批量获取未返回数据")
                    
            except Exception as e:
                integration_results['batch_fetch'] = {
                    'success': False,
                    'error': str(e)
                }
                logger.error(f"❌ 批量获取失败: {e}")
            
            self.test_results['integration_tests'] = integration_results
            
            # 统计成功率
            success_count = sum(1 for result in integration_results.values() if result['success'])
            total_count = len(integration_results)
            success_rate = success_count / total_count * 100 if total_count > 0 else 0
            
            logger.info(f"📈 全局数据获取成功率: {success_rate:.1f}% ({success_count}/{total_count})")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ 全局数据获取测试失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _validate_dataframe(self, df: pd.DataFrame, symbol: str, source: str) -> Dict:
        """验证DataFrame的数据质量"""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        try:
            # 基本结构检查
            if df.empty:
                validation_result['is_valid'] = False
                validation_result['issues'].append("DataFrame为空")
                return validation_result
            
            # 检查必要的列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_result['issues'].append(f"缺少必要列: {missing_columns}")
            
            # 检查数据类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        validation_result['issues'].append(f"列 {col} 不是数值类型")
            
            # 检查数据范围
            if 'close' in df.columns:
                close_data = df['close'].dropna()
                if len(close_data) > 0:
                    if (close_data <= 0).any():
                        validation_result['issues'].append("收盘价存在非正值")
                    
                    validation_result['statistics']['close_price_range'] = {
                        'min': float(close_data.min()),
                        'max': float(close_data.max()),
                        'mean': float(close_data.mean())
                    }
            
            # 检查日期索引
            if hasattr(df.index, 'name') and df.index.name == 'date':
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    validation_result['issues'].append("日期索引不是datetime类型")
            elif 'date' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    validation_result['issues'].append("date列不是datetime类型")
            
            # 数据完整性统计
            validation_result['statistics']['total_rows'] = len(df)
            validation_result['statistics']['null_counts'] = df.isnull().sum().to_dict()
            
            if validation_result['issues']:
                validation_result['is_valid'] = False
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"验证过程出错: {e}")
        
        return validation_result
    
    def generate_test_report(self) -> str:
        """生成详细的测试报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("多SQLite数据源集成测试报告")
        report_lines.append("=" * 80)
        report_lines.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"测试股票: {', '.join(self.test_symbols)}")
        report_lines.append(f"测试时间范围: {self.test_start_date} 到 {self.test_end_date}")
        report_lines.append("")
        
        # CSMR测试结果
        report_lines.append("📊 CSMR数据库测试结果:")
        report_lines.append("-" * 40)
        if 'csmr_tests' in self.test_results:
            for symbol, result in self.test_results['csmr_tests'].items():
                if result['success']:
                    report_lines.append(f"✅ {symbol}: 成功 ({result['data_shape'][0]} 条数据)")
                    if 'validation' in result and result['validation']['issues']:
                        report_lines.append(f"   ⚠️ 数据质量问题: {result['validation']['issues']}")
                else:
                    report_lines.append(f"❌ {symbol}: 失败 - {result.get('error', 'Unknown error')}")
        report_lines.append("")
        
        # JY测试结果
        report_lines.append("📊 JY数据库测试结果:")
        report_lines.append("-" * 40)
        if 'jy_tests' in self.test_results:
            for symbol, result in self.test_results['jy_tests'].items():
                if result['success']:
                    report_lines.append(f"✅ {symbol}: 成功 ({result['data_shape'][0]} 条数据)")
                    if 'validation' in result and result['validation']['issues']:
                        report_lines.append(f"   ⚠️ 数据质量问题: {result['validation']['issues']}")
                else:
                    report_lines.append(f"❌ {symbol}: 失败 - {result.get('error', 'Unknown error')}")
        report_lines.append("")
        
        # 优先级机制测试结果
        report_lines.append("🎯 优先级机制测试结果:")
        report_lines.append("-" * 40)
        if 'priority_tests' in self.test_results:
            for symbol, result in self.test_results['priority_tests'].items():
                if result['success']:
                    report_lines.append(f"✅ {symbol}: 优先级机制工作正常")
                    report_lines.append(f"   测试的数据源: {result['sources_tested']}")
                    report_lines.append(f"   首个成功源: {result['first_success_source']}")
                else:
                    report_lines.append(f"❌ {symbol}: 优先级机制失败")
                    report_lines.append(f"   测试的数据源: {result['sources_tested']}")
                    report_lines.append(f"   错误: {result.get('error', 'Unknown error')}")
        report_lines.append("")
        
        # 全局集成测试结果
        report_lines.append("🌐 全局集成测试结果:")
        report_lines.append("-" * 40)
        if 'integration_tests' in self.test_results:
            for test_name, result in self.test_results['integration_tests'].items():
                if result['success']:
                    report_lines.append(f"✅ {test_name}: 成功")
                    if 'data_shape' in result:
                        report_lines.append(f"   数据形状: {result['data_shape']}")
                else:
                    report_lines.append(f"❌ {test_name}: 失败 - {result.get('error', 'Unknown error')}")
        report_lines.append("")
        
        # 总体评估
        report_lines.append("📈 总体评估:")
        report_lines.append("-" * 40)
        
        total_tests = 0
        passed_tests = 0
        
        for category in ['csmr_tests', 'jy_tests', 'priority_tests', 'integration_tests']:
            if category in self.test_results:
                for result in self.test_results[category].values():
                    total_tests += 1
                    if result['success']:
                        passed_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        report_lines.append(f"总测试数: {total_tests}")
        report_lines.append(f"通过测试数: {passed_tests}")
        report_lines.append(f"成功率: {success_rate:.1f}%")
        
        if success_rate >= 80:
            report_lines.append("🎉 测试结果: 优秀 - 多SQLite数据源集成工作正常")
        elif success_rate >= 60:
            report_lines.append("👍 测试结果: 良好 - 基本功能正常，但存在一些问题")
        else:
            report_lines.append("⚠️ 测试结果: 需要改进 - 存在较多问题需要解决")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("🚀 开始多SQLite数据源集成测试")
        logger.info("=" * 60)
        
        # 设置测试环境
        if not self.setup_test_environment():
            logger.error("❌ 测试环境设置失败，终止测试")
            return False
        
        # 运行各项测试
        tests = [
            ("CSMR数据获取", self.test_csmr_data_fetch),
            ("JY数据获取", self.test_jy_data_fetch),
            ("优先级机制", self.test_priority_mechanism),
            ("全局数据获取", self.test_global_data_fetch)
        ]
        
        overall_success = True
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\n🧪 开始执行: {test_name}")
                success = test_func()
                if success:
                    logger.info(f"✅ {test_name} 测试通过")
                else:
                    logger.error(f"❌ {test_name} 测试失败")
                    overall_success = False
            except Exception as e:
                logger.error(f"❌ {test_name} 测试异常: {e}")
                overall_success = False
        
        # 生成测试报告
        logger.info("\n📋 生成测试报告...")
        report = self.generate_test_report()
        logger.info("\n" + report)
        
        # 保存报告到文件
        with open('integration_test_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.test_results['overall_success'] = overall_success
        
        # 清理测试环境
        try:
            if hasattr(self, 'data_manager'):
                del self.data_manager
            # 删除测试数据库文件
            test_db_path = './test_integration_quant_data.db'
            if os.path.exists(test_db_path):
                os.remove(test_db_path)
                logger.info("🧹 清理测试数据库文件")
        except Exception as e:
            logger.warning(f"⚠️ 清理测试环境时出现警告: {e}")
        
        return overall_success


def main():
    """主函数"""
    tester = MultiSQLiteIntegrationTester()
    success = tester.run_all_tests()
    
    if success:
        logger.info("🎉 所有测试通过！多SQLite数据源集成功能验证成功！")
        return 0
    else:
        logger.error("❌ 部分测试失败，请检查相关配置和数据源")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)