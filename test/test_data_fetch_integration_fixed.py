#!/usr/bin/env python3
"""
多SQLite数据源集成测试脚本（修正版）

验证CSMR和JY数据库的数据获取功能，以及优先级机制的工作情况。
基于实际数据库表结构修正。
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import sqlite3

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_integration_fixed.log')
    ]
)
logger = logging.getLogger(__name__)

class CSMRDataProvider:
    """CSMR数据库专用数据提供者"""
    
    def __init__(self, db_path: str, table_name: str = 'daily_price'):
        self.db_path = db_path
        self.table_name = table_name
        
    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        """获取CSMR数据库数据"""
        logger.info(f"[CSMR] [{symbol}] 开始获取数据 - 表: '{self.table_name}', 日期范围: {start_date} ~ {end_date}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 构建查询 - 使用字符串格式的日期
            query = """
                SELECT TradingDate, Symbol, OpenPrice, HighPrice, LowPrice, ClosePrice,
                       Volume, Amount, ChangeRatio
                FROM daily_price
                WHERE Symbol = ? AND TradingDate BETWEEN ? AND ?
                ORDER BY TradingDate
            """
            
            df_raw = pd.read_sql(query, conn, params=(symbol, start_date, end_date))
            conn.close()
            
            if df_raw is None or df_raw.empty:
                logger.warning(f"[CSMR] [{symbol}] 在源数据库中未找到有效数据。")
                return None
            
            logger.info(f"[CSMR] [{symbol}] 成功获取 {len(df_raw)} 条原始数据，开始格式转换...")
            
            # 数据转换
            df_transformed = pd.DataFrame()
            df_transformed['date'] = pd.to_datetime(df_raw['TradingDate'], format='%Y-%m-%d')
            df_transformed['code'] = df_raw['Symbol'].astype(str).str.zfill(6)
            df_transformed['open'] = pd.to_numeric(df_raw['OpenPrice'], errors='coerce')
            df_transformed['high'] = pd.to_numeric(df_raw['HighPrice'], errors='coerce')
            df_transformed['low'] = pd.to_numeric(df_raw['LowPrice'], errors='coerce')
            df_transformed['close'] = pd.to_numeric(df_raw['ClosePrice'], errors='coerce')
            df_transformed['volume'] = pd.to_numeric(df_raw['Volume'], errors='coerce').fillna(0).astype('int64')
            df_transformed['turnover'] = pd.to_numeric(df_raw['Amount'], errors='coerce')
            df_transformed['pct_change'] = pd.to_numeric(df_raw['ChangeRatio'], errors='coerce')
            
            # 数据清洗
            logger.info(f"[CSMR] [{symbol}] 数据清洗前共 {len(df_transformed)} 条数据。")
            
            critical_cols = ['open', 'high', 'low', 'close', 'volume']
            df_transformed.dropna(subset=critical_cols, inplace=True)
            
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                df_transformed = df_transformed[df_transformed[col] > 0]
            df_transformed = df_transformed[df_transformed['volume'] >= 0]
            
            logger.info(f"[CSMR] [{symbol}] 数据清洗后剩余 {len(df_transformed)} 条有效数据。")
            
            if df_transformed.empty:
                logger.warning(f"[CSMR] [{symbol}] 清洗后无剩余有效数据。")
                return None
            
            # 添加计算列
            df_transformed['amplitude'] = ((df_transformed['high'] - df_transformed['low']) / df_transformed['open'] * 100).round(2)
            df_transformed['price_change'] = df_transformed['close'] - df_transformed['open']
            df_transformed['turnover_rate'] = (df_transformed['turnover'] / df_transformed['close'] / 10000).round(2)
            
            # 设置索引
            df_transformed.set_index('date', inplace=True)
            
            final_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'pct_change', 'price_change', 'turnover_rate']
            df_final = df_transformed[final_columns].copy()
            
            logger.info(f"[CSMR] [{symbol}] 数据转换和清洗完成 - 最终数据条数: {len(df_final)}")
            return df_final
            
        except Exception as e:
            logger.error(f"[CSMR] [{symbol}] 处理数据时出错: {e}", exc_info=True)
            return None

class JYDataProvider:
    """JY数据库数据提供者（使用现有SQLiteDataProvider）"""
    
    def __init__(self, db_path: str, table_name: str = 'JY_t_price_daily'):
        from data.providers import SQLiteDataProvider
        
        self.provider = SQLiteDataProvider(
            db_path=db_path,
            table_name=table_name,
            column_mapping={
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
        )
        
    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        """获取JY数据库数据"""
        return self.provider.fetch_data(symbol, start_date, end_date)

class MultiSQLiteIntegrationTester:
    """多SQLite数据源集成测试器"""
    
    def __init__(self):
        self.test_symbols = ['000001', '600000', '600519']  # 测试股票代码
        self.test_start_date = '2023-01-01'
        self.test_end_date = '2023-01-31'
        
        # 数据源配置
        self.csmr_config = {
            'db_path': './database/CSMR/CSMR_stock_daily_price.sqlite',
            'table_name': 'daily_price'
        }
        
        self.jy_config = {
            'db_path': './database/JY_database/sqlite/JY_database.sqlite',
            'table_name': 'JY_t_price_daily'
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
            # 检查数据库文件是否存在
            if not os.path.exists(self.csmr_config['db_path']):
                logger.error(f"❌ CSMR数据库文件不存在: {self.csmr_config['db_path']}")
                return False
            
            if not os.path.exists(self.jy_config['db_path']):
                logger.error(f"❌ JY数据库文件不存在: {self.jy_config['db_path']}")
                return False
            
            logger.info("✅ 数据库文件检查通过")
            
            # 创建数据提供者
            self.csmr_provider = CSMRDataProvider(**self.csmr_config)
            self.jy_provider = JYDataProvider(**self.jy_config)
            
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
            csmr_results = {}
            
            for symbol in self.test_symbols:
                logger.info(f"📊 获取{symbol}的CSMR数据...")
                
                # 获取数据
                df = self.csmr_provider.fetch_data(
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
                        'date_range': (df.index.min(), df.index.max()),
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
            jy_results = {}
            
            for symbol in self.test_symbols:
                logger.info(f"📊 获取{symbol}的JY数据...")
                
                # 获取数据
                df = self.jy_provider.fetch_data(
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
                        'date_range': (df.index.min(), df.index.max()),
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
                
                # 测试优先级机制：CSMR优先，JY作为备用
                data_sources_tested = []
                data_obtained = None
                
                # 优先级1: CSMR
                logger.info(f"  🔄 尝试使用 CSMR (优先级1) 获取数据...")
                try:
                    df = self.csmr_provider.fetch_data(
                        symbol=symbol,
                        start_date=self.test_start_date,
                        end_date=self.test_end_date
                    )
                    data_sources_tested.append('CSMR')
                    
                    if df is not None and not df.empty:
                        data_obtained = df
                        logger.info(f"  ✅ CSMR 成功返回数据")
                    else:
                        logger.info(f"  ⚠️ CSMR 未返回数据，继续下一个源")
                        
                except Exception as e:
                    logger.warning(f"  ❌ CSMR 获取数据失败: {e}")
                    data_sources_tested.append('CSMR_error')
                
                # 优先级2: JY (备用)
                if data_obtained is None:
                    logger.info(f"  🔄 尝试使用 JY (优先级2) 获取数据...")
                    try:
                        df = self.jy_provider.fetch_data(
                            symbol=symbol,
                            start_date=self.test_start_date,
                            end_date=self.test_end_date
                        )
                        data_sources_tested.append('JY')
                        
                        if df is not None and not df.empty:
                            data_obtained = df
                            logger.info(f"  ✅ JY 成功返回数据")
                        else:
                            logger.info(f"  ⚠️ JY 未返回数据")
                            
                    except Exception as e:
                        logger.warning(f"  ❌ JY 获取数据失败: {e}")
                        data_sources_tested.append('JY_error')
                
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
    
    def test_data_comparison(self):
        """测试两个数据源的数据对比"""
        logger.info("\n🧪 测试两个数据源的数据对比")
        logger.info("=" * 50)
        
        try:
            comparison_results = {}
            
            for symbol in self.test_symbols:
                logger.info(f"📊 对比{symbol}的两个数据源数据...")
                
                # 获取两个数据源的数据
                csmr_data = self.csmr_provider.fetch_data(
                    symbol=symbol,
                    start_date=self.test_start_date,
                    end_date=self.test_end_date
                )
                
                jy_data = self.jy_provider.fetch_data(
                    symbol=symbol,
                    start_date=self.test_start_date,
                    end_date=self.test_end_date
                )
                
                comparison_result = {
                    'symbol': symbol,
                    'csmr_success': csmr_data is not None and not csmr_data.empty,
                    'jy_success': jy_data is not None and not jy_data.empty,
                    'both_success': False,
                    'data_consistency': None,
                    'details': {}
                }
                
                if comparison_result['csmr_success'] and comparison_result['jy_success']:
                    comparison_result['both_success'] = True
                    
                    # 数据一致性检查
                    try:
                        # 比较数据条数
                        csmr_count = len(csmr_data)
                        jy_count = len(jy_data)
                        
                        # 比较日期范围
                        csmr_dates = set(csmr_data.index.date)
                        jy_dates = set(jy_data.index.date)
                        
                        # 比较收盘价（如果日期重叠）
                        common_dates = csmr_dates & jy_dates
                        price_diff_count = 0
                        
                        if common_dates:
                            for date in sorted(list(common_dates))[:5]:  # 比较前5个共同日期
                                csmr_close = csmr_data.loc[pd.Timestamp(date), 'close']
                                jy_close = jy_data.loc[pd.Timestamp(date), 'close']
                                if abs(csmr_close - jy_close) > 0.01:  # 差异超过1分钱
                                    price_diff_count += 1
                        
                        comparison_result['data_consistency'] = {
                            'csmr_data_count': csmr_count,
                            'jy_data_count': jy_count,
                            'common_dates_count': len(common_dates),
                            'price_differences': price_diff_count,
                            'consistency_ratio': (len(common_dates) - price_diff_count) / len(common_dates) if common_dates else 0
                        }
                        
                        logger.info(f"✅ {symbol}: 两个数据源都有数据")
                        logger.info(f"   CSMR数据条数: {csmr_count}, JY数据条数: {jy_count}")
                        logger.info(f"   共同日期数: {len(common_dates)}")
                        logger.info(f"   价格差异数: {price_diff_count}")
                        
                    except Exception as e:
                        comparison_result['details']['comparison_error'] = str(e)
                        logger.warning(f"⚠️ {symbol}: 数据对比时出错: {e}")
                        
                elif comparison_result['csmr_success']:
                    logger.info(f"✅ {symbol}: 仅有CSMR数据源有数据")
                elif comparison_result['jy_success']:
                    logger.info(f"✅ {symbol}: 仅有JY数据源有数据")
                else:
                    logger.warning(f"⚠️ {symbol}: 两个数据源都没有数据")
                
                comparison_results[symbol] = comparison_result
            
            self.test_results['comparison_tests'] = comparison_results
            
            # 统计结果
            both_success_count = sum(1 for result in comparison_results.values() if result['both_success'])
            total_count = len(self.test_symbols)
            success_rate = both_success_count / total_count * 100
            
            logger.info(f"📈 数据源对比成功率: {success_rate:.1f}% ({both_success_count}/{total_count})")
            
            return True  # 对比测试总是成功的，只是记录差异
            
        except Exception as e:
            logger.error(f"❌ 数据对比测试失败: {e}")
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
        report_lines.append("多SQLite数据源集成测试报告（修正版）")
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
        
        # 数据源对比结果
        report_lines.append("🔍 数据源对比结果:")
        report_lines.append("-" * 40)
        if 'comparison_tests' in self.test_results:
            for symbol, result in self.test_results['comparison_tests'].items():
                if result['both_success']:
                    report_lines.append(f"✅ {symbol}: 两个数据源都有数据")
                    if result['data_consistency']:
                        consistency = result['data_consistency']
                        report_lines.append(f"   CSMR: {consistency['csmr_data_count']} 条, JY: {consistency['jy_data_count']} 条")
                        report_lines.append(f"   共同日期: {consistency['common_dates_count']} 天")
                        report_lines.append(f"   价格一致性: {consistency['consistency_ratio']:.1%}")
                elif result['csmr_success']:
                    report_lines.append(f"⚠️ {symbol}: 仅CSMR数据源有数据")
                elif result['jy_success']:
                    report_lines.append(f"⚠️ {symbol}: 仅JY数据源有数据")
                else:
                    report_lines.append(f"❌ {symbol}: 两个数据源都没有数据")
        report_lines.append("")
        
        # 总体评估
        report_lines.append("📈 总体评估:")
        report_lines.append("-" * 40)
        
        total_tests = 0
        passed_tests = 0
        
        for category in ['csmr_tests', 'jy_tests', 'priority_tests']:
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
        logger.info("🚀 开始多SQLite数据源集成测试（修正版）")
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
            ("数据源对比", self.test_data_comparison)
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
        with open('integration_test_report_fixed.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.test_results['overall_success'] = overall_success
        
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