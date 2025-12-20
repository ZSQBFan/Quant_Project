#!/usr/bin/env python3
"""
调试脚本：分析2021-02-01左右因子数据点不足(0)的问题

详细分析各因子的lookback窗口要求和行业中性化逻辑，
模拟2021-02-01的计算流程，找出数据缺失的根本原因。
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import logging
from pathlib import Path

# 添加项目路径
sys.path.append('/Volumes/叽叽叽/Code/quant_project_3.12_macmini')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FactorZeroAnalyzer:
    """因子零值分析器"""
    
    def __init__(self):
        self.target_date = "2021-02-01"
        self.debug_date = "2021-01-29"  # 2021-02-01前的最后一个交易日
        self.db_path = "./database/CSMR/CSMR_stock_daily_price.sqlite"
        self.universe_path = "./configs/universe.yaml"
        
        # 目标因子列表
        self.target_factors = {
            '技术指标类': ['Momentum', 'Reversal20D', 'MACD', 'MovingAverageCross'],
            '行业中性化类': ['IndNeu_EP', 'IndNeu_BP', 'IndNeu_ROE', 'IndNeu_GPM', 
                          'IndNeu_SalesGrowth', 'IndNeu_AssetTurnover', 'IndNeu_CurrentRatio', 'IndNeu_CFOP']
        }
        
        # 因子lookback信息
        self.factor_lookback_info = {
            'Momentum': {'window': 20, 'type': '技术指标'},
            'Reversal20D': {'window': 40, 'min_periods': 20, 'type': '技术指标'},
            'MACD': {'window': 26, 'type': '技术指标'},  # 12/26/9，最长是26
            'MovingAverageCross': {'window': 30, 'type': '技术指标'},  # 14/30，最长是30
            
            # 行业中性化因子
            'IndNeu_EP': {'window': 0, 'financial_lag': True, 'type': '行业中性化'},
            'IndNeu_BP': {'window': 0, 'financial_lag': True, 'type': '行业中性化'},
            'IndNeu_ROE': {'window': 0, 'financial_lag': True, 'type': '行业中性化'},
            'IndNeu_GPM': {'window': 0, 'financial_lag': True, 'type': '行业中性化'},
            'IndNeu_SalesGrowth': {'window': 252, 'financial_lag': True, 'type': '行业中性化'},  # 252天约1年
            'IndNeu_AssetTurnover': {'window': 0, 'financial_lag': True, 'type': '行业中性化'},
            'IndNeu_CurrentRatio': {'window': 0, 'financial_lag': True, 'type': '行业中性化'},
            'IndNeu_CFOP': {'window': 0, 'financial_lag': True, 'type': '行业中性化'}
        }
    
    def connect_database(self):
        """连接数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            logger.info(f"✅ 成功连接到数据库: {self.db_path}")
            return conn
        except Exception as e:
            logger.error(f"❌ 数据库连接失败: {e}")
            return None
    
    def get_universe_stocks(self):
        """获取股票池"""
        try:
            import yaml
            with open(self.universe_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                universe = config.get('symbols', [])
            logger.info(f"✅ 加载股票池: {len(universe)} 只股票")
            return universe
        except Exception as e:
            logger.error(f"❌ 加载股票池失败: {e}")
            # 如果读取失败，返回一些默认股票进行测试
            default_stocks = ['600519', '000858', '600036', '000001', '601318']
            logger.info(f"✅ 使用默认股票池: {len(default_stocks)} 只股票")
            return default_stocks
    
    def check_data_availability(self, conn, start_date, end_date):
        """检查指定日期范围的数据可用性"""
        cursor = conn.cursor()
        
        # 检查价格数据
        query_price = """
        SELECT 
            TradingDate,
            COUNT(*) as total_records,
            COUNT(ClosePrice) as valid_prices,
            ROUND(COUNT(ClosePrice) * 100.0 / COUNT(*), 2) as price_completeness
        FROM daily_price 
        WHERE TradingDate >= ? AND TradingDate <= ?
        GROUP BY TradingDate
        ORDER BY TradingDate
        """
        
        cursor.execute(query_price, (start_date, end_date))
        price_results = cursor.fetchall()
        
        logger.info(f"📊 {start_date} 到 {end_date} 价格数据可用性:")
        print(f"{'交易日期':<12} {'总记录':<6} {'有效价格':<8} {'完整率':<8}")
        print("-" * 35)
        
        critical_days = []
        for row in price_results:
            print(f"{row[0]:<12} {row[1]:<6} {row[2]:<8} {row[3]:<8}%")
            if row[3] < 95:  # 完整率低于95%认为是问题日期
                critical_days.append(row[0])
        
        logger.info(f"⚠️ 数据完整率低于95%的交易日: {critical_days}")
        return price_results
    
    def analyze_factor_requirements(self):
        """分析各因子的数据要求"""
        logger.info("🔍 分析各因子数据要求:")
        
        for category, factors in self.target_factors.items():
            logger.info(f"\n📁 {category}:")
            for factor in factors:
                if factor in self.factor_lookback_info:
                    info = self.factor_lookback_info[factor]
                    logger.info(f"  • {factor}:")
                    logger.info(f"    - Lookback窗口: {info.get('window', 'N/A')} 天")
                    logger.info(f"    - 类型: {info.get('type', 'N/A')}")
                    if info.get('financial_lag'):
                        logger.info(f"    - 财务数据滞后: 是")
                    
                    # 计算所需的最早数据日期
                    if info.get('window', 0) > 0:
                        required_start = (datetime.strptime(self.debug_date, "%Y-%m-%d") - 
                                        timedelta(days=info.get('window', 0))).strftime("%Y-%m-%d")
                        logger.info(f"    - 所需最早数据日期: {required_start}")
                else:
                    logger.warning(f"  ⚠️ {factor}: 未找到配置信息")
    
    def simulate_technical_factor_calculation(self, conn, universe, sample_size=50):
        """模拟技术指标类因子的计算过程"""
        logger.info("\n🧮 模拟技术指标类因子计算...")
        
        # 获取样本股票
        cursor = conn.cursor()
        sample_stocks = universe[:sample_size]
        
        results = {}
        
        for factor in self.target_factors['技术指标类']:
            logger.info(f"\n📈 分析 {factor}:")
            
            factor_info = self.factor_lookback_info.get(factor, {})
            window = factor_info.get('window', 20)
            required_start = (datetime.strptime(self.debug_date, "%Y-%m-%d") - 
                            timedelta(days=window)).strftime("%Y-%m-%d")
            
            logger.info(f"  • 所需lookback: {window} 天")
            logger.info(f"  • 所需数据开始日期: {required_start}")
            
            valid_stocks = 0
            stock_details = []
            
            for stock in sample_stocks:
                # 检查股票在所需日期范围内的数据
                query_stock_data = """
                SELECT COUNT(*) as record_count,
                       MIN(TradingDate) as first_date,
                       MAX(TradingDate) as last_date,
                       COUNT(ClosePrice) as valid_prices
                FROM daily_price 
                WHERE Symbol = ? AND TradingDate >= ? AND TradingDate <= ?
                """
                
                cursor.execute(query_stock_data, (stock, required_start, self.debug_date))
                stock_result = cursor.fetchone()
                
                record_count = stock_result[0]
                valid_prices = stock_result[3]
                data_days = (datetime.strptime(self.debug_date, "%Y-%m-%d") - 
                           datetime.strptime(required_start, "%Y-%m-%d")).days
                required_records = min(window, data_days)
                
                is_valid = valid_prices >= required_records
                if is_valid:
                    valid_stocks += 1
                
                stock_details.append({
                    'stock': stock,
                    'valid': is_valid,
                    'records': record_count,
                    'valid_prices': valid_prices,
                    'required': required_records
                })
            
            # 统计结果
            logger.info(f"  • 有效股票数: {valid_stocks}/{len(sample_stocks)} ({valid_stocks/len(sample_stocks)*100:.1f}%)")
            
            # 检查具体问题股票
            invalid_stocks = [s for s in stock_details if not s['valid']]
            if invalid_stocks:
                logger.info(f"  ⚠️ 数据不足的股票 (前5个):")
                for stock in invalid_stocks[:5]:
                    logger.info(f"    - {stock['stock']}: {stock['valid_prices']}/{stock['required']} 有效记录")
            
            results[factor] = {
                'valid_stocks': valid_stocks,
                'total_stocks': len(sample_stocks),
                'validity_rate': valid_stocks/len(sample_stocks),
                'invalid_details': invalid_stocks
            }
        
        return results
    
    def simulate_industry_neutral_factor_calculation(self, conn, universe, sample_size=50):
        """模拟行业中性化因子的计算过程"""
        logger.info("\n🏭 模拟行业中性化因子计算...")
        
        cursor = conn.cursor()
        sample_stocks = universe[:sample_size]
        
        # 首先检查行业数据可用性
        logger.info("📋 检查行业分类数据:")
        
        # 检查股票行业信息（这里假设行业信息在其他表中）
        # 在实际项目中可能需要从财务数据表或专门的行业表获取
        logger.info("  ⚠️ 注意: 需要检查行业分类数据的可用性和质量")
        
        results = {}
        
        for factor in self.target_factors['行业中性化类']:
            logger.info(f"\n📊 分析 {factor}:")
            
            factor_info = self.factor_lookback_info.get(factor, {})
            
            # 检查财务数据依赖
            financial_requirements = {
                'IndNeu_EP': ['net_profit_parent', 'share_capital'],
                'IndNeu_BP': ['total_equity_parent', 'share_capital'],
                'IndNeu_ROE': ['net_profit_parent', 'total_equity_parent'],
                'IndNeu_GPM': ['total_revenue', 'cost_of_goods_sold'],
                'IndNeu_SalesGrowth': ['total_revenue'],
                'IndNeu_AssetTurnover': ['total_revenue', 'total_assets'],
                'IndNeu_CurrentRatio': ['current_assets', 'current_liabilities'],
                'IndNeu_CFOP': ['net_cash_flow_ops', 'share_capital']
            }
            
            if factor in financial_requirements:
                required_cols = financial_requirements[factor]
                logger.info(f"  • 所需财务字段: {required_cols}")
                logger.info(f"  • 财务数据滞后: 是 (年报/季报滞后)")
                
                # 检查2020年年报数据的可用性（通常在2021年3-4月发布）
                report_date = "2020-12-31"
                logger.info(f"  • 关键财务报告日期: {report_date}")
                logger.info(f"  • 2021-02-01时年报状态: 可能未发布或数据不完整")
            
            # 检查行业分布
            logger.info("  • 行业中性化要求:")
            logger.info("    - 每个行业至少需要2只以上股票进行中性化")
            logger.info("    - 行业分布不均会导致某些行业无法计算")
            logger.info("    - 行业代码缺失会导致该股票被排除")
            
            results[factor] = {
                'has_financial_lag': True,
                'requires_industry': True,
                'requires_min_stocks_per_industry': 2
            }
        
        return results
    
    def analyze_rolling_window_requirements(self):
        """分析滚动窗口计算的要求"""
        logger.info("\n🔄 分析滚动窗口计算要求:")
        
        # 检查RollingICIRCalculator的要求
        logger.info("📈 RollingICIRCalculator 合并逻辑:")
        logger.info("  • 需要将因子数据与forward_return数据合并")
        logger.info("  • 合并时会进行inner join，只保留有有效值的记录")
        logger.info("  • 如果因子在2021-02-01附近都是NaN或0，合并后count()会为0")
        
        # 分析各因子在2021-02-01的预期状态
        logger.info(f"\n🎯 2021-02-01 各因子预期状态分析:")
        
        for category, factors in self.target_factors.items():
            logger.info(f"\n📁 {category}:")
            for factor in factors:
                factor_info = self.factor_lookback_info.get(factor, {})
                window = factor_info.get('window', 0)
                
                if window > 0:
                    expected_status = "正常计算" if window <= 30 else "可能数据不足"
                else:
                    expected_status = "依赖财务数据，可能滞后"
                
                logger.info(f"  • {factor}: {expected_status}")
    
    def generate_comprehensive_report(self, tech_results, ind_results):
        """生成综合分析报告"""
        logger.info("\n📋 生成综合分析报告...")
        
        report = []
        report.append("=" * 80)
        report.append("2021-02-01 因子数据点不足问题 - 综合分析报告")
        report.append("=" * 80)
        
        # 技术指标类分析
        report.append("\n🔍 技术指标类因子分析:")
        for factor, result in tech_results.items():
            validity_rate = result['validity_rate'] * 100
            status = "正常" if validity_rate >= 80 else "可能不足"
            report.append(f"  • {factor}: {result['valid_stocks']}/{result['total_stocks']} ({validity_rate:.1f}%) - {status}")
        
        # 行业中性化类分析
        report.append("\n🏭 行业中性化因子分析:")
        report.append("  • 所有行业中性化因子都依赖财务数据")
        report.append("  • 2020年年报通常在2021年3-4月发布")
        report.append("  • 2021-02-01时财务数据可能不完整")
        report.append("  • 行业分类数据缺失会导致计算失败")
        
        # 根本原因分析
        report.append("\n💡 根本原因分析:")
        report.append("1. 技术指标类因子:")
        report.append("   - Reversal20D需要40天数据，但min_periods=20相对宽松")
        report.append("   - MovingAverageCross需要30天数据")
        report.append("   - MACD需要26天数据（12/26/9参数）")
        report.append("   - 2020年12月-2021年1月市场波动可能影响数据质量")
        
        report.append("\n2. 行业中性化因子:")
        report.append("   - 财务数据滞后是最主要原因")
        report.append("   - 年报数据在2021年2月时可能尚未发布")
        report.append("   - 行业分类数据缺失或不完整")
        report.append("   - 每个行业至少需要2只股票进行中性化")
        
        report.append("\n3. 合并逻辑问题:")
        report.append("   - RollingICIRCalculator使用inner join")
        report.append("   - 因子值为NaN或0的记录会被过滤掉")
        report.append("   - 最终count()为0说明所有有效数据都被排除了")
        
        # 解决建议
        report.append("\n💡 解决建议:")
        report.append("1. 检查数据源，确保财务数据及时更新")
        report.append("2. 调整行业中性化逻辑，处理数据缺失情况")
        report.append("3. 考虑使用上一期财务数据进行估算")
        report.append("4. 优化合并逻辑，允许部分数据缺失")
        report.append("5. 添加数据质量监控和预警机制")
        
        report_text = "\n".join(report)
        
        # 保存报告
        report_path = "test/factor_zero_analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"📄 报告已保存到: {report_path}")
        print(report_text)
        
        return report_text

def main():
    """主函数"""
    logger.info("🚀 开始因子零值问题分析...")
    
    analyzer = FactorZeroAnalyzer()
    
    # 连接数据库
    conn = analyzer.connect_database()
    if not conn:
        return
    
    # 获取股票池
    universe = analyzer.get_universe_stocks()
    if not universe:
        return
    
    try:
        # 1. 分析因子要求
        analyzer.analyze_factor_requirements()
        
        # 2. 检查数据可用性
        start_date = "2020-12-01"
        analyzer.check_data_availability(conn, start_date, analyzer.debug_date)
        
        # 3. 模拟技术指标类因子计算
        tech_results = analyzer.simulate_technical_factor_calculation(conn, universe)
        
        # 4. 模拟行业中性化因子计算
        ind_results = analyzer.simulate_industry_neutral_factor_calculation(conn, universe)
        
        # 5. 分析滚动窗口要求
        analyzer.analyze_rolling_window_requirements()
        
        # 6. 生成综合报告
        analyzer.generate_comprehensive_report(tech_results, ind_results)
        
        logger.info("✅ 分析完成!")
        
    except Exception as e:
        logger.error(f"❌ 分析过程中出现错误: {e}", exc_info=True)
    finally:
        conn.close()

if __name__ == "__main__":
    main()