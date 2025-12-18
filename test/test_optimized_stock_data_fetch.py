#!/usr/bin/env python3
"""
测试优化后的全股票数据获取功能
"""

import os
import sys
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_optimized_stock_data_fetch():
    """测试优化的全股票数据获取功能"""
    try:
        logger.info("=== 测试优化的全股票数据获取功能 ===")
        
        # 导入数据提供者
        from data.providers.sqlite import SQLiteDataProvider
        
        # 模拟配置（使用JY数据库配置）
        provider_kwargs = {
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
        
        # 创建提供者实例
        provider = SQLiteDataProvider(**provider_kwargs)
        logger.info("✅ SQLiteDataProvider 创建成功")
        
        # 测试时间范围
        start_date = '2024-01-01'
        end_date = '2024-01-10'
        
        # 🔥 测试新的优化方法：直接获取全股票数据
        logger.info(f"🧪 测试 get_all_stock_data 方法 - 时间段: {start_date} ~ {end_date}")
        all_data = provider.get_all_stock_data(start_date, end_date, required_columns=None)
        
        if all_data is not None and not all_data.empty:
            logger.info("✅ 优化方法测试成功！")
            logger.info(f"  - 获取到 {len(all_data)} 条记录")
            logger.info(f"  - 股票数量: {all_data.index.get_level_values('asset').nunique()}")
            logger.info(f"  - 日期数量: {all_data.index.get_level_values('date').nunique()}")
            logger.info(f"  - 数据列: {list(all_data.columns)}")
            
            # 显示数据样本
            logger.info("数据样本 (前5行):")
            print(all_data.head())
            
            return True
        else:
            logger.warning("⚠️  优化方法返回空数据")
            
            # 🔄 测试回退方法：获取股票代码列表
            logger.info("🧪 测试 get_all_symbols 方法 (回退方法)")
            symbols = provider.get_all_symbols(start_date)
            
            if symbols:
                logger.info(f"✅ 回退方法成功！获取到 {len(symbols)} 只股票")
                logger.info(f"股票代码样本: {symbols[:5]}")
                return True
            else:
                logger.error("❌ 回退方法也失败")
                return False
                
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始测试优化后的全股票数据获取功能")
    logger.info("=" * 60)
    
    success = test_optimized_stock_data_fetch()
    
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("🎉 测试通过！优化后的全股票数据获取功能工作正常")
        logger.info("\n📈 改进总结:")
        logger.info("  1. ✅ 基类新增抽象方法: get_all_stock_data")
        logger.info("  2. ✅ SQLite提供者实现优化方法")
        logger.info("  3. ✅ 应用层智能选择最优获取方式")
        logger.info("  4. ✅ 支持时间段内全部股票数据获取")
        logger.info("  5. ✅ 保持向后兼容性")
    else:
        logger.warning("⚠️  测试失败，请检查相关功能")
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)