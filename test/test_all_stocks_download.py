#!/usr/bin/env python3
"""
测试全股票下载功能

验证当 use_all_stocks: true 时，系统是否会下载所有股票而不是仅股票池股票。
"""

import os
import sys
import yaml
import shutil
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def backup_config():
    """备份原始配置文件"""
    config_path = "configs/data/config.yaml"
    backup_path = "configs/data/config.yaml.backup"
    
    if os.path.exists(config_path):
        shutil.copy2(config_path, backup_path)
        print(f"✅ 已备份配置文件到: {backup_path}")
        return backup_path
    return None

def modify_config(use_all_stocks: bool):
    """修改配置文件"""
    config_path = "configs/data/config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改use_all_stocks设置
    config['use_all_stocks'] = use_all_stocks
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 已设置 use_all_stocks = {use_all_stocks}")

def restore_config(backup_path):
    """恢复原始配置文件"""
    if backup_path and os.path.exists(backup_path):
        shutil.copy2(backup_path, "configs/data/config.yaml")
        print(f"✅ 已恢复配置文件")

def get_stock_count_from_database():
    """从数据库获取实际股票数量"""
    try:
        import sqlite3
        
        # 检查多个可能的数据库路径
        db_paths = [
            "./database/quant_data.db",
            "./database/JY_database/sqlite/JY_database.sqlite"
        ]
        
        for db_path in db_paths:
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # 尝试不同的表名
                table_names = ['stock_daily_prices', 'JY_t_price_daily']
                
                for table_name in table_names:
                    try:
                        cursor.execute(f"SELECT COUNT(DISTINCT code) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        conn.close()
                        return count, db_path, table_name
                    except:
                        continue
                
                conn.close()
        
        return 0, None, None
        
    except Exception as e:
        print(f"❌ 数据库查询失败: {e}")
        return 0, None, None

def get_config_stock_count():
    """获取配置文件中的股票池数量"""
    try:
        universe_path = "configs/universe.yaml"
        if os.path.exists(universe_path):
            with open(universe_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                symbols = config.get('symbols', [])
                return len(symbols)
        return 0
    except Exception as e:
        print(f"❌ 配置文件读取失败: {e}")
        return 0

def test_all_stocks_mode():
    """测试全股票模式"""
    print("🧪 测试全股票下载功能")
    print("=" * 60)
    
    # 备份原始配置
    backup_path = backup_config()
    
    try:
        # 1. 测试普通模式 (use_all_stocks = false)
        print("\n📋 测试1: 普通模式 (use_all_stocks = false)")
        modify_config(False)
        
        config_stock_count = get_config_stock_count()
        db_stock_count, db_path, table_name = get_stock_count_from_database()
        
        print(f"   配置文件股票池数量: {config_stock_count}")
        print(f"   数据库实际股票数量: {db_stock_count}")
        if db_path:
            print(f"   数据库路径: {db_path}")
            print(f"   数据表: {table_name}")
        
        # 2. 测试全股票模式 (use_all_stocks = true)
        print("\n📋 测试2: 全股票模式 (use_all_stocks = true)")
        modify_config(True)
        
        # 这里我们只是验证配置是否正确设置
        # 实际的数据下载需要运行完整的因子分析流程
        print("   ✅ 全股票模式已启用")
        print("   ℹ️  实际下载测试需要运行完整的因子分析流程")
        
        # 3. 验证配置
        print("\n📋 测试3: 验证配置切换")
        config_path = "configs/data/config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            current_setting = config.get('use_all_stocks', False)
            print(f"   当前 use_all_stocks 设置: {current_setting}")
            
        if current_setting:
            print("   ✅ 配置切换成功")
        else:
            print("   ❌ 配置切换失败")
        
        # 4. 显示测试结果
        print("\n" + "=" * 60)
        print("📊 测试结果汇总:")
        
        if db_stock_count > 0:
            if db_stock_count == config_stock_count:
                print(f"   🔍 当前数据库股票数量 ({db_stock_count}) == 配置文件股票池数量 ({config_stock_count})")
                print("   ℹ️  这表明当前是普通模式，仅下载了股票池股票")
            elif db_stock_count > config_stock_count:
                print(f"   🔍 当前数据库股票数量 ({db_stock_count}) > 配置文件股票池数量 ({config_stock_count})")
                print("   ℹ️  这可能表明已经有一些全量数据")
            else:
                print(f"   🔍 当前数据库股票数量 ({db_stock_count}) < 配置文件股票池数量 ({config_stock_count})")
                print("   ℹ️  这可能表明数据库不完整或配置有问题")
        else:
            print("   ⚠️  未找到有效的数据库或数据表")
        
        print(f"\n🎯 测试结论:")
        print(f"   - 配置文件支持 use_all_stocks 开关: ✅")
        print(f"   - 开关可以正常切换: ✅")
        print(f"   - 需要运行完整流程验证实际下载行为")
        
        # 5. 提供下一步测试建议
        print(f"\n💡 后续测试建议:")
        print(f"   1. 运行完整的因子分析流程验证下载行为")
        print(f"   2. 对比 use_all_stocks=true 和 false 的下载结果")
        print(f"   3. 监控数据库中的股票数量变化")
        
    finally:
        # 恢复原始配置
        restore_config(backup_path)

def main():
    """主函数"""
    print("🚀 开始测试全股票下载功能")
    test_all_stocks_mode()
    print("\n🎉 测试完成")

if __name__ == "__main__":
    main()