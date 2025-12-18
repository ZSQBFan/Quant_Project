#!/usr/bin/env python3
"""
测试股票池逻辑

直接测试use_all_stocks开关对股票池获取的影响
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_universe_logic():
    """测试股票池逻辑"""
    print("🧪 测试股票池获取逻辑")
    print("=" * 60)
    
    try:
        from core.config import load_config
        
        config_loader = load_config()
        
        # 1. 测试普通模式 (use_all_stocks = false)
        print("\n📋 测试1: 普通模式股票池")
        
        # 临时修改配置
        import yaml
        config_path = "configs/data/config.yaml"
        
        # 读取当前配置
        with open(config_path, 'r', encoding='utf-8') as f:
            original_config = yaml.safe_load(f)
        
        # 设置为普通模式
        original_config['use_all_stocks'] = False
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(original_config, f, default_flow_style=False, allow_unicode=True)
        
        # 重新加载配置
        config_loader = load_config()
        universe_config = config_loader.load_universe()
        data_config = config_loader.load_data()
        
        print(f"   普通模式股票池数量: {len(universe_config)}")
        print(f"   use_all_stocks: {data_config.use_all_stocks}")
        print(f"   示例股票: {universe_config[:5] if universe_config else 'None'}")
        
        # 2. 测试全股票模式逻辑
        print("\n📋 测试2: 全股票模式逻辑")
        
        # 设置为全股票模式
        original_config['use_all_stocks'] = True
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(original_config, f, default_flow_style=False, allow_unicode=True)
        
        # 重新加载配置
        config_loader = load_config()
        data_config = config_loader.load_data()
        
        print(f"   全股票模式已启用: {data_config.use_all_stocks}")
        print(f"   数据提供者优先级: {data_config.provider_priority}")
        
        # 3. 模拟全股票模式获取逻辑
        print("\n📋 测试3: 模拟全股票获取逻辑")
        
        universe = []
        provider_used = None
        
        # 获取数据提供者配置
        providers_config = config_loader.load_providers()
        
        for provider_name in data_config.provider_priority:
            if provider_name in providers_config and providers_config[provider_name].enabled:
                provider_config = providers_config[provider_name]
                
                try:
                    # 动态导入数据提供者类
                    if provider_name == 'sqlite':
                        from data.providers import SQLiteDataProvider
                        provider_class = SQLiteDataProvider
                    elif provider_name == 'tushare':
                        from data.providers import TushareDataProvider
                        provider_class = TushareDataProvider
                    elif provider_name == 'akshare':
                        from data.providers import AkshareDataProvider
                        provider_class = AkshareDataProvider
                    else:
                        continue
                    
                    # 创建提供者实例
                    provider_kwargs = provider_config.config.copy()
                    
                    # 处理特殊配置
                    if provider_name == 'tushare' and 'token' not in provider_kwargs:
                        import os
                        provider_kwargs['token'] = os.getenv('TUSHARE_TOKEN')
                        if not provider_kwargs['token']:
                            continue
                    elif provider_name == 'sqlite':
                        # SQLite需要使用与之前测试相同的配置
                        provider_kwargs = {
                            'db_path': "./database/JY_database/sqlite/JY_database.sqlite",
                            'table_name': "JY_t_price_daily"
                        }
                    
                    provider_instance = provider_class(**provider_kwargs)
                    
                    # 智能选择目标日期
                    target_date = None
                    if provider_name in ['sqlite']:
                        target_date = "2024-01-01"  # 使用示例日期
                    
                    # 获取全部股票代码
                    print(f"   尝试使用 {provider_name} 获取股票列表...")
                    universe = provider_instance.get_all_symbols(target_date)
                    
                    if universe:
                        provider_used = provider_name
                        print(f"   ✅ 成功从 {provider_name} 获取到 {len(universe)} 只股票")
                        break
                    else:
                        print(f"   ❌ {provider_name} 未返回股票列表")
                        
                except Exception as e:
                    print(f"   ❌ {provider_name} 获取失败: {e}")
                    continue
        
        # 4. 对比结果
        print("\n" + "=" * 60)
        print("📊 对比结果:")
        
        print(f"   普通模式股票池: {len(universe_config)} 只")
        print(f"   全股票模式股票池: {len(universe)} 只")
        print(f"   差异: {len(universe) - len(universe_config)} 只")
        print(f"   使用的数据源: {provider_used}")
        
        if len(universe) > len(universe_config):
            print(f"   🎉 全股票模式成功获取更多股票!")
            print(f"   📈 股票数量增加了 {len(universe) / len(universe_config):.1f} 倍")
        elif len(universe) == len(universe_config):
            print(f"   ⚠️  股票数量相同，可能数据源返回的是相同列表")
        else:
            print(f"   ⚠️  全股票模式股票数量反而更少，需要检查实现")
        
        # 5. 恢复原始配置
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(original_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n✅ 配置已恢复")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🚀 开始测试股票池逻辑")
    test_universe_logic()
    print("\n🎉 测试完成")

if __name__ == "__main__":
    main()