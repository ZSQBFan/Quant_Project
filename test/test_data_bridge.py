# test_data_bridge.py
import backtrader as bt
import pandas as pd
import numpy as np
import os
import sys

# 添加项目根目录到路径
# 修复：添加项目根目录而非test目录，确保能导入bt包
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir)
sys.path.insert(0, project_root)

from bt.data import BTDataExporter, FactorPandasData

print("=" * 60)
print("测试 Part 3: 数据桥梁")
print("=" * 60)

# 1. 测试导入
print("\n[1] 测试模块导入...")
try:
    from bt.data.exporter import BTDataExporter
    from bt.data.feeds import FactorPandasData
    print("✅ 模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    exit(1)

# 2. 检查导出目录
print("\n[2] 检查导出目录...")
export_dir = './test/bt_data_export_test/'
if os.path.exists(export_dir):
    files = [f for f in os.listdir(export_dir) if f.endswith('.parquet')]
    print(f"✅ 导出目录存在，包含 {len(files)} 个 Parquet 文件")
    
    if files:
        # 3. 检查示例文件结构
        print("\n[3] 检查示例文件结构...")
        sample_file = os.path.join(export_dir, files[0])
        df = pd.read_parquet(sample_file)
        
        print(f"  文件: {files[0]}")
        print(f"  行数: {len(df)}")
        print(f"  列名: {list(df.columns)}")
        
        # 验证必要列
        required_cols = ['open', 'high', 'low', 'close', 'volume', 
                        'combined_signal', 'suspended']
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if missing_cols:
            print(f"❌ 缺少列: {missing_cols}")
        else:
            print("✅ 所有必要列都存在")
        
        # 验证 combined_signal 不是全 0
        if 'combined_signal' in df.columns:
            non_nan_count = df['combined_signal'].notna().sum()
            print(f"  combined_signal 非空值数量: {non_nan_count}")
            if (df['combined_signal'] == 0).all():
                print("⚠️ combined_signal 全为 0，可能有问题")
        
        # 验证 suspended 列
        if 'suspended' in df.columns:
            suspended_count = df['suspended'].sum()
            print(f"  停牌天数: {suspended_count}")
        
        # 4. 测试 FactorPandasData 加载
        print("\n[4] 测试 FactorPandasData 加载...")
        try:
            import backtrader as bt
            data_feed = FactorPandasData(dataname=df, name='TEST')
            print("✅ FactorPandasData 实例化成功")
        except Exception as e:
            print(f"❌ FactorPandasData 实例化失败: {e}")
else:
    print("⚠️ 导出目录不存在，请先运行数据导出")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)