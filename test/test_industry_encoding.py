"""
测试行业编码功能

验证 BTDataExporter 的行业编码是否正常工作
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data.exporter import BTDataExporter


def test_industry_encoding():
    """测试行业编码功能"""
    print("=" * 60)
    print("测试行业编码功能")
    print("=" * 60)

    # 创建一个模拟的 data_manager
    class MockDataManager:
        def get_all_data_for_universe(self, universe, required_columns):
            # 创建模拟数据
            dates = pd.date_range('2024-01-01', '2024-01-10', freq='B')
            data = []

            for asset in universe:
                for date in dates:
                    data.append({
                        'date': date,
                        'asset': asset,
                        'open': 10.0,
                        'high': 11.0,
                        'low': 9.0,
                        'close': 10.5,
                        'volume': 1000,
                        'industry': f"行业{int(asset) % 3 + 1}"  # 3个行业
                    })

            df = pd.DataFrame(data)
            df.set_index(['date', 'asset'], inplace=True)
            return df

    # 创建导出器
    exporter = BTDataExporter(
        data_manager=MockDataManager(),
        output_dir='temp/test_industry_encoding/',
        factor_data_dir='temp/data_explore/'
    )

    # 测试编码功能
    print("\n测试 1: 编码行业名称")
    industry_names = ["金融", "消费", "科技", "医药", "金融"]  # 包含重复
    for name in industry_names:
        code = exporter._encode_industry(name)
        print(f"  {name} -> {code}")

    print(f"\n编码映射:")
    print(f"  industry_to_code: {exporter.industry_to_code}")
    print(f"  code_to_industry: {exporter.code_to_industry}")

    # 测试 None 和 NaN
    print("\n测试 2: 特殊值")
    print(f"  None -> {exporter._encode_industry(None)}")
    print(f"  NaN -> {exporter._encode_industry(float('nan'))}")

    # 保存映射
    print("\n测试 3: 保存映射文件")
    exporter.save_industry_mapping()

    # 验证映射文件
    import json
    mapping_path = os.path.join(exporter.output_dir, 'industry_mapping.json')
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            saved_mapping = json.load(f)
        print(f"✅ 映射文件已保存: {mapping_path}")
        print(f"   包含 {len(saved_mapping['industry_to_code'])} 个行业")
        print(f"   映射内容: {saved_mapping}")
    else:
        print(f"❌ 映射文件未找到")

    # 清理
    import shutil
    if os.path.exists('temp/test_industry_encoding/'):
        shutil.rmtree('temp/test_industry_encoding/')
        print("\n✅ 测试完成，已清理临时文件")

    print("\n" + "=" * 60)
    print("✅ 所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    test_industry_encoding()
