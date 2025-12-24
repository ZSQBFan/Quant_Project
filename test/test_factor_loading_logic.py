"""
验证因子加载逻辑的单元测试
"""
import os
import pandas as pd
import numpy as np
import logging
from backtest.data.exporter import BTDataExporter

def test_factor_loading():
    # 1. 准备临时因子数据
    test_dir = 'temp/test_factor_load'
    os.makedirs(test_dir, exist_ok=True)
    
    date_str = '2023-01-03' # 周二
    file_path = os.path.join(test_dir, f"{date_str}.parquet")
    
    df = pd.DataFrame({
        'asset': ['000001.SZ', '000002.SZ'],
        'factor_value': [0.5, -0.5]
    })
    df.to_parquet(file_path)
    
    # 2. 初始化导出器
    exporter = BTDataExporter(data_manager=None, factor_data_dir=test_dir)
    
    # 3. 测试加载
    factor_series = exporter.load_factor_data('2023-01-01', '2023-01-05')
    
    print(f"Loaded factor series:\n{factor_series}")
    
    # 4. 验证
    assert not factor_series.empty
    assert len(factor_series) == 2
    assert factor_series.index.get_level_values('date')[0] == pd.to_datetime(date_str)
    
    # 5. 清理
    import shutil
    shutil.rmtree(test_dir)
    print("✅ Factor loading test passed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_factor_loading()
