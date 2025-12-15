# bt/data/exporter.py

"""
Backtrader 数据导出器

职责：
- 从 DataManager 获取原始 OHLCV 数据
- 合并 composite_factor_series（大因子）
- 处理停牌标记和缺失值
- 导出为 Backtrader 可用的 Parquet 格式

设计原则：
- DataManager 保持纯粹，只负责数据查询
- 所有 BT 特定的格式转换在此处理
"""

import os
import pandas as pd
import numpy as np
import logging


class BTDataExporter:
    """
    将因子分析数据导出为 Backtrader 可用格式
    """
    
    def __init__(self, data_manager, output_dir='./bt/data_export/'):
        """
        参数:
            data_manager: DataProviderManager 实例
            output_dir: 导出目录
        """
        self.dm = data_manager
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export(self, universe: list, start_date: str, end_date: str, 
               factor_series: pd.Series) -> list:
        """
        导出数据供 Backtrader 使用
        
        参数:
            universe: 股票代码列表
            start_date: 开始日期 (格式: 'YYYY-MM-DD')
            end_date: 结束日期 (格式: 'YYYY-MM-DD')
            factor_series: 合成后的大因子 Series
                          - 索引为 (date, asset) 的 MultiIndex
                          - 值为 combined_signal
        
        返回:
            成功导出的文件路径列表
        """
        logging.info(f"⚙️ [BTDataExporter] 开始导出 {len(universe)} 只股票的数据...")
        
        exported_files = []
        
        for asset in universe:
            logging.debug(f"  > 开始处理股票 {asset}...")
            try:
                result = self._export_single_asset(
                    asset, start_date, end_date, factor_series
                )
                if result:
                    exported_files.append(result)
                    logging.debug(f"  > ✅ {asset}: 导出成功")
                else:
                    logging.debug(f"  > ❌ {asset}: 导出返回 None")
            except Exception as e:
                logging.warning(f"⚠️ 导出 {asset} 失败: {e}")
                import traceback
                logging.debug(f"详细错误: {traceback.format_exc()}")
        
        logging.info(f"✅ [BTDataExporter] 成功导出 {len(exported_files)}/{len(universe)} 只股票")
        return exported_files
    
    def _export_single_asset(self, asset: str, start_date: str, end_date: str,
                             factor_series: pd.Series) -> str | None:
        """
        导出单只股票数据，确保日期连续
        
        核心功能：
        - 使用完整工作日索引确保所有交易日都存在
        - 正确标记停牌日期
        - 处理新股上市前的情况
        - 与因子数据正确合并
        
        参数:
            asset: 股票代码
            start_date: 开始日期 (格式: 'YYYY-MM-DD')
            end_date: 结束日期 (格式: 'YYYY-MM-DD')
            factor_series: 合成后的大因子 Series
            
        返回:
            成功导出的文件路径，失败返回 None
        """
        # 1. 从 DataManager 获取 OHLCV
        prices = self.dm.get_dataframe(
            asset,
            columns=['open', 'high', 'low', 'close', 'volume']
        )
        if prices is None or prices.empty:
            logging.debug(f" > {asset}: 无 OHLCV 数据")
            return None
            
        # 2. 确保索引是 DatetimeIndex
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index)
            
        # 3. 筛选到指定日期范围
        prices = prices.loc[start_date:end_date]
        if prices.empty:
            logging.debug(f" > {asset}: 指定日期范围内无数据")
            return None
            
        # 4. 创建完整的工作日索引（关键修复！）
        # 使用 B = 工作日频率，确保包含所有交易日
        full_index = pd.date_range(start_date, end_date, freq='B')
        
        # 5. 重建索引，使日期连续（缺失日变为 NaN）
        prices = prices.reindex(full_index)
        
        # 6. 标记原始缺失日期为停牌（在填充之前！）
        # 注意：停牌日标记应该在填充前记录，否则会被覆盖
        suspended_mask = prices['close'].isna()
        
        # 7. 前向填充 OHLCV（保持价格连续性）
        prices = prices.ffill()
        
        # 8. 后向填充（处理开头的 NaN，如新股上市前）
        prices = prices.bfill()
        
        # 9. 如果填充后仍有 NaN（整个区间都没数据），跳过
        if prices['close'].isna().any():
            logging.debug(f" > {asset}: 填充后仍有缺失，跳过")
            return None
            
        # 10. 添加停牌标记
        prices['suspended'] = suspended_mask
        
        # 11. 拼接因子数据（保持原逻辑）
        if factor_series is not None:
            try:
                # 检查该资产是否在因子数据中
                if asset in factor_series.index.get_level_values('asset').unique():
                    asset_factors = factor_series.xs(asset, level='asset')
                    prices = prices.join(
                        asset_factors.rename('combined_signal'),
                        how='left'
                    )
                else:
                    prices['combined_signal'] = np.nan
            except Exception as e:
                logging.debug(f" > {asset}: 因子数据合并失败 - {e}")
                prices['combined_signal'] = np.nan
        else:
            prices['combined_signal'] = np.nan
            
        # 12. 添加 openinterest 列（Backtrader 必需）
        prices['openinterest'] = 0
        
        # 13. 最终列顺序（Backtrader 标准格式）
        final_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'openinterest', 'combined_signal', 'suspended'
        ]
        prices = prices[final_columns]
        
        # 14. 导出到 Parquet 文件
        output_path = os.path.join(self.output_dir, f'{asset}.parquet')
        prices.to_parquet(output_path)
        
        # 统计停牌天数并记录日志
        suspended_days = suspended_mask.sum()
        logging.debug(f" > ✅ {asset}: {len(prices)} 行, 停牌 {suspended_days} 天")
        return output_path
    
    def clear_export_dir(self):
        """清空导出目录（用于重新导出）"""
        import shutil
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"✅ 已清空导出目录: {self.output_dir}")