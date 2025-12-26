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
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from functools import partial


class BTDataExporter:
    """
    将因子分析数据导出为 Backtrader 可用格式
    """
    
    def __init__(self, data_manager, output_dir='./bt/data_export/', factor_data_dir='temp/data_explore/'):
        """
        参数:
            data_manager: DataProviderManager 实例
            output_dir: 导出目录
            factor_data_dir: 因子数据存储目录 (按天存储的 Parquet 文件)
        """
        self.dm = data_manager
        self.output_dir = output_dir
        self.factor_data_dir = factor_data_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _read_single_parquet(self, file_path: str, date_str: str) -> pd.DataFrame | None:
        """
        读取单个 Parquet 文件（用于并行读取）
        
        参数:
            file_path: 文件路径
            date_str: 日期字符串
            
        返回:
            DataFrame 或 None（失败时）
        """
        if not os.path.exists(file_path):
            return None
            
        try:
            df = pd.read_parquet(file_path)
            # 确保包含必要的列
            if 'asset' in df.columns and 'factor_value' in df.columns:
                df['date'] = pd.to_datetime(date_str)
                return df[['date', 'asset', 'factor_value']]
            else:
                logging.warning(f"⚠️ 文件 {file_path} 格式不正确，缺少 asset 或 factor_value 列")
                return None
        except Exception as e:
            logging.error(f"❌ 读取因子文件 {file_path} 失败: {e}")
            return None
    
    def load_factor_data(self, start_date: str, end_date: str) -> pd.Series:
        """
        从 factor_data_dir 加载按天存储的因子数据（并行读取优化版）
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            MultiIndex Series (date, asset) -> factor_value
        """
        logging.info(f"📂 [BTDataExporter] 从 {self.factor_data_dir} 并行加载因子数据 ({start_date} ~ {end_date})...")
        
        date_range = pd.date_range(start_date, end_date, freq='B')
        file_date_pairs = []
        
        for dt in date_range:
            date_str = dt.strftime('%Y-%m-%d')
            file_path = os.path.join(self.factor_data_dir, f"{date_str}.parquet")
            if os.path.exists(file_path):
                file_date_pairs.append((file_path, date_str))
        
        if not file_date_pairs:
            logging.warning(f"⚠️ 在指定范围内未找到任何因子数据: {start_date} ~ {end_date}")
            return pd.Series(dtype=float)
        
        # 使用 ThreadPoolExecutor 并行读取所有 Parquet 文件
        all_factors = []
        with ThreadPoolExecutor(max_workers=min(len(file_date_pairs), 16)) as executor:
            # 提交所有读取任务
            future_to_pair = {
                executor.submit(self._read_single_parquet, file_path, date_str): (file_path, date_str)
                for file_path, date_str in file_date_pairs
            }
            
            # 收集结果
            for future in as_completed(future_to_pair):
                result = future.result()
                if result is not None:
                    all_factors.append(result)
        
        if not all_factors:
            logging.warning(f"⚠️ 在指定范围内未找到任何因子数据: {start_date} ~ {end_date}")
            return pd.Series(dtype=float)
            
        combined_df = pd.concat(all_factors, ignore_index=True)
        combined_df.set_index(['date', 'asset'], inplace=True)
        
        logging.info(f"✅ 成功加载 {len(combined_df)} 条因子记录 (并行读取 {len(file_date_pairs)} 个文件)")
        return combined_df['factor_value']

    def _preprocess_factor_dict(self, factor_series: pd.Series, full_index: pd.DatetimeIndex) -> dict:
        """
        预处理因子数据为字典，显著加速单股票查找
        
        将 MultiIndex Series 转换为 {asset: np.array} 字典，
        避免每个线程重复进行 xs 查询和 reindex 操作
        
        参数:
            factor_series: MultiIndex Series (date, asset) -> factor_value
            full_index: 完整的日期索引（用于预先对齐）
            
        返回:
            {asset: factor_values_array} 字典
        """
        if factor_series is None or factor_series.empty:
            return {}
            
        factor_dict = {}
        
        # 获取所有唯一资产
        assets = factor_series.index.get_level_values('asset').unique()
        
        for asset in assets:
            try:
                # 使用 xs 获取单只股票的因子数据
                asset_factors = factor_series.xs(asset, level='asset')
                
                # 去除重复日期 - 保留第一个
                asset_factors = asset_factors[~asset_factors.index.duplicated(keep='first')]
                
                # 预先 reindex 到 full_index（主线程只做一次）
                if full_index is not None:
                    aligned_factors = asset_factors.reindex(full_index)
                else:
                    aligned_factors = asset_factors
                
                # 转换为 numpy 数组（更快）
                factor_dict[asset] = aligned_factors.values
            except Exception as e:
                logging.debug(f"预处理资产 {asset} 的因子数据失败: {e}")
                continue
        
        logging.info(f"✅ 因子数据预处理完成: {len(factor_dict)} 只股票有因子数据")
        return factor_dict
        
    def export(self, universe: list, start_date: str, end_date: str, 
               factor_series: pd.Series = None, max_workers: int = None) -> list:
        """
        导出数据供 Backtrader 使用（并行优化版）
        
        参数:
            universe: 股票代码列表
            start_date: 开始日期 (格式: 'YYYY-MM-DD')
            end_date: 结束日期 (格式: 'YYYY-MM-DD')
            factor_series: 合成后的大因子 Series (可选)
                          如果为 None，则尝试从 factor_data_dir 加载
            max_workers: 最大并行进程数，默认为CPU核心数
        
        返回:
            成功导出的文件路径列表
        """
        if factor_series is None:
            factor_series = self.load_factor_data(start_date, end_date)
            
        # 设置默认并行数：对于 I/O 密集型任务，线程数可以设得更高
        if max_workers is None:
            max_workers = min(mp.cpu_count() * 4, 64)  # 提升并发数加速 I/O
            
        logging.info(f"⚙️ [BTDataExporter] 开始并行导出 {len(universe)} 只股票的数据 (并发数: {max_workers})...")
        
        # 批量预加载所有数据（显著提升性能）
        logging.info("📊 预加载所有股票数据...")
        all_data_dict = self._batch_load_all_data(universe, start_date, end_date)
        
        exported_files = []
        failed_assets = []
        
        # 预创建工作日索引（避免每个线程重复创建）
        full_index = pd.date_range(start_date, end_date, freq='B')
        
        # 【关键优化】预处理因子数据为字典（主线程只做一次）
        factor_dict = self._preprocess_factor_dict(factor_series, full_index)
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务（传入预处理后的因子字典）
            future_to_asset = {
                executor.submit(
                    self._export_single_asset_optimized, 
                    asset, all_data_dict.get(asset), start_date, end_date, factor_dict.get(asset)
                ): asset 
                for asset in universe
            }
            
            # 使用进度条跟踪进度
            with tqdm(total=len(universe), desc="导出回测数据", unit="只") as pbar:
                for future in as_completed(future_to_asset):
                    asset = future_to_asset[future]
                    try:
                        result = future.result()
                        if result:
                            exported_files.append(result)
                            logging.debug(f"  > ✅ {asset}: 导出成功")
                        else:
                            failed_assets.append(asset)
                            logging.debug(f"  > ❌ {asset}: 导出返回 None")
                    except Exception as e:
                        failed_assets.append(asset)
                        logging.warning(f"⚠️ 导出 {asset} 失败: {e}")
                    finally:
                        pbar.update(1)
        
        logging.info(f"✅ [BTDataExporter] 并行导出完成! 成功: {len(exported_files)}/{len(universe)} 只股票")
        if failed_assets:
            logging.warning(f"⚠️ 失败股票: {len(failed_assets)} 只 - {failed_assets[:10]}{'...' if len(failed_assets) > 10 else ''}")
        
        return exported_files
    
    def _batch_load_all_data(self, universe: list, start_date: str, end_date: str) -> dict:
        """批量加载所有股票数据，显著提升性能"""
        logging.info("🔄 开始批量加载数据...")
        
        # 批量查询所有股票数据
        all_data = self.dm.get_all_data_for_universe(
            universe, 
            required_columns=['open', 'high', 'low', 'close', 'volume']
        )
        
        if all_data is None or all_data.empty:
            logging.warning("批量加载数据为空")
            return {}
            
        # 按股票分组
        data_dict = {}
        for asset in universe:
            try:
                asset_data = all_data.xs(asset, level='asset')
                if not asset_data.empty:
                    data_dict[asset] = asset_data
            except KeyError:
                logging.debug(f"股票 {asset} 无数据")
                continue
                
        logging.info(f"✅ 批量加载完成，成功加载 {len(data_dict)} 只股票数据")
        return data_dict
        
    def _export_single_asset_optimized(self, asset: str, asset_data: pd.DataFrame, 
                                      start_date: str, end_date: str,
                                      asset_factors: np.ndarray = None) -> str | None:
        """
        优化的单股票导出（使用预加载数据和预处理后的因子数据）
        
        参数:
            asset: 股票代码
            asset_data: 预加载的股票数据 DataFrame
            start_date: 开始日期
            end_date: 结束日期
            asset_factors: 预处理后的因子数据 numpy 数组（已对齐 full_index）
            
        返回:
            成功导出的文件路径，失败返回 None
        """
        if asset_data is None or asset_data.empty:
            return None
            
        # 1. 筛选日期范围
        try:
            asset_data = asset_data.loc[start_date:end_date]
        except KeyError:
            return None
        
        if asset_data.empty:
            return None
            
        # 2. 创建完整工作日索引（本地缓存）
        full_index = pd.date_range(start_date, end_date, freq='B')
        
        # 3. 重新索引
        asset_data = asset_data.reindex(full_index)
        
        # 4. 记录停牌标记（在填充前）
        suspended_mask = asset_data['close'].isna()
        
        # 5. 数据填充（链式操作避免 copy）
        close_values = asset_data['close'].ffill().bfill().values
        
        # 6. 检查数据完整性
        if np.isnan(close_values).any():
            return None
            
        # 7. 直接赋值（避免中间变量 copy）
        asset_data['close'] = close_values
        asset_data['open'] = asset_data['open'].ffill().bfill().values
        asset_data['high'] = asset_data['high'].ffill().bfill().values
        asset_data['low'] = asset_data['low'].ffill().bfill().values
        asset_data['volume'] = asset_data['volume'].ffill().bfill().values
        
        # 8. 添加停牌标记
        asset_data['suspended'] = suspended_mask.values
        
        # 9. 添加因子数据（预处理后的 numpy 数组直接赋值）
        if asset_factors is not None:
            asset_data['combined_signal'] = asset_factors
        else:
            asset_data['combined_signal'] = np.nan
        
        # 10. 添加 openinterest 列
        asset_data['openinterest'] = 0
        
        # 11. 最终列顺序
        asset_data = asset_data[['open', 'high', 'low', 'close', 'volume',
                                  'openinterest', 'combined_signal', 'suspended']]
        
        # 12. 导出到 Parquet 文件（优化写入参数，减小文件体积并提高写入速度）
        output_path = os.path.join(self.output_dir, f'{asset}.parquet')
        asset_data.to_parquet(output_path, engine='pyarrow', compression='snappy', use_dictionary=True)
        
        return output_path
        
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
        if factor_series is not None and not factor_series.empty:
            try:
                # 检查该资产是否在因子数据中
                # factor_series 索引为 (date, asset)
                if asset in factor_series.index.get_level_values('asset').unique():
                    asset_factors = factor_series.xs(asset, level='asset')
                    # 确保 asset_factors 的索引是 DatetimeIndex
                    if not isinstance(asset_factors.index, pd.DatetimeIndex):
                        asset_factors.index = pd.to_datetime(asset_factors.index)
                    
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