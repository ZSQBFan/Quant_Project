"""
Akshare 数据提供者
"""

import pandas as pd
import akshare as ak
import time
import random
import logging
from typing import List, Optional

from .base import BaseDataProvider


class AkshareDataProvider(BaseDataProvider):
    """
    使用 Akshare 作为数据源的具体实现。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adjust = kwargs.get('adjust', "hfq")

    def fetch_data(self, symbol: str, start_date: str,
                   end_date: str) -> pd.DataFrame | None:
        logging.info(
            f"[Akshare] 正在获取 {symbol} 从 {start_date} 到 {end_date} 的数据..."
        )
        for attempt in range(self.retries):
            try:
                if symbol.startswith('sh') or symbol.startswith('sz'):
                    df_raw = ak.stock_zh_index_daily(symbol=symbol)
                else:
                    df_raw = ak.stock_zh_a_hist(
                        symbol=symbol,
                        start_date=start_date.replace('-', ''),
                        end_date=end_date.replace('-', ''),
                        adjust=self.adjust)

                if df_raw is None or df_raw.empty or '日期' not in df_raw.columns:
                    logging.warning(
                        f"[Akshare] 在 {start_date} - {end_date} 范围内未返回 '{symbol}' 的有效数据。"
                    )
                    return None

                ak_columns = [
                    '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅',
                    '涨跌额', '换手率'
                ]
                db_columns = [
                    'date', 'open', 'close', 'high', 'low', 'volume',
                    'turnover', 'amplitude', 'pct_change', 'price_change',
                    'turnover_rate'
                ]
                df_raw.rename(columns=dict(zip(ak_columns, db_columns)),
                              inplace=True)

                for col in db_columns:
                    if col not in df_raw.columns:
                        df_raw[col] = None

                df = df_raw[db_columns].copy()
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

                numeric_cols = [
                    'open', 'close', 'high', 'low', 'turnover', 'amplitude',
                    'pct_change', 'price_change', 'turnover_rate'
                ]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['volume'] = pd.to_numeric(
                    df['volume'], errors='coerce').fillna(0).astype(int)

                df_final = df.loc[start_date:end_date]
                if not df_final.empty:
                    logging.info(
                        f"[Akshare] 成功获取 {symbol} 的 {len(df_final)} 条数据。"
                    )
                    return df_final
                else:
                    return None

            except Exception as e:
                logging.error(
                    f"[Akshare] 获取 {symbol} 数据时出错 (尝试 {attempt + 1}/{self.retries}): {e}",
                    exc_info=True)
                if attempt < self.retries - 1:
                    logging.warning(f"将在 {self.delay} 秒后重试...")
                    time.sleep(self.delay + random.uniform(0, 1))
                else:
                    logging.error(
                        f"[Akshare] 已达到最大重试次数，放弃获取 {symbol} 的数据。"
                    )
                    return None
        return None

    def get_all_symbols(self, target_date: Optional[str] = None) -> List[str]:
        """
        获取全部股票代码列表（Akshare实现）

        Args:
            target_date: 目标日期（AKShare不支持历史查询，将使用当前日期）

        Returns:
            股票代码列表

        Raises:
            NotImplementedError: 不支持此功能
        """
        if target_date is not None:
            logging.warning(
                f"[AKShare] 不支持按历史日期查询股票列表，将使用当前日期而非目标日期 {target_date}"
            )

        logging.info("[AKShare] 正在获取全部A股股票代码列表...")
        
        try:
            # 获取A股所有上市公司的代码和名称
            df_raw = ak.stock_info_a_code_name()
            
            if df_raw is None or df_raw.empty:
                logging.warning("[AKShare] 未能获取到股票列表")
                return []

            # 检查是否有股票代码列
            if 'code' not in df_raw.columns:
                logging.error("[AKShare] 返回数据中没有找到股票代码列")
                return []

            # 获取股票代码列表并去重
            symbols = df_raw['code'].dropna().unique().tolist()
            
            # 过滤掉无效的股票代码
            valid_symbols = []
            for symbol in symbols:
                if isinstance(symbol, str) and len(symbol) == 6 and symbol.isdigit():
                    valid_symbols.append(symbol)
            
            logging.info(f"[AKShare] 成功获取 {len(valid_symbols)} 只股票代码")
            return valid_symbols
            
        except Exception as e:
            logging.error(f"[AKShare] 获取全部股票代码失败: {e}", exc_info=True)
            return []
