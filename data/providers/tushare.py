"""
Tushare 数据提供者
"""

import pandas as pd
import tushare as ts
import time
import random
import logging
from typing import List, Optional
from datetime import datetime

from .base import BaseDataProvider


class TushareDataProvider(BaseDataProvider):
    """
    使用 Tushare 作为数据源的具体实现。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.token = kwargs.get('token')
        if not self.token:
            logging.critical("TushareDataProvider 需要一个有效的 'token' 参数。")
            raise ValueError("TushareDataProvider 需要一个有效的 'token' 参数。")
        self.pro = ts.pro_api(self.token)
        self.adjust = kwargs.get('adjust', "hfq")
        logging.info("TushareDataProvider 已初始化。")

    def _convert_symbol_to_ts_code(self, symbol):
        if symbol.startswith('sh') or symbol.startswith('sz'):
            return f"{symbol.replace('sh', '').replace('sz', '')}.SH" if symbol.startswith(
                'sh') else f"{symbol.replace('sh', '').replace('sz', '')}.SZ"
        return f"{symbol}.SH" if symbol.startswith('6') else f"{symbol}.SZ"

    def fetch_data(self, symbol: str, start_date: str,
                   end_date: str) -> pd.DataFrame | None:
        logging.info(
            f"[Tushare] 正在获取 {symbol} 从 {start_date} 到 {end_date} 的数据..."
        )
        ts_code = self._convert_symbol_to_ts_code(symbol)

        for attempt in range(self.retries):
            try:
                df_raw = self.pro.daily(ts_code=ts_code,
                                        start_date=start_date.replace('-', ''),
                                        end_date=end_date.replace('-', ''))

                if df_raw is None or df_raw.empty:
                    logging.warning(
                        f"[Tushare] 在 {start_date} - {end_date} 范围内未返回 '{symbol}' 的有效数据。"
                    )
                    return None

                adj_factor = self.pro.adj_factor(
                    ts_code=ts_code,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', ''))

                if not adj_factor.empty:
                    df_raw = pd.merge(df_raw,
                                      adj_factor[['trade_date', 'adj_factor']],
                                      on='trade_date',
                                      how='left')
                    df_raw['adj_factor'] = df_raw['adj_factor'].ffill()

                    for col in ['open', 'high', 'low', 'close']:
                        df_raw[col] = df_raw[col] * df_raw['adj_factor']

                ts_columns = [
                    'trade_date', 'open', 'close', 'high', 'low', 'vol',
                    'amount', 'pct_chg', 'change'
                ]
                db_columns = [
                    'date', 'open', 'close', 'high', 'low', 'volume',
                    'turnover', 'pct_change', 'price_change'
                ]
                df_raw.rename(columns=dict(zip(ts_columns, db_columns)),
                              inplace=True)

                df_raw['volume'] = pd.to_numeric(
                    df_raw['volume'],
                    errors='coerce').fillna(0).astype(int) * 100
                df_raw['turnover'] = pd.to_numeric(
                    df_raw['turnover'],
                    errors='coerce').fillna(0).astype(float) * 1000

                full_db_columns = [
                    'date', 'open', 'close', 'high', 'low', 'volume',
                    'turnover', 'amplitude', 'pct_change', 'price_change',
                    'turnover_rate'
                ]
                for col in full_db_columns:
                    if col not in df_raw.columns:
                        df_raw[col] = None

                df = df_raw[full_db_columns].copy()
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(ascending=True, inplace=True)

                logging.info(
                    f"[Tushare] 成功获取 {symbol} 的 {len(df)} 条数据。")
                return df

            except Exception as e:
                logging.error(
                    f"[Tushare] 获取 {symbol} 数据时出错 (尝试 {attempt + 1}/{self.retries}): {e}",
                    exc_info=True)
                if attempt < self.retries - 1:
                    logging.warning(f"将在 {self.delay} 秒后重试...")
                    time.sleep(self.delay + random.uniform(0, 1))
                else:
                    logging.error(
                        f"[Tushare] 已达到最大重试次数，放弃获取 {symbol} 的数据。"
                    )
                    return None
        return None

    def get_all_symbols(self, target_date: Optional[str] = None) -> List[str]:
        """
        获取全部股票代码列表（Tushare实现）

        Args:
            target_date: 目标日期，用于获取该时间点的股票列表
                        如果为None，使用当前日期

        Returns:
            股票代码列表
        """
        logging.info(f"[Tushare] 正在获取{target_date or '当前'}的全部股票代码列表...")
        
        try:
            # 如果没有指定目标日期，使用当前日期
            if target_date is None:
                target_date = datetime.now().strftime('%Y%m%d')
            else:
                # 将YYYY-MM-DD转换为YYYYMMDD格式
                target_date = target_date.replace('-', '')
            
            # 调用stock_basic接口获取股票基础信息
            logging.info(f"[Tushare] 调用stock_basic接口查询{target_date}的股票列表...")
            
            df_raw = self.pro.stock_basic(
                exchange='',
                list_status='L',  # 上市状态
                fields='ts_code,list_date,delist_date'
            )
            
            if df_raw is None or df_raw.empty:
                logging.warning("[Tushare] 未能获取到股票基础信息")
                return []

            # 筛选指定日期存在的股票
            # 在target_date之前上市，且在target_date之后（仍）未退市
            valid_stocks = df_raw[
                (df_raw['list_date'] <= target_date) &
                ((df_raw['delist_date'].isna()) | (df_raw['delist_date'] > target_date))
            ]
            
            # 提取股票代码（去掉交易所后缀）
            symbols = []
            for ts_code in valid_stocks['ts_code']:
                if isinstance(ts_code, str):
                    # 去掉后缀.SZ, .SH等
                    symbol = ts_code.split('.')[0]
                    if len(symbol) == 6 and symbol.isdigit():
                        symbols.append(symbol)
            
            logging.info(f"[Tushare] 成功获取 {len(symbols)} 只股票代码")
            return symbols
            
        except Exception as e:
            logging.error(f"[Tushare] 获取全部股票代码失败: {e}", exc_info=True)
            return []
