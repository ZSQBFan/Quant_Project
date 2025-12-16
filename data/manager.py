"""
统一数据管理器

负责数据的下载、清洗、入库，以及向因子计算器提供按需加载的数据。
"""

import pandas as pd
from datetime import timedelta
import queue
import threading
from tqdm import tqdm
import logging
import sqlite3
import os
import sys
import numpy as np

import yaml

from .handlers import DatabaseHandler
from .calendar import TushareTradingCalendar, AkshareTradingCalendar, create_trading_calendar
from .providers import AkshareDataProvider, TushareDataProvider, SQLiteDataProvider


def _calculate_forward_returns(df: pd.DataFrame, periods: list) -> pd.DataFrame:
    """
    计算单个资产的未来收益率。
    用于 calculate_universe_forward_returns 中的 apply 操作。
    """
    if hasattr(df, 'index'):
        logging.debug(f"_calculate_forward_returns 输入索引: {df.index.names}")

    df = df.sort_index()
    for p in periods:
        future_price = df['close'].shift(-p)
        df[f'forward_return_{p}d'] = (future_price / df['close']) - 1

    return df


class DataProviderManager:
    """
    统一数据管理器。
    负责数据的下载、清洗、入库，以及向因子计算器提供按需加载的数据。
    """

    def __init__(self,
                 provider_configs,
                 symbols,
                 start_date,
                 end_date,
                 db_path='quant_data.db',
                 num_checker_threads=4,
                 num_downloader_threads=8,
                 batch_size=100,
                 auto_detect_universe: bool = True):

        self.start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        self.end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
        self.provider_configs = provider_configs
        self.db_handler = DatabaseHandler(db_path)
        self.table_name = 'stock_daily_prices'

        # 多线程下载相关配置
        self.num_checker_threads = num_checker_threads
        self.num_downloader_threads = num_downloader_threads
        self.batch_size = batch_size
        self.symbols_queue = queue.Queue()
        self.download_tasks_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.producers_finished_event = threading.Event()
        self.check_progress_bar = None
        self.download_progress_bar = None

        # 数据源初始化
        self._local = threading.local()

        # 初始化交易日历（读取配置文件）
        self.calendar_provider = self._init_calendar_provider()

        # 股票池初始化
        if not symbols and auto_detect_universe:
            logging.info("'symbols' 为空。正在从数据库自动检测股票池...")
            try:
                query = f"SELECT DISTINCT code FROM {self.table_name}"
                df = self.db_handler.query_data(query)
                if df is not None and not df.empty:
                    self.symbols = df['code'].tolist()
                    logging.info(f"自动检测到 {len(self.symbols)} 只股票。")
                else:
                    logging.warning("数据库为空，且未提供 symbols 列表。")
                    self.symbols = []
            except Exception as e:
                logging.error(f"自动检测股票池失败: {e}")
                self.symbols = []
        else:
            self.symbols = symbols if symbols else []

        # 列名映射字典
        self.COLUMN_MAPPING = {
            # 日线行情表 (stock_daily_prices)
            'open': ('stock_daily_prices', 'open'),
            'high': ('stock_daily_prices', 'high'),
            'low': ('stock_daily_prices', 'low'),
            'close': ('stock_daily_prices', 'close'),
            'volume': ('stock_daily_prices', 'volume'),
            'turnover': ('stock_daily_prices', 'turnover'),
            'pct_change': ('stock_daily_prices', 'pct_change'),
            'turnover_rate': ('stock_daily_prices', 'turnover_rate'),
            'amplitude': ('stock_daily_prices', 'amplitude'),
            'price_change': ('stock_daily_prices', 'price_change'),

            # 行业/元数据表 (stock_kind)
            'industry': ('stock_kind', 'Nnindnme'),
            'industry_code': ('stock_kind', 'Nnindcd'),
            'stk_name': ('stock_kind', 'Stknme'),
            'list_date': ('stock_kind', 'Listdt'),
            'ownership': ('stock_kind', 'OWNERSHIPTYPE'),
            'market_type': ('stock_kind', 'Markettype'),
            'status': ('stock_kind', 'Statco'),

            # 资产负债表 (Stock_BalanceSheet)
            'total_equity_parent': ('Stock_BalanceSheet', 'A003100000'),
            'share_capital': ('Stock_BalanceSheet', 'A003101000'),
            'total_assets': ('Stock_BalanceSheet', 'A001000000'),
            'current_assets': ('Stock_BalanceSheet', 'A001100000'),
            'fixed_assets': ('Stock_BalanceSheet', 'A001212000'),
            'intangible_assets': ('Stock_BalanceSheet', 'A001218000'),
            'goodwill': ('Stock_BalanceSheet', 'A001220000'),
            'inventory': ('Stock_BalanceSheet', 'A001123000'),
            'accounts_receivable': ('Stock_BalanceSheet', 'A001111000'),
            'total_liabilities': ('Stock_BalanceSheet', 'A002000000'),
            'current_liabilities': ('Stock_BalanceSheet', 'A002100000'),

            # 利润表 (stock_ProfitSheet)
            'total_revenue': ('stock_ProfitSheet', 'B001100000'),
            'cost_of_goods_sold': ('stock_ProfitSheet', 'B001201000'),
            'operating_profit': ('stock_ProfitSheet', 'B001300000'),
            'total_profit': ('stock_ProfitSheet', 'B001000000'),
            'net_profit_parent': ('stock_ProfitSheet', 'B002000101'),
            'selling_expenses': ('stock_ProfitSheet', 'B001209000'),
            'admin_expenses': ('stock_ProfitSheet', 'B001210000'),
            'rd_finance_expenses': ('stock_ProfitSheet', 'B001216000'),
            'income_tax_expense': ('stock_ProfitSheet', 'B002100000'),

            # 现金流量表 (stock_CashFlowDirect)
            'net_cash_flow_ops': ('stock_CashFlowDirect', 'C001000000'),
            'net_cash_flow_inv': ('stock_CashFlowDirect', 'C002000000'),
            'capex': ('stock_CashFlowDirect', 'C002006000'),
            'net_cash_flow_fin': ('stock_CashFlowDirect', 'C003000000'),
            'borrowing_cash': ('stock_CashFlowDirect', 'C003002000'),
            'dividends_paid': ('stock_CashFlowDirect', 'C003005000'),
            'other_fin_payment': ('stock_CashFlowDirect', 'C003006000'),
        }

    def _init_calendar_provider(self):
        """
        根据配置文件初始化交易日历提供者。

        优先读取 configs/data/calendar.yaml，使用其中的 provider 配置。
        """
        calendar_config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'configs', 'data', 'calendar.yaml'
        )

        # 尝试从 provider_configs 获取 tushare token
        tushare_token = None
        for cls, kwargs in self.provider_configs:
            if cls.__name__ in ['TushareDataProvider', 'TushareTradingCalendar']:
                tushare_token = kwargs.get('token')
                break

        if not tushare_token:
            tushare_token = os.getenv('TUSHARE_TOKEN')

        # 尝试读取配置文件
        if os.path.exists(calendar_config_path):
            try:
                with open(calendar_config_path, 'r', encoding='utf-8') as f:
                    calendar_config = yaml.safe_load(f)

                provider_config = calendar_config.get('provider', {})
                return create_trading_calendar(provider_config, tushare_token)

            except Exception as e:
                logging.warning(f"读取日历配置文件失败: {e}，使用默认 Akshare")
        else:
            logging.info(f"日历配置文件不存在: {calendar_config_path}，使用默认 Akshare")

        # 默认使用 Akshare
        return AkshareTradingCalendar()

    def _get_provider(self, name):
        """获取线程本地的数据提供者实例。"""
        if not hasattr(self._local, 'providers'):
            self._local.providers = {}

        if name not in self._local.providers:
            for cls, kwargs in self.provider_configs:
                if cls.__name__ == name or (name == 'sqlite' and cls.__name__ == 'SQLiteDataProvider'):
                    self._local.providers[name] = cls(**kwargs)
                    break
        return self._local.providers.get(name)

    def get_dataframe(self, symbol: str, columns: list = None) -> pd.DataFrame | None:
        """
        从数据库获取单只股票的日线行情数据。

        Args:
            symbol: 股票代码
            columns: 需要的列名列表

        Returns:
            索引为 date 的 DataFrame
        """
        table_name = 'stock_daily_prices'

        if columns:
            query_cols = ['date']
            for col in columns:
                mapping = self.COLUMN_MAPPING.get(col)
                if mapping and mapping[0] == table_name:
                    query_cols.append(mapping[1])
                elif col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                    query_cols.append(col)

            query_cols = list(set(query_cols))
            cols_str = ", ".join(query_cols)
        else:
            cols_str = "*"

        query = f"SELECT {cols_str} FROM {table_name} WHERE code = ? AND date BETWEEN ? AND ?"
        params = (symbol, self.start_date, self.end_date)

        try:
            df = self.db_handler.query_data(query, params)
            if df is None or df.empty:
                return None

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

            return df
        except Exception as e:
            logging.error(f"获取 {symbol} 数据失败: {e}")
            return None

    def get_all_data_for_universe(self, universe: list,
                                  required_columns: list = None) -> pd.DataFrame:
        """
        获取全市场合并数据。

        Args:
            universe: 股票列表
            required_columns: 需要的列名列表

        Returns:
            MultiIndex (date, asset) 的 DataFrame
        """
        logging.info(f"[数据加载] 正在解析 {len(universe)} 只股票的数据需求...")

        table_col_map = {}
        load_industry = False

        if required_columns:
            for col in required_columns:
                if col == 'industry':
                    load_industry = True
                    continue
                mapping = self.COLUMN_MAPPING.get(col)
                if mapping:
                    table_name, db_col = mapping
                    if table_name not in table_col_map:
                        table_col_map[table_name] = []
                    table_col_map[table_name].append(col)
                elif col in ['open', 'high', 'low', 'close', 'volume']:
                    if 'stock_daily_prices' not in table_col_map:
                        table_col_map['stock_daily_prices'] = []
                    table_col_map['stock_daily_prices'].append(col)

        # 预加载行业数据
        industry_map = {}
        if load_industry:
            logging.info("正在预加载行业数据...")
            ind_query = "SELECT Stkcd, Nnindnme FROM stock_kind"
            ind_df = self.db_handler.query_data(ind_query)
            if ind_df is not None and not ind_df.empty:
                ind_df['Stkcd'] = ind_df['Stkcd'].astype(str).str.zfill(6)
                industry_map = ind_df.set_index('Stkcd')['Nnindnme'].to_dict()

        # 基本面数据预加载
        fundamental_cache = {}
        for table_name, cols in table_col_map.items():
            if table_name == 'stock_daily_prices':
                continue

            logging.info(f"[预加载] 正在全量读取 {table_name} ({cols})...")
            df_all = self._preload_fundamental_table(table_name, cols)

            if df_all is not None and not df_all.empty:
                grouped = df_all.groupby('Stkcd')
                fundamental_cache[table_name] = {
                    str(k).zfill(6): v for k, v in grouped
                }
                logging.info(f"已缓存 {len(fundamental_cache[table_name])} 只股票的 {table_name} 数据。")

        logging.info(f"开始批量合并数据...")
        return self._batch_merge_data(universe, table_col_map, fundamental_cache, industry_map, load_industry)

    def _batch_merge_data(self, universe: list, table_col_map: dict,
                          fundamental_cache: dict, industry_map: dict,
                          load_industry: bool) -> pd.DataFrame:
        """批量合并数据。"""
        logging.info(f"[批量查询] 正在为 {len(universe)} 只股票执行批量数据合并...")

        price_cols = table_col_map.get('stock_daily_prices', ['close'])
        symbols_str = "','".join([sym.replace("'", "''") for sym in universe])
        query_cols = ['code', 'date'] + [col for col in price_cols if col not in ['code', 'date']]
        cols_str = ", ".join(query_cols)

        batch_query = f"""
            SELECT {cols_str}
            FROM stock_daily_prices
            WHERE code IN ('{symbols_str}')
            AND date BETWEEN ? AND ?
            ORDER BY code, date
        """

        logging.info(f"执行批量行情数据查询 ({len(universe)} 只股票)...")
        all_price_df = self.db_handler.query_data(batch_query, (self.start_date, self.end_date))

        if all_price_df is None or all_price_df.empty:
            logging.warning("批量查询返回空数据")
            return pd.DataFrame()

        logging.info(f"批量查询完成，获得 {len(all_price_df)} 行数据")

        # 合并基本面数据
        if fundamental_cache:
            logging.info(f"开始批量合并基本面数据...")
            all_fund_dfs = []
            for table_name, fund_data_dict in fundamental_cache.items():
                for symbol, fund_df in fund_data_dict.items():
                    if symbol in universe:
                        fund_df_copy = fund_df.copy()
                        fund_df_copy['code'] = symbol
                        fund_df_copy = fund_df_copy.reset_index()
                        all_fund_dfs.append(fund_df_copy)

            if all_fund_dfs:
                combined_fund_df = pd.concat(all_fund_dfs, ignore_index=True)
                combined_fund_df['date'] = pd.to_datetime(combined_fund_df['date'])
                merge_cols = [col for col in combined_fund_df.columns if col not in ['code', 'date']]
                if merge_cols:
                    all_price_df = all_price_df.merge(
                        combined_fund_df,
                        on=['code', 'date'],
                        how='left',
                        suffixes=('', '_fund')
                    )

        # 添加行业数据
        if load_industry and industry_map:
            logging.info(f"向量化添加行业数据...")
            all_price_df['industry'] = all_price_df['code'].map(industry_map)

        # 重塑索引格式
        logging.info(f"重塑数据索引...")

        if 'date' not in all_price_df.columns:
            if all_price_df.index.name == 'date':
                all_price_df = all_price_df.reset_index()
            else:
                all_price_df['date'] = pd.Timestamp.now()

        all_price_df.rename(columns={'code': 'asset'}, inplace=True)

        if isinstance(all_price_df.index, pd.MultiIndex):
            all_price_df = all_price_df.reset_index()

        all_price_df.set_index(['date', 'asset'], inplace=True)
        all_price_df.sort_index(inplace=True)

        # 前向填充
        try:
            all_price_df = all_price_df.groupby(level=1, group_keys=False).apply(
                lambda x: x.ffill() if not x.empty else x
            )
        except Exception as e:
            logging.warning(f"前向填充失败: {e}")

        # 去除重复索引
        if isinstance(all_price_df.index, pd.MultiIndex):
            if all_price_df.index.duplicated().any():
                dup_count = all_price_df.index.duplicated().sum()
                logging.warning(f"[数据加载] 发现 {dup_count} 个重复索引，正在处理...")
                all_price_df = all_price_df[~all_price_df.index.duplicated(keep='last')]

        logging.info(f"批量合并完成！最终数据形状: {all_price_df.shape}")
        return all_price_df

    def _preload_fundamental_table(self, table_name: str,
                                   required_cols: list) -> pd.DataFrame | None:
        """全量读取一张基本面表。"""
        db_cols = []
        rename_map = {}
        for col in required_cols:
            mapping = self.COLUMN_MAPPING.get(col)
            if mapping:
                db_cols.append(mapping[1])
                rename_map[mapping[1]] = col

        if not db_cols:
            return None

        cols_str = ", ".join(['Stkcd', 'Accper'] + db_cols)
        query = f"SELECT {cols_str} FROM {table_name} ORDER BY Accper"

        try:
            df = self.db_handler.query_data(query)
            if df is None or df.empty:
                return None

            df['date'] = pd.to_datetime(df['Accper'])
            df.set_index('date', inplace=True)
            df.drop(columns=['Accper'], inplace=True, errors='ignore')
            df.rename(columns=rename_map, inplace=True)

            cols_to_numeric = list(rename_map.values())
            df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors='coerce')

            if not df.empty:
                df = df.reset_index()
                df = df.drop_duplicates(subset=['date', 'Stkcd'], keep='last')
                df = df.set_index('date')

            return df
        except Exception as e:
            logging.error(f"全量预加载失败 {table_name}: {e}")
            return None

    def calculate_universe_forward_returns(self, universe: list,
                                           forward_return_periods: list) -> pd.DataFrame:
        """统一计算未来收益率。"""
        logging.info(f"[收益率计算] 正在为 {len(universe)} 只股票计算未来收益...")

        all_data = self.get_all_data_for_universe(universe, required_columns=['close'])

        if all_data is None or all_data.empty:
            logging.error("无法加载数据用于计算收益率。")
            return None

        try:
            returns_df = all_data.groupby(level=1, group_keys=False).apply(
                lambda x: _calculate_forward_returns(x, forward_return_periods))
        except Exception as e:
            logging.error(f"计算未来收益率失败: {e}")
            return pd.DataFrame()

        returns_df.reset_index(inplace=True)
        cols_to_keep = ['date', 'asset'] + [f'forward_return_{p}d' for p in forward_return_periods]
        return returns_df[cols_to_keep]

    def get_industry_mapping(self) -> pd.DataFrame | None:
        """获取行业映射。"""
        logging.info("正在加载行业映射数据...")
        query = "SELECT Stkcd, Nnindnme FROM stock_kind"
        try:
            df = self.db_handler.query_data(query)
            if df is not None:
                df['asset'] = df['Stkcd'].astype(str).str.zfill(6)
                df.rename(columns={'Nnindnme': 'industry'}, inplace=True)
                return df[['asset', 'industry']]
            return None
        except Exception as e:
            logging.error(f"加载行业数据失败: {e}")
            return None

    def prepare_data_for_universe(self):
        """数据下载/更新流程。"""
        if not self.provider_configs:
            logging.info("未配置数据源，跳过下载。")
            return

        logging.info("--- 开始数据准备流程 ---")

        # 获取日历
        logging.info("正在获取交易日历...")
        try:
            all_trade_dates_str = self.calendar_provider.get_trading_days(
                self.start_date, self.end_date)
            self.all_trade_dates_set = set(pd.to_datetime(all_trade_dates_str).date)
        except Exception as e:
            logging.critical(f"获取交易日历失败: {e}")
            return

        # 填充检查队列
        for symbol in self.symbols:
            self.symbols_queue.put(symbol)

        # 启动检查线程
        logging.info(f"启动 {self.num_checker_threads} 个检查线程...")
        with tqdm(total=len(self.symbols), desc="[数据检查]", ncols=100, file=sys.stdout) as pbar:
            self.check_progress_bar = pbar
            threads = []
            for i in range(self.num_checker_threads):
                t = threading.Thread(target=self._checker_worker, name=f"Checker-{i}", daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

        self.check_progress_bar = None

        # 启动下载和写入
        total_downloads = self.download_tasks_queue.qsize()
        if total_downloads > 0:
            logging.info(f"共 {total_downloads} 只股票需要下载/更新。")
            logging.info(f"启动 {self.num_downloader_threads} 个下载线程和 1 个写入线程...")

            writer_thread = threading.Thread(target=self._consumer_worker, name="DB-Writer", daemon=True)
            writer_thread.start()

            with tqdm(total=total_downloads, desc="[数据下载]", ncols=100, file=sys.stdout) as pbar:
                self.download_progress_bar = pbar
                dl_threads = []
                for i in range(self.num_downloader_threads):
                    t = threading.Thread(target=self._producer_worker, name=f"Downloader-{i}", daemon=True)
                    t.start()
                    dl_threads.append(t)
                for t in dl_threads:
                    t.join()

            self.producers_finished_event.set()
            writer_thread.join()
            logging.info("--- 数据准备流程结束 ---")
        else:
            logging.info("所有数据已完整，无需下载。")

    def _checker_worker(self):
        """检查工作线程。"""
        while True:
            try:
                symbol = self.symbols_queue.get_nowait()
            except queue.Empty:
                break

            missing = self._find_missing_date_ranges(symbol)
            if missing:
                self.download_tasks_queue.put((symbol, missing))

            self.symbols_queue.task_done()
            if self.check_progress_bar:
                self.check_progress_bar.update(1)

    def _find_missing_date_ranges(self, symbol):
        """查找缺失的日期范围。"""
        query = f"SELECT date FROM {self.table_name} WHERE code = ?"
        df = self.db_handler.query_data(query, (symbol,))
        if df is None or df.empty:
            return [(pd.to_datetime(self.start_date).date(),
                     pd.to_datetime(self.end_date).date())]

        existing_dates = set(pd.to_datetime(df['date']).dt.date)
        missing_dates = sorted(list(self.all_trade_dates_set - existing_dates))

        if not missing_dates:
            return []

        return [(missing_dates[0], missing_dates[-1])]

    def _producer_worker(self):
        """下载工作线程。"""
        while True:
            try:
                task = self.download_tasks_queue.get_nowait()
            except queue.Empty:
                break

            symbol, ranges = task
            start, end = ranges[0]

            df = None
            provider_names = ['SQLiteDataProvider', 'TushareDataProvider', 'AkshareDataProvider']
            for provider_name in provider_names:
                provider = self._get_provider(provider_name)
                if provider:
                    try:
                        df = provider.fetch_data(symbol,
                                                 start.strftime('%Y-%m-%d'),
                                                 end.strftime('%Y-%m-%d'))
                        if df is not None and not df.empty:
                            df['code'] = symbol
                            break
                    except Exception as e:
                        logging.debug(f"[{provider_name}] 获取 {symbol} 数据失败: {e}")
                        continue

            if df is not None and not df.empty:
                self.results_queue.put(df)

            self.download_tasks_queue.task_done()
            if self.download_progress_bar:
                self.download_progress_bar.update(1)

    def _consumer_worker(self):
        """写入工作线程。"""
        batch = []
        while not (self.producers_finished_event.is_set() and self.results_queue.empty()):
            try:
                df = self.results_queue.get(timeout=1)
                batch.append(df)
                if len(batch) >= self.batch_size:
                    self._save_batch(batch)
                    batch = []
            except queue.Empty:
                continue
        if batch:
            self._save_batch(batch)

    def _save_batch(self, batch):
        """批量保存数据。"""
        if not batch:
            return
        try:
            # 调试日志：检查批量数据
            logging.debug(f"[批量保存] 准备合并 {len(batch)} 个DataFrame")
            for i, df in enumerate(batch):
                logging.debug(f"[批量保存] DataFrame {i}: 形状={df.shape}, 列={df.columns.tolist()}")
                if df.index.name == 'date':
                    logging.debug(f"[批量保存] DataFrame {i}: date是索引")
                elif 'date' in df.columns:
                    null_dates = df['date'].isnull().sum()
                    logging.debug(f"[批量保存] DataFrame {i}: date列中有 {null_dates} 个NULL值")
            
            # 修复：确保所有DataFrame都有date列
            processed_dfs = []
            for i, df in enumerate(batch):
                df_copy = df.copy()
                # 如果date是索引，将其重置为列
                if df_copy.index.name == 'date':
                    logging.debug(f"[批量保存] DataFrame {i}: 将date索引重置为列")
                    df_copy = df_copy.reset_index()
                # 如果索引是MultiIndex且包含date，也重置
                elif hasattr(df_copy.index, 'names') and 'date' in df_copy.index.names:
                    logging.debug(f"[批量保存] DataFrame {i}: 将MultiIndex中的date重置为列")
                    df_copy = df_copy.reset_index()
                # 确保date列存在
                if 'date' not in df_copy.columns:
                    logging.warning(f"[批量保存] DataFrame {i}: 处理后仍然没有date列，跳过")
                    continue
                processed_dfs.append(df_copy)
            
            if not processed_dfs:
                logging.error("[批量保存] 没有有效的DataFrame可保存")
                return
                
            full_df = pd.concat(processed_dfs, ignore_index=True)
            logging.debug(f"[批量保存] 合并后DataFrame形状: {full_df.shape}, 列: {full_df.columns.tolist()}")
            
            # 去重处理：移除重复的 (code, date) 组合
            before_dedup = len(full_df)
            full_df = full_df.drop_duplicates(subset=['code', 'date'], keep='last')
            after_dedup = len(full_df)
            if before_dedup != after_dedup:
                logging.info(f"[批量保存] 去重处理: 移除了 {before_dedup - after_dedup} 条重复数据")
            
            self.db_handler.save_data(full_df, self.table_name)
        except Exception as e:
            logging.error(f"批量写入失败: {e}")

    def __del__(self):
        if hasattr(self, 'db_handler'):
            self.db_handler.close_connection()
