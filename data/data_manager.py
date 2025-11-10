# data_manager.py (已优化 get_all_data_for_universe)
import pandas as pd
import backtrader as bt
from datetime import timedelta
import queue
import threading
from tqdm import tqdm
import logging
import sqlite3
import os
import sys
import numpy as np  # <- 【【【新增】】】 (用于分块)

from .database_handler import DatabaseHandler
from .trading_calendars import TushareTradingCalendar, AkshareTradingCalendar
from .data_providers import AkshareDataProvider, TushareDataProvider, SQLiteDataProvider


class DataProviderManager:
    """
    【生产者-消费者重构版】统一数据提供者管理器。
    
    【【重构日志】】:
    - 2025-11-10 (性能优化):
      - 优化 'get_all_data_for_universe'：
        - 移除 N 次查询的循环。
        - 替换为【分块查询】，以避免 SQLite "too many SQL variables"
          (限制~999) 的错误。
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

        # (已修正：仅在 auto_detect_universe=True 时才加载全市场)
        if not symbols and auto_detect_universe:
            logging.info("ℹ️ 'symbols' 列表为空。将从源数据库自动检测全市场股票池...")
            try:
                if not provider_configs:
                    raise ValueError("provider_configs 为空, 无法自动检测股票池。")

                source_provider_config = self.provider_configs[0][1]
                source_db_path = source_provider_config.get('db_path')
                source_table_name = source_provider_config.get('table_name')

                if not source_db_path or not source_table_name:
                    raise ValueError(
                        "在 provider_configs[0] 中未找到 'db_path' 或 'table_name'")

                logging.info(f"  > 正在连接源数据库: {source_db_path}")

                conn = sqlite3.connect(source_db_path)
                query = f"SELECT DISTINCT ticker FROM {source_table_name}"
                all_tickers_df = pd.read_sql(query, conn)
                conn.close()

                self.symbols = [
                    str(ticker).zfill(6) for ticker in all_tickers_df['ticker']
                ]

                if not self.symbols:
                    raise Exception("未能从数据库加载股票列表 (查询结果为空)。")

                logging.info(f"  > ✅ 成功加载 {len(self.symbols)} 只股票作为全市场股票池。")

            except Exception as e:
                logging.error(f"  > ❌ 动态获取股票池失败: {e}", exc_info=True)
                self.symbols = []

        elif not symbols and not auto_detect_universe:
            logging.debug("  > ℹ️ DataProviderManager (Worker) 已初始化 (无股票池)。")
            self.symbols = []

        else:
            logging.info(f"  > ℹ️ 正在使用传入的 {len(symbols)} 只股票的静态股票池。")
            self.symbols = symbols if isinstance(symbols, list) else [symbols]

        # (省略... 线程/队列/日历 初始化 ...)
        self.num_checker_threads = num_checker_threads
        self.num_downloader_threads = num_downloader_threads
        self.batch_size = batch_size
        self.symbols_queue = queue.Queue()
        self.download_tasks_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.producers_finished_event = threading.Event()
        self.check_progress_bar = None
        self.download_progress_bar = None
        if self.provider_configs:
            if self.provider_configs[0][0].__name__ == 'TushareDataProvider':
                self.calendar_provider = TushareTradingCalendar(
                    token=self.provider_configs[0][1].get('token'))
            else:
                self.calendar_provider = AkshareTradingCalendar()
        else:
            self.calendar_provider = AkshareTradingCalendar()

    # (省略... _find_missing_date_ranges, _fetch_data_from_providers ...)
    # (省略... _producer_worker, _consumer_worker, _save_batch_to_db, _checker_worker ...)
    # (省略... prepare_data_for_universe ...)
    # (省略... get_bt_feed, get_dataframe, validate_data_quality, get_industry_mapping ...)
    #
    # (为保持清晰，仅粘贴被修改和必须的函数)
    #

    def get_dataframe(self, symbol: str) -> pd.DataFrame | None:
        """从数据库获取并返回单个标的的DataFrame。"""
        query = f"SELECT * FROM {self.table_name} WHERE code = ? AND date BETWEEN ? AND ?"
        params = (symbol, self.start_date, self.end_date)
        df = self.db_handler.query_data(query, params)
        if df is not None and not df.empty:
            df.sort_index(ascending=True, inplace=True)
        return df

    # ==============================================================================
    # 【【【【【【 核心修改：get_all_data_for_universe 】】】】】】
    # ==============================================================================

    def get_all_data_for_universe(self, universe: list) -> pd.DataFrame | None:
        """
        获取股票池中所有股票的所有日线数据，并合并为一个大的 MultiIndex DataFrame。
        
        【【重构日志】】:
        - 2025-11-10 (性能优化 - 方案C):
          - 移除了 N 次查询的循环。
          - 替换为【分块查询】。为避免 SQLite "too many SQL variables" 错误
            (限制~999), 我们将 5000+ 的股票池 分块 (e.g., 900/块) 
            并执行 N/900 次查询。
        """
        if not universe:
            logging.warning(
                "  > ⚠️ [get_all_data_for_universe] 传入的 universe 列表为空，无法加载数据。")
            return None

        logging.info(
            f"--- ⚙️ 正在为 {len(universe)} 只股票加载【全部】日线数据 (执行分块SQL查询)... ---")

        all_stock_dfs = []  # 用于收集所有分块的 DataFrame

        # (SQLite 变量上限通常是 999，我们使用 900 作为安全值)
        SQLITE_VAR_LIMIT = 900

        num_chunks = int(np.ceil(len(universe) / SQLITE_VAR_LIMIT))

        # 【【【新增】】】: 使用 TQDM 包裹分块循环
        tqdm_loop = tqdm(
            range(num_chunks),
            desc="[数据加载] 分块加载股票数据",
            ncols=100,
            file=sys.stdout  # (保持与 logger_config.py 一致)
        )

        try:
            for i in tqdm_loop:
                # 1. 获取当前分块的股票
                start_idx = i * SQLITE_VAR_LIMIT
                end_idx = (i + 1) * SQLITE_VAR_LIMIT
                chunk_universe = universe[start_idx:end_idx]

                if not chunk_universe:
                    continue

                tqdm_loop.set_description(
                    f"[数据加载] 分块 {i+1}/{num_chunks} (含 {len(chunk_universe)} 只股票)"
                )

                # 2. 准备 SQL 查询
                placeholders = ', '.join('?' for _ in chunk_universe)
                query = f"""
                    SELECT * FROM {self.table_name} 
                    WHERE code IN ({placeholders}) 
                    AND date BETWEEN ? AND ?
                """

                # 3. 准备参数
                params = tuple(chunk_universe) + (self.start_date,
                                                  self.end_date)

                # 4. 执行【分块】查询
                # (db_handler.query_data 返回以 'date' 为索引的 DF)
                chunk_df = self.db_handler.query_data(query, params=params)

                if chunk_df is not None and not chunk_df.empty:
                    all_stock_dfs.append(chunk_df)

            if not all_stock_dfs:
                logging.error(f"  > ❌ 错误: 未能为股票池加载任何日线数据 (所有分块查询均为空)。")
                return None

            # 5. 合并所有分块
            logging.info("  > ⚙️ 正在合并所有数据分块...")
            full_df = pd.concat(all_stock_dfs)

            # 6. 转换为 MultiIndex (date, asset)
            full_df.rename(columns={'code': 'asset'}, inplace=True)
            full_df.reset_index(inplace=True)  # 释放 'date'

            full_df['date'] = pd.to_datetime(full_df['date'])
            full_df.set_index(['date', 'asset'], inplace=True)
            full_df.sort_index(inplace=True)

            logging.info(f"  > ✅ 成功加载并合并 {len(full_df)} 行总数据。")
            return full_df

        except Exception as e:
            logging.error(f"  > ❌ [get_all_data_for_universe] 执行分块查询时出错: {e}",
                          exc_info=True)
            return None

    # ==============================================================================
    # 【【【【【【 修改结束 】】】】】】
    # ==============================================================================

    def __del__(self):
        """在对象销毁时，确保关闭数据库连接。"""
        self.db_handler.close_connection()

    #
    # (为了让这个文件可以被完整替换，我把其他函数也粘贴在下面)
    #

    def _find_missing_date_ranges(self, symbol: str) -> list[tuple[str, str]]:
        logging.debug(
            f"  > [检查线程 {threading.get_ident()}] 正在为 {symbol} 检查数据完整性")
        all_trade_dates = self.all_trade_dates_set
        query = f"SELECT DISTINCT DATE(date) FROM {self.table_name} WHERE code = ? AND DATE(date) BETWEEN ? AND ?"
        existing_dates_df = self.db_handler.query_data(query,
                                                       params=(symbol,
                                                               self.start_date,
                                                               self.end_date))
        if existing_dates_df is not None and not existing_dates_df.empty:
            date_col = existing_dates_df.columns[0]
            existing_dates = set(
                pd.to_datetime(existing_dates_df[date_col]).dt.date)
        else:
            existing_dates = set()
        missing_dates = sorted(list(all_trade_dates - existing_dates))

        if not missing_dates:
            logging.info(f"  > ✅ [{symbol}] 数据完整，无需下载。")
            return []

        logging.info(
            f"  > 📥 [{symbol}] 发现 {len(missing_dates)} 个缺失的交易日，正在合并为下载区间...")
        ranges = []
        if not missing_dates:
            return ranges
        start_range = missing_dates[0]
        for i in range(1, len(missing_dates)):
            if (missing_dates[i] - missing_dates[i - 1]).days > 7:
                end_range = missing_dates[i - 1]
                ranges.append((start_range.strftime('%Y-%m-%d'),
                               end_range.strftime('%Y-%m-%d')))
                start_range = missing_dates[i]
        ranges.append((start_range.strftime('%Y-%m-%d'),
                       missing_dates[-1].strftime('%Y-%m-%d')))
        for start, end in ranges:
            logging.debug(f"    -> [{symbol}] 需要下载区间: {start} to {end}")
        return ranges

    def _fetch_data_from_providers(self, symbol: str, start_date: str,
                                   end_date: str) -> pd.DataFrame | None:
        for provider_class, params in self.provider_configs:
            provider_instance = provider_class(**params)
            fetched_df = provider_instance.fetch_data(symbol, start_date,
                                                      end_date)
            if fetched_df is not None and not fetched_df.empty:
                logging.info(
                    f"  > ✅ [下载者 {threading.get_ident()}] 从 {provider_instance.__class__.__name__} 成功获取 {symbol} 的 {len(fetched_df)} 条数据。"
                )
                fetched_df['code'] = symbol
                return fetched_df.reset_index()
        logging.warning(
            f"  > ⚠️ [下载者 {threading.get_ident()}] 警告：尝试所有数据源后，仍未能获取到 {symbol} ({start_date} to {end_date}) 的数据。"
        )
        return None

    def _producer_worker(self):
        while True:
            try:
                symbol, missing_ranges = self.download_tasks_queue.get(
                    block=False)
            except queue.Empty:
                logging.debug(f"  > [下载者 {threading.get_ident()}] 任务队列已空，退出。")
                break
            for start_date, end_date in missing_ranges:
                result_df = self._fetch_data_from_providers(
                    symbol, start_date, end_date)
                if result_df is not None:
                    self.results_queue.put(result_df)
            self.download_tasks_queue.task_done()
            if self.download_progress_bar:
                self.download_progress_bar.update(1)

    def _consumer_worker(self):
        batch = []
        while not (self.producers_finished_event.is_set()
                   and self.results_queue.empty()):
            try:
                result_df = self.results_queue.get(timeout=1)
                batch.append(result_df)
                if len(batch) >= self.batch_size:
                    self._save_batch_to_db(batch)
                    batch = []
            except queue.Empty:
                continue
        if batch:
            self._save_batch_to_db(batch)
        logging.info("--- [写入者] 所有数据已处理完毕，写入线程退出。 ---")

    def _save_batch_to_db(self, batch: list):
        if not batch:
            return
        try:
            full_df = pd.concat(batch, ignore_index=True)
            logging.info(
                f"--- [写入者] 正在合并 {len(batch)} 个DataFrame ({len(full_df)} 行)，并存入数据库... ---"
            )
            self.db_handler.save_data(full_df, self.table_name)
        except Exception as e:
            logging.error(f"--- ❌ [写入者] 批量数据保存至数据库时出错: {e} ---", exc_info=True)

    def _checker_worker(self):
        while True:
            try:
                symbol = self.symbols_queue.get(block=False)
            except queue.Empty:
                logging.debug(f"  > [检查线程 {threading.get_ident()}] 队列为空，退出。")
                break
            missing_ranges = self._find_missing_date_ranges(symbol)
            if missing_ranges:
                self.download_tasks_queue.put((symbol, missing_ranges))
            self.symbols_queue.task_done()
            if self.check_progress_bar:
                self.check_progress_bar.update(1)

    def prepare_data_for_universe(self):
        if not self.provider_configs:
            logging.info("ℹ️  未配置数据源，跳过数据下载，仅使用本地数据库数据。")
            return
        logging.info("--- 🏁 开始数据准备流程 (生产者-消费者模式) ---")
        logging.info("🗓️  正在获取交易日历...")
        try:
            all_trade_dates_str = self.calendar_provider.get_trading_days(
                self.start_date, self.end_date)
            if not all_trade_dates_str:
                logging.critical(f"  ❌ 致命错误：无法获取交易日历，程序终止。")
                return
            self.all_trade_dates_set = set(
                pd.to_datetime(all_trade_dates_str).date)
            logging.info(f"  ✅ 成功获取 {len(self.all_trade_dates_set)} 个交易日。")
        except Exception as e:
            logging.critical(f"  ❌ 致命错误：获取交易日历时出错: {e}，程序终止。", exc_info=True)
            return
        for symbol in self.symbols:
            self.symbols_queue.put(symbol)
        logging.info(
            f"🔍 正在启动 {self.num_checker_threads} 个检查线程，检查 {len(self.symbols)} 只股票..."
        )
        with tqdm(total=len(self.symbols),
                  desc="[数据检查] 检查进度",
                  ncols=100,
                  file=sys.stdout) as pbar:
            self.check_progress_bar = pbar
            checker_threads = []
            for i in range(self.num_checker_threads):
                thread = threading.Thread(target=self._checker_worker,
                                          name=f"Checker-{i}")
                thread.daemon = True
                thread.start()
                checker_threads.append(thread)
            for thread in checker_threads:
                thread.join()
        self.check_progress_bar = None
        total_downloads = self.download_tasks_queue.qsize()
        if total_downloads > 0:
            logging.info(f"📥 检查完毕。共 {total_downloads} 只股票需要下载数据。")
            logging.info(
                f"🚀 即将启动 {self.num_downloader_threads} 个下载线程(生产者) 和 1 个写入线程(消费者)..."
            )
        else:
            logging.info("✅ 检查完毕。所有数据均已完整，无需下载。")
            return
        consumer_thread = threading.Thread(target=self._consumer_worker,
                                           name="DB-Writer")
        consumer_thread.daemon = True
        consumer_thread.start()
        with tqdm(total=total_downloads,
                  desc="[数据下载] 下载进度",
                  ncols=100,
                  file=sys.stdout) as pbar:
            self.download_progress_bar = pbar
            producer_threads = []
            for i in range(self.num_downloader_threads):
                thread = threading.Thread(target=self._producer_worker,
                                          name=f"Downloader-{i}")
                thread.daemon = True
                thread.start()
                producer_threads.append(thread)
            for thread in producer_threads:
                thread.join()
        self.producers_finished_event.set()
        consumer_thread.join()
        logging.info("--- ✅ 所有数据准备流程执行完毕 ---")

    def get_bt_feed(self, symbol: str) -> bt.feeds.PandasData | None:
        df = self.get_dataframe(symbol)
        if df is not None and not df.empty:
            return bt.feeds.PandasData(dataname=df)
        logging.error(f"❌ 未能为 {symbol} 获取有效数据，无法创建Backtrader feed。")
        return None

    def validate_data_quality(self, symbol: str) -> bool:
        logging.info(f"--- 正在校验 '{symbol}' 的数据质量...")
        if not self.all_trade_dates_set:
            logging.error(f"🔴 交易日历数据未加载，无法对 '{symbol}' 进行完整性校验。")
            return False
        df = self.get_dataframe(symbol)
        if df is None or df.empty:
            logging.warning(
                f"🟡 '{symbol}' 在 {self.start_date} 到 {self.end_date} 期间无数据，已剔除。"
            )
            return False
        existing_dates = set(pd.to_datetime(df.index).date)
        missing_dates = self.all_trade_dates_set - existing_dates
        if missing_dates:
            example_missing = sorted(list(missing_dates))[:3]
            logging.warning(
                f"🔴 '{symbol}' 缺失 {len(missing_dates)} 个交易日的数据 (例如: {example_missing})，已剔除。"
            )
            return False
        cols_to_check = ['open', 'high', 'low', 'close', 'volume']
        if df[cols_to_check].isnull().values.any():
            problematic_rows = df[df[cols_to_check].isnull().any(axis=1)]
            logging.warning(
                f"🔴 '{symbol}' 的数据中包含空值 (NaN)，已剔出。问题数据快照:\n{problematic_rows.head(3)}"
            )
            return False
        if (df[cols_to_check] <= 0).any().any():
            problematic_rows = df[(df[cols_to_check] <= 0).any(axis=1)]
            logging.warning(
                f"🔴 '{symbol}' 的数据中包含0或负值，已剔出。问题数据快照:\n{problematic_rows.head(3)}"
            )
            return False
        logging.info(f"✅ '{symbol}' 数据质量校验通过。")
        return True

    def get_industry_mapping(self) -> pd.DataFrame | None:
        logging.info("  > ⚙️ 正在从 'stock_kind' 表加载行业映射数据...")
        try:
            query = "SELECT Stkcd, Nnindnme FROM stock_kind"
            df = self.db_handler.query_data(query)
            if df is None or df.empty:
                logging.warning("  > ⚠️ 警告: 未能从 'stock_kind' 表中加载到数据。")
                return None
            df['asset'] = df['Stkcd'].astype(str).str.zfill(6)
            df.rename(columns={'Nnindnme': 'industry'}, inplace=True)
            logging.info(f"  > ✅ 成功加载 {len(df)} 条行业映射记录。")
            return df[['asset', 'industry']]
        except Exception as e:
            logging.error(f"  > ❌ 加载 'stock_kind' 时出错: {e}", exc_info=True)
            return None
