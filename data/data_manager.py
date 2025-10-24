# data_manager.py
import pandas as pd
import backtrader as bt
from datetime import timedelta
import queue
import threading
from tqdm import tqdm
import logging

# 保持原有导入
from .database_handler import DatabaseHandler
from .trading_calendars import TushareTradingCalendar, AkshareTradingCalendar
from .data_providers import AkshareDataProvider, TushareDataProvider, SQLiteDataProvider


class DataProviderManager:
    """
    【生产者-消费者重构版】统一数据提供者管理器。

    本类采用生产者-消费者设计模式，高效地处理数据下载与存储任务：
    - **检查者(Checker)线程**: 快速检查本地数据库，确定需要下载的数据范围。
    - **下载者(Producer)线程**: 并行地从配置的数据源获取数据，但不直接写入数据库，
      而是将获取到的数据放入一个中央队列。
    - **写入者(Consumer)线程**: 单独一个线程，从中央队列中取出数据，合并成大批量后，
      一次性写入数据库。

    该设计旨在解决SQLite的并发写入瓶颈，通过批量写入大幅提升性能，并显著降低CPU负载。
    """

    def __init__(
            self,
            provider_configs,
            symbols,
            start_date,
            end_date,
            db_path='quant_data.db',
            num_checker_threads=4,
            num_downloader_threads=8,  # 下载线程现在作为“生产者”
            batch_size=100):
        """
        初始化数据管理器。

        Args:
            provider_configs (list): 数据源提供者的配置列表。
            symbols (list): 需要处理的股票代码列表。
            start_date (str): 数据开始日期。
            end_date (str): 数据结束日期。
            db_path (str, optional): 回测专用数据库的路径。默认为 'quant_data.db'。
            num_checker_threads (int, optional): 检查数据完整性的线程数。默认为 4。
            num_downloader_threads (int, optional): 下载数据的线程数（生产者）。默认为 8。
            batch_size (int, optional): 消费者一次性写入数据库的最大数据批次。默认为 100。
        """
        self.start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        self.end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
        self.provider_configs = provider_configs
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.db_handler = DatabaseHandler(db_path)
        self.table_name = 'stock_daily_prices'

        # --- 线程与队列配置 ---
        self.num_checker_threads = num_checker_threads
        self.num_downloader_threads = num_downloader_threads  # 保留命名，实际为生产者数量
        self.batch_size = batch_size

        # 用于检查者确定下载任务的队列
        self.symbols_queue = queue.Queue()
        # 用于存放待下载任务的队列 (由检查者填充，供下载者消费)
        self.download_tasks_queue = queue.Queue()
        # 用于存放已下载数据的中央队列 (由下载者填充，供写入者消费)
        self.results_queue = queue.Queue()
        # 用于通知消费者(写入者)所有生产者(下载者)已完成任务的事件
        self.producers_finished_event = threading.Event()

        # --- 进度条 ---
        self.check_progress_bar = None
        self.download_progress_bar = None

        # --- 交易日历 ---
        self.all_trade_dates_set = set()
        if self.provider_configs:
            if self.provider_configs[0][0].__name__ == 'TushareDataProvider':
                self.calendar_provider = TushareTradingCalendar(
                    token=self.provider_configs[0][1].get('token'))
            else:
                self.calendar_provider = AkshareTradingCalendar()
        else:
            # 如果没有配置在线数据源，也需要一个日历提供者用于后续的数据质量校验
            self.calendar_provider = AkshareTradingCalendar()

    def _find_missing_date_ranges(self, symbol: str) -> list[tuple[str, str]]:
        """检查单个标的，返回其缺失数据的日期区间列表。"""
        logging.debug(f"检查线程 {threading.get_ident()} 正在为 {symbol} 检查数据完整性")
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
            logging.info(f"✅ [{symbol}] 数据完整，无需下载。")
            return []

        logging.info(
            f"📥 [{symbol}] 发现 {len(missing_dates)} 个缺失的交易日，正在合并为下载区间...")
        ranges = []
        if not missing_dates:
            return ranges
        start_range = missing_dates[0]
        for i in range(1, len(missing_dates)):
            # 如果两个缺失日期间隔超过一周，就认为是不连续的区间
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
        """
        按顺序尝试所有配置的数据源来获取单个标的数据。
        此方法由生产者(下载者)线程调用，仅负责获取和返回数据，不保存。
        """
        for provider_class, params in self.provider_configs:
            provider_instance = provider_class(**params)
            fetched_df = provider_instance.fetch_data(symbol, start_date,
                                                      end_date)
            if fetched_df is not None and not fetched_df.empty:
                logging.info(
                    f"  -> [下载者] 从 {provider_instance.__class__.__name__} 成功获取 {symbol} 的 {len(fetched_df)} 条数据。"
                )
                # 在返回前，将股票代码加入DataFrame中，以便消费者识别
                fetched_df['code'] = symbol
                return fetched_df.reset_index()

        logging.warning(
            f"  ‼️ [下载者] 警告：尝试所有数据源后，仍未能获取到 {symbol} ({start_date} to {end_date}) 的数据。"
        )
        return None

    def _producer_worker(self):
        """
        生产者(下载者)线程的工作逻辑。
        不断从下载任务队列中取出任务，获取数据，然后将结果DataFrame放入中央结果队列。
        """
        while True:
            try:
                # 非阻塞地获取任务，如果队列为空则会立即引发 queue.Empty 异常
                symbol, missing_ranges = self.download_tasks_queue.get(
                    block=False)
            except queue.Empty:
                # 任务队列已空，此生产者线程可以结束工作
                break

            for start_date, end_date in missing_ranges:
                result_df = self._fetch_data_from_providers(
                    symbol, start_date, end_date)
                if result_df is not None:
                    # 将处理好的DataFrame放入结果队列
                    self.results_queue.put(result_df)

            # 标记此任务完成，用于主线程的 .join() 判断
            self.download_tasks_queue.task_done()
            if self.download_progress_bar:
                self.download_progress_bar.update(1)

    def _consumer_worker(self):
        """
        消费者(写入者)线程的工作逻辑。
        在单一线程中运行，不断从中央结果队列中获取数据，攒成一批后统一写入数据库，
        以避免并发写入冲突。
        """
        batch = []
        # 循环条件：只要“生产者尚未全部结束”或者“结果队列里还有东西”，就继续工作
        while not (self.producers_finished_event.is_set()
                   and self.results_queue.empty()):
            try:
                # 设置1秒超时，避免在生产者工作慢时永久阻塞。
                # 这也让循环可以周期性地检查上面的退出条件。
                result_df = self.results_queue.get(timeout=1)
                batch.append(result_df)

                # 当攒够一个批次时，就执行一次写入
                if len(batch) >= self.batch_size:
                    self._save_batch_to_db(batch)
                    batch = []  # 清空批次，准备下一批
            except queue.Empty:
                # 队列暂时为空是正常现象，消费者会继续循环等待，直到退出条件满足
                continue

        # 所有生产者都结束后，处理最后一批可能不满尺寸的数据
        if batch:
            self._save_batch_to_db(batch)
        logging.info("--- [写入者] 所有数据已处理完毕，写入线程退出。 ---")

    def _save_batch_to_db(self, batch: list):
        """辅助方法：合并多个DataFrame并调用数据库处理器进行保存。"""
        if not batch:
            return
        try:
            # 将列表中的所有DataFrame合并成一个大的DataFrame
            full_df = pd.concat(batch, ignore_index=True)
            logging.info(
                f"--- [写入者] 正在合并 {len(batch)} 个DataFrame ({len(full_df)} 行)，并存入数据库... ---"
            )
            self.db_handler.save_data(full_df, self.table_name)
        except Exception as e:
            logging.error(f"--- ❌ [写入者] 批量数据保存至数据库时出错: {e} ---")

    def _checker_worker(self):
        """检查者线程的工作逻辑。从股票池队列取股票，检查后将任务放入下载队列。"""
        while True:
            try:
                symbol = self.symbols_queue.get(block=False)
            except queue.Empty:
                break
            missing_ranges = self._find_missing_date_ranges(symbol)
            if missing_ranges:
                self.download_tasks_queue.put((symbol, missing_ranges))
            self.symbols_queue.task_done()
            if self.check_progress_bar:
                self.check_progress_bar.update(1)

    def prepare_data_for_universe(self):
        """
        准备所有数据的主入口方法，负责编排整个多线程流程。
        """
        if not self.provider_configs:
            logging.info("ℹ️  未配置数据源，跳过数据下载，仅使用本地数据库数据。")
            return

        logging.info("--- 开始数据准备流程 (生产者-消费者模式) ---")

        # 流程1: 获取交易日历，作为后续检查的基准
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
            logging.critical(f"  ❌ 致命错误：获取交易日历时出错: {e}，程序终止。")
            return

        # 流程2: 启动检查者线程，并行检查所有股票，填充下载任务队列
        for symbol in self.symbols:
            self.symbols_queue.put(symbol)
        logging.info("🔍 正在检查数据完整性...")
        with tqdm(total=len(self.symbols), desc="检查进度", ncols=100) as pbar:
            self.check_progress_bar = pbar
            checker_threads = []
            for _ in range(self.num_checker_threads):
                thread = threading.Thread(target=self._checker_worker)
                thread.start()
                checker_threads.append(thread)
            # 等待所有检查者完成工作
            for thread in checker_threads:
                thread.join()
        self.check_progress_bar = None

        total_downloads = self.download_tasks_queue.qsize()
        if total_downloads > 0:
            logging.info(
                f"📥 即将启动 {self.num_downloader_threads} 个下载线程(生产者) 和 1 个写入线程(消费者)..."
            )
        else:
            logging.info("✅ 所有数据均已完整，无需下载。")
            return

        # 流程3: 启动唯一的消费者(写入者)线程，它会立刻开始等待结果队列中的数据
        consumer_thread = threading.Thread(target=self._consumer_worker)
        consumer_thread.start()

        # 流程4: 启动多个生产者(下载者)线程，它们会开始处理下载任务
        with tqdm(total=total_downloads, desc="下载进度", ncols=100) as pbar:
            self.download_progress_bar = pbar
            producer_threads = []
            for _ in range(self.num_downloader_threads):
                thread = threading.Thread(target=self._producer_worker)
                thread.start()
                producer_threads.append(thread)

            # 等待所有生产者(下载者)完成它们的工作
            for thread in producer_threads:
                thread.join()

        # 流程5: 所有生产者已结束，设置事件，这是给消费者的信号：不会再有新数据了
        self.producers_finished_event.set()

        # 流程6: 等待消费者处理完所有剩余任务并最终退出
        consumer_thread.join()

        logging.info("--- 所有数据准备流程执行完毕 ---")

    # --- 以下为供外部调用的标准接口方法，保持不变 ---

    def get_bt_feed(self, symbol: str) -> bt.feeds.PandasData | None:
        """从数据库获取单个标的的数据，并包装成Backtrader的feed格式。"""
        df = self.get_dataframe(symbol)
        if df is not None and not df.empty:
            return bt.feeds.PandasData(dataname=df)
        logging.error(f"❌ 未能为 {symbol} 获取有效数据，无法创建Backtrader feed。")
        return None

    def get_dataframe(self, symbol: str) -> pd.DataFrame | None:
        """从数据库获取并返回单个标的的DataFrame。"""
        query = f"SELECT * FROM {self.table_name} WHERE code = ? AND date BETWEEN ? AND ?"
        params = (symbol, self.start_date, self.end_date)
        df = self.db_handler.query_data(query, params)
        if df is not None and not df.empty:
            df.sort_index(ascending=True, inplace=True)
        return df

    def validate_data_quality(self, symbol: str) -> bool:
        """严格校验单个标的数据质量，确保数据完整且有效。"""
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

    def __del__(self):
        """在对象销毁时，确保关闭数据库连接。"""
        self.db_handler.close_connection()
