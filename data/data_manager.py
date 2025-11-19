# data/data_manager.py

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
import numpy as np

from .database_handler import DatabaseHandler
from .trading_calendars import TushareTradingCalendar, AkshareTradingCalendar
from .data_providers import AkshareDataProvider, TushareDataProvider, SQLiteDataProvider

# ==============================================================================
# 辅助函数
# ==============================================================================


def _calculate_forward_returns(df: pd.DataFrame,
                               periods: list) -> pd.DataFrame:
    """
    计算单个资产的未来收益率。
    用于 calculate_universe_forward_returns 中的 apply 操作。
    """
    # 确保按日期排序
    df = df.sort_index()
    for p in periods:
        # shift(-p) 将未来的价格向上平移，对齐到当前日期
        future_price = df['close'].shift(-p)
        df[f'forward_return_{p}d'] = (future_price / df['close']) - 1
    return df


# ==============================================================================
# 核心类: DataProviderManager
# ==============================================================================


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
        self.table_name = 'stock_daily_prices'  # 基础行情表名

        # --- [多线程下载相关配置] ---
        self.num_checker_threads = num_checker_threads
        self.num_downloader_threads = num_downloader_threads
        self.batch_size = batch_size
        self.symbols_queue = queue.Queue()
        self.download_tasks_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.producers_finished_event = threading.Event()
        self.check_progress_bar = None
        self.download_progress_bar = None

        # --- [数据源初始化] ---
        self._local = threading.local()

        # 初始化交易日历 (优先尝试 Tushare，失败则用 Akshare)
        try:
            self.calendar_provider = TushareTradingCalendar(self.db_handler)
        except:
            self.calendar_provider = AkshareTradingCalendar(self.db_handler)

        # --- [股票池初始化] ---
        if not symbols and auto_detect_universe:
            logging.info("ℹ️ 'symbols' 为空。正在从数据库自动检测股票池...")
            try:
                # 从 stock_daily_prices 表中获取所有去重的 code
                query = f"SELECT DISTINCT code FROM {self.table_name}"
                df = self.db_handler.query_data(query)
                if df is not None and not df.empty:
                    self.symbols = df['code'].tolist()
                    logging.info(f"✅ 自动检测到 {len(self.symbols)} 只股票。")
                else:
                    logging.warning("⚠️ 数据库为空，且未提供 symbols 列表。")
                    self.symbols = []
            except Exception as e:
                logging.error(f"❌ 自动检测股票池失败: {e}")
                self.symbols = []
        else:
            self.symbols = symbols if symbols else []

        # ======================================================================
        # 【【【核心配置：列名映射字典】】】
        # 用于告诉程序：当我们想要某个“因子所需的列”时，应该去哪个表、哪个字段找。
        # ======================================================================
        self.COLUMN_MAPPING = {
            # --- 1. 日线行情表 (stock_daily_prices) ---
            'open': ('stock_daily_prices', 'open'),
            'high': ('stock_daily_prices', 'high'),
            'low': ('stock_daily_prices', 'low'),
            'close': ('stock_daily_prices', 'close'),
            'volume': ('stock_daily_prices', 'volume'),
            'turnover': ('stock_daily_prices', 'turnover'),
            'pct_change': ('stock_daily_prices', 'pct_change'),
            'turnover_rate': ('stock_daily_prices', 'turnover_rate'),

            # --- 2. 行业/元数据表 (stock_kind) ---
            'industry': ('stock_kind', 'Nnindnme'),  # 行业名称
            'stk_name': ('stock_kind', 'Stknme'),  # 股票简称
            'list_date': ('stock_kind', 'Listdt'),  # 上市日期

            # --- 3. 资产负债表 (Stock_BalanceSheet) ---
            'total_equity_parent':
            ('Stock_BalanceSheet', 'A003100000'),  # 归母所有者权益 (B/P, ROE分母)
            'total_assets': ('Stock_BalanceSheet', 'A001000000'),  # 资产总计
            'total_liabilities': ('Stock_BalanceSheet', 'A002000000'),  # 负债合计
            'share_capital':
            ('Stock_BalanceSheet', 'A003101000'),  # 实收资本/股本 (计算市值)
            'current_assets': ('Stock_BalanceSheet', 'A001100000'),  # 流动资产
            'current_liabilities':
            ('Stock_BalanceSheet', 'A002100000'),  # 流动负债
            'inventory': ('Stock_BalanceSheet', 'A001123000'),  # 存货净额
            'accounts_receivable':
            ('Stock_BalanceSheet', 'A001111000'),  # 应收账款净额
            'fixed_assets': ('Stock_BalanceSheet', 'A001212000'),  # 固定资产净额
            'intangible_assets': ('Stock_BalanceSheet',
                                  'A001218000'),  # 无形资产净额
            'goodwill': ('Stock_BalanceSheet', 'A001220000'),  # 商誉净额

            # --- 4. 利润表 (stock_ProfitSheet) ---
            'total_revenue': ('stock_ProfitSheet',
                              'B001100000'),  # 营业总收入 (成长因子)
            'cost_of_goods_sold': ('stock_ProfitSheet', 'B001201000'),  # 营业成本
            'operating_profit': ('stock_ProfitSheet', 'B001300000'),  # 营业利润
            'total_profit': ('stock_ProfitSheet', 'B001000000'),  # 利润总额
            'net_profit_parent': ('stock_ProfitSheet',
                                  'B002000101'),  # 归母净利润 (E/P, ROE分子)
            'income_tax_expense': ('stock_ProfitSheet', 'B002100000'),  # 所得税费用
            'selling_expenses': ('stock_ProfitSheet', 'B001209000'),  # 销售费用
            'admin_expenses': ('stock_ProfitSheet', 'B001210000'),  # 管理费用
            'rd_expenses': ('stock_ProfitSheet', 'B001216000'),  # 研发费用

            # --- 5. 现金流量表 (stock_CashFlowDirect) ---
            'net_cash_flow_ops': ('stock_CashFlowDirect',
                                  'C001000000'),  # 经营活动现金流净额 (CFO)
            'net_cash_flow_inv': ('stock_CashFlowDirect',
                                  'C002000000'),  # 投资活动现金流净额 (CFI)
            'net_cash_flow_fin': ('stock_CashFlowDirect',
                                  'C003000000'),  # 筹资活动现金流净额 (CFF)
            'capex': ('stock_CashFlowDirect',
                      'C002006000'),  # 购建长期资产支付 (CapEx)
            'dividends_paid': ('stock_CashFlowDirect',
                               'C003005000'),  # 分配股利/利息支付
        }

    def _get_provider(self, name):
        """获取线程本地的数据提供者实例 (保持线程安全)。"""
        if not hasattr(self._local, 'providers'):
            self._local.providers = {}

        # 简单的缓存机制
        if name not in self._local.providers:
            for cls, kwargs in self.provider_configs:
                # 匹配配置中的类名
                if cls.__name__ == name or (name == 'sqlite' and cls.__name__
                                            == 'SQLiteDataProvider'):
                    self._local.providers[name] = cls(**kwargs)
                    break
        return self._local.providers.get(name)

    # ==========================================================================
    # 【【【核心方法 1：按需获取单只股票行情】】】
    # ==========================================================================
    def get_dataframe(self,
                      symbol: str,
                      columns: list = None) -> pd.DataFrame | None:
        """
        从数据库获取单只股票的【日线行情数据】。
        支持列筛选，仅查询 stock_daily_prices 表。

        Args:
            symbol: 股票代码
            columns: 需要的列名列表 (例如 ['close', 'volume'])。如果不传则查所有。

        Returns:
            pd.DataFrame: 索引为 date 的数据框
        """
        table_name = 'stock_daily_prices'

        # 1. 构建 SQL 查询字段
        if columns:
            # 总是包含 date，因为它是索引
            query_cols = ['date']
            for col in columns:
                # 查表：只处理属于 stock_daily_prices 的列
                mapping = self.COLUMN_MAPPING.get(col)
                if mapping and mapping[0] == table_name:
                    query_cols.append(mapping[1])  # 添加原始列名

                # 保底逻辑：如果列名不在映射中，但看起来像基础行情，也尝试查询
                elif col in [
                        'open', 'high', 'low', 'close', 'volume', 'turnover'
                ]:
                    query_cols.append(col)

            # 去重并转为字符串
            query_cols = list(set(query_cols))
            cols_str = ", ".join(query_cols)
        else:
            # 如果未指定，默认查所有
            cols_str = "*"

        # 2. 构建 SQL 语句
        # 使用 BETWEEN 优化日期范围查询
        query = f"SELECT {cols_str} FROM {table_name} WHERE code = ? AND date BETWEEN ? AND ?"
        params = (symbol, self.start_date, self.end_date)

        try:
            # 执行查询
            df = self.db_handler.query_data(query, params)
            if df is None or df.empty:
                return None

            # 确保 date 是 datetime 类型并设为索引
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

            return df
        except Exception as e:
            logging.error(f"❌ 获取 {symbol} 数据失败: {e}")
            return None

    # ==========================================================================
    # 【【【核心方法 2：获取全市场合并数据】】】
    # ==========================================================================
    def get_all_data_for_universe(
            self,
            universe: list,
            required_columns: list = None) -> pd.DataFrame:
        """
        为整个股票池获取合并了所有所需数据(行情+行业+基本面)的大宽表。
        采用 Pandas-Native 方式：分别读取，内存合并。
        """
        # --- [控制台输出] 告知用户正在分析数据需求 ---
        logging.info(f"⚙️ [数据加载] 正在解析 {len(universe)} 只股票的数据需求...")

        # 1. 分析需求：我们需要加载哪些表的数据？
        load_industry = False

        if required_columns:
            # 检查是否请求了 'industry'
            if 'industry' in required_columns:
                load_industry = True

        # 2. 预加载静态数据 (优化：避免在循环中 N 次查询数据库)
        industry_map = {}
        if load_industry:
            # --- [控制台输出] 告知用户正在预加载行业数据 ---
            logging.info("  > 正在预加载行业数据 (stock_kind)...")

            # 查询 Stkcd (代码) 和 Nnindnme (行业名)
            ind_query = "SELECT Stkcd, Nnindnme FROM stock_kind"
            ind_df = self.db_handler.query_data(ind_query)

            if ind_df is not None and not ind_df.empty:
                # 清洗代码格式，确保与 universe 中的 symbol 格式一致 (如补零)
                # 假设 universe 中的 symbol 是 6 位数字符串
                ind_df['Stkcd'] = ind_df['Stkcd'].astype(str).str.zfill(6)
                # 转为字典: {'000001': '银行', ...}
                industry_map = ind_df.set_index('Stkcd')['Nnindnme'].to_dict()
                logging.info(f"  > ✅ 成功加载 {len(industry_map)} 条行业记录。")
            else:
                logging.warning("  > ⚠️ 警告: 未能加载到行业数据，'industry' 列将为空。")

        # 3. 循环获取每只股票的数据并组装
        all_dfs = []

        # --- [控制台输出] 开始主循环，显示进度条 ---
        logging.info(f"🚀 开始加载并合并数据 (按需加载列: {required_columns})...")

        # 过滤出只属于行情表的列，传给 get_dataframe
        # 这样避免把 'industry' 这种列传给 SQL 报错
        price_cols = []
        if required_columns:
            for c in required_columns:
                mapping = self.COLUMN_MAPPING.get(c)
                if mapping and mapping[0] == 'stock_daily_prices':
                    price_cols.append(c)
                elif c in ['open', 'high', 'low', 'close', 'volume']:  # 基础列保底
                    price_cols.append(c)

        for symbol in tqdm(universe, desc="[Data Load]"):
            # A. 获取基础行情 (已按需筛选列)
            df = self.get_dataframe(symbol, columns=price_cols)

            if df is None or df.empty:
                continue

            # B. 合并行业数据 (Pandas Native: 字典映射)
            if load_industry:
                # 使用 map 比 apply 更快
                # get(symbol) 获取该股票的行业，如果没有则为 None
                ind = industry_map.get(symbol)
                df['industry'] = ind

            # C. (未来) 合并基本面数据
            # 这里将是 pd.merge_asof 的位置，用于对齐财报日期

            # 添加 asset 列，用于构建 MultiIndex
            df['asset'] = symbol
            all_dfs.append(df)

        if not all_dfs:
            logging.error("❌ 未能加载任何数据。")
            return pd.DataFrame()

        # 4. 最终合并所有股票的数据
        # --- [控制台输出] 告知用户正在进行最终合并 ---
        logging.info("⚙️ 正在合并所有股票的数据框...")

        final_df = pd.concat(all_dfs)

        # 【【【修复点】】】: 之前这里有两行 set_index，导致了 KeyError
        # 现在的逻辑：
        # 1. concat 后，索引是 date，列有 asset (和其他数据)
        # 2. reset_index() -> 索引变成 0,1,2...，date 变回普通列
        # 3. set_index(['date', 'asset']) -> 建立多重索引

        final_df.reset_index(inplace=True)
        final_df.set_index(['date', 'asset'], inplace=True)

        # 排序 (这对 rolling 计算至关重要)
        final_df.sort_index(inplace=True)

        logging.info(f"✅ 数据加载完成，共 {len(final_df)} 行。")
        return final_df

    def calculate_universe_forward_returns(
            self, universe: list,
            forward_return_periods: list) -> pd.DataFrame:
        """
        统一计算未来收益率。
        优化：只加载 close 列。
        """
        logging.info(f"⚙️ [收益率计算] 正在为 {len(universe)} 只股票计算未来收益...")

        # 仅加载 close 列，极大提升速度
        all_data = self.get_all_data_for_universe(universe,
                                                  required_columns=['close'])

        if all_data is None or all_data.empty:
            logging.error("❌ 无法加载数据用于计算收益率。")
            return None

        # 使用 groupby().apply()
        # group_keys=False 防止索引层级增加
        returns_df = all_data.groupby(level='asset', group_keys=False).apply(
            lambda x: _calculate_forward_returns(x, forward_return_periods))

        # 筛选出需要的列 (date, asset, forward_return_*)
        # 由于 apply 后索引是 (date, asset)，我们需要 reset_index 来获得这两列
        returns_df.reset_index(inplace=True)

        cols_to_keep = ['date', 'asset'] + [
            f'forward_return_{p}d' for p in forward_return_periods
        ]
        return returns_df[cols_to_keep]

    # ==========================================================================
    # 获取行业映射 (兼容旧代码)
    # ==========================================================================
    def get_industry_mapping(self) -> pd.DataFrame | None:
        logging.info("  > ⚙️ 正在加载行业映射数据...")
        query = "SELECT Stkcd, Nnindnme FROM stock_kind"
        try:
            df = self.db_handler.query_data(query)
            if df is not None:
                df['asset'] = df['Stkcd'].astype(str).str.zfill(6)
                df.rename(columns={'Nnindnme': 'industry'}, inplace=True)
                return df[['asset', 'industry']]
            return None
        except Exception as e:
            logging.error(f"❌ 加载行业数据失败: {e}")
            return None

    # ==========================================================================
    # 下面是数据下载/更新流程 (保持原有逻辑)
    # ==========================================================================
    def prepare_data_for_universe(self):
        if not self.provider_configs:
            logging.info("ℹ️ 未配置数据源，跳过下载。")
            return

        logging.info("--- 🏁 开始数据准备流程 (生产者-消费者模式) ---")

        # 1. 获取日历
        logging.info("🗓️ 正在获取交易日历...")
        try:
            all_trade_dates_str = self.calendar_provider.get_trading_days(
                self.start_date, self.end_date)
            self.all_trade_dates_set = set(
                pd.to_datetime(all_trade_dates_str).date)
        except Exception as e:
            logging.critical(f"❌ 获取交易日历失败: {e}")
            return

        # 2. 填充检查队列
        for symbol in self.symbols:
            self.symbols_queue.put(symbol)

        # 3. 启动检查线程
        logging.info(f"🔍 启动 {self.num_checker_threads} 个检查线程...")
        with tqdm(total=len(self.symbols),
                  desc="[数据检查]",
                  ncols=100,
                  file=sys.stdout) as pbar:
            self.check_progress_bar = pbar
            threads = []
            for i in range(self.num_checker_threads):
                t = threading.Thread(target=self._checker_worker,
                                     name=f"Checker-{i}",
                                     daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

        self.check_progress_bar = None

        # 4. 启动下载和写入
        total_downloads = self.download_tasks_queue.qsize()
        if total_downloads > 0:
            logging.info(f"📥 共 {total_downloads} 只股票需要下载/更新。")
            logging.info(
                f"🚀 启动 {self.num_downloader_threads} 个下载线程 和 1 个写入线程...")

            writer_thread = threading.Thread(target=self._consumer_worker,
                                             name="DB-Writer",
                                             daemon=True)
            writer_thread.start()

            with tqdm(total=total_downloads,
                      desc="[数据下载]",
                      ncols=100,
                      file=sys.stdout) as pbar:
                self.download_progress_bar = pbar
                dl_threads = []
                for i in range(self.num_downloader_threads):
                    t = threading.Thread(target=self._producer_worker,
                                         name=f"Downloader-{i}",
                                         daemon=True)
                    t.start()
                    dl_threads.append(t)
                for t in dl_threads:
                    t.join()

            self.producers_finished_event.set()
            writer_thread.join()
            logging.info("--- ✅ 数据准备流程结束 ---")
        else:
            logging.info("✅ 所有数据已完整，无需下载。")

    def _checker_worker(self):
        while True:
            try:
                symbol = self.symbols_queue.get_nowait()
            except queue.Empty:
                break

            missing = self._find_missing_date_ranges(symbol)
            if missing:
                self.download_tasks_queue.put((symbol, missing))

            self.symbols_queue.task_done()
            if self.check_progress_bar: self.check_progress_bar.update(1)

    def _find_missing_date_ranges(self, symbol):
        # 简化的检查逻辑：查询已有数据的日期集合，与日历对比
        query = f"SELECT date FROM {self.table_name} WHERE code = ?"
        df = self.db_handler.query_data(query, (symbol, ))
        if df is None or df.empty:
            return [(pd.to_datetime(self.start_date).date(),
                     pd.to_datetime(self.end_date).date())]

        existing_dates = set(pd.to_datetime(df['date']).dt.date)
        missing_dates = sorted(list(self.all_trade_dates_set - existing_dates))

        if not missing_dates: return []

        # 将离散日期合并为区间 (简化处理，这里直接返回起止时间，实际下载会覆盖中间已有的)
        return [(missing_dates[0], missing_dates[-1])]

    def _producer_worker(self):
        while True:
            try:
                task = self.download_tasks_queue.get_nowait()
            except queue.Empty:
                break

            symbol, ranges = task
            # 简单起见，取第一个区间的起止
            start, end = ranges[0]

            # 优先尝试 SQLite (如果是本地源)，否则尝试 Tushare/Akshare
            # 这里简化逻辑，直接遍历 providers
            df = None
            for name, provider in self._local.providers.items():
                try:
                    df = provider.get_daily_price(symbol,
                                                  start.strftime('%Y%m%d'),
                                                  end.strftime('%Y%m%d'))
                    if df is not None and not df.empty:
                        break
                except:
                    continue

            if df is not None and not df.empty:
                self.results_queue.put(df)

            self.download_tasks_queue.task_done()
            if self.download_progress_bar: self.download_progress_bar.update(1)

    def _consumer_worker(self):
        batch = []
        while not (self.producers_finished_event.is_set()
                   and self.results_queue.empty()):
            try:
                df = self.results_queue.get(timeout=1)
                batch.append(df)
                if len(batch) >= self.batch_size:
                    self._save_batch(batch)
                    batch = []
            except queue.Empty:
                continue
        if batch: self._save_batch(batch)

    def _save_batch(self, batch):
        if not batch: return
        try:
            full_df = pd.concat(batch)
            # 确保 code 列存在 (provider 返回的数据应该包含 code)
            self.db_handler.save_data(full_df, self.table_name)
        except Exception as e:
            logging.error(f"❌ 批量写入失败: {e}")

    def __del__(self):
        if hasattr(self, 'db_handler'):
            self.db_handler.close_connection()
