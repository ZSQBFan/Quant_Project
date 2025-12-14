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
        # 【【【核心配置：列名映射字典 (完整版)】】】
        # 格式: '因子计算用的通用列名': ('数据库表名', 'CSMAR原始字段代码')
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
            'amplitude': ('stock_daily_prices', 'amplitude'),
            'price_change': ('stock_daily_prices', 'price_change'),

            # --- 2. 行业/元数据表 (stock_kind) ---
            'industry': ('stock_kind', 'Nnindnme'),  # 行业名称
            'industry_code': ('stock_kind', 'Nnindcd'),  # 行业代码
            'stk_name': ('stock_kind', 'Stknme'),  # 股票简称
            'list_date': ('stock_kind', 'Listdt'),  # 上市日期
            'ownership': ('stock_kind', 'OWNERSHIPTYPE'),  # 企业性质
            'market_type': ('stock_kind', 'Markettype'),  # 市场类型
            'status': ('stock_kind', 'Statco'),  # 上市状态

            # --- 3. 资产负债表 (Stock_BalanceSheet) ---
            # 核心权益与资本
            'total_equity_parent':
            ('Stock_BalanceSheet', 'A003100000'),  # 归母所有者权益 (B/P, ROE分母)
            'share_capital':
            ('Stock_BalanceSheet', 'A003101000'),  # 实收资本/股本 (计算市值)

            # 资产端
            'total_assets': ('Stock_BalanceSheet', 'A001000000'),  # 资产总计
            'current_assets': ('Stock_BalanceSheet', 'A001100000'),  # 流动资产
            'fixed_assets': ('Stock_BalanceSheet', 'A001212000'),  # 固定资产净额
            'intangible_assets':
            ('Stock_BalanceSheet', 'A001218000'),  # 无形资产净额
            'goodwill': ('Stock_BalanceSheet', 'A001220000'),  # 商誉净额
            'inventory': ('Stock_BalanceSheet', 'A001123000'),  # 存货净额
            'accounts_receivable':
            ('Stock_BalanceSheet', 'A001111000'),  # 应收账款净额

            # 负债端
            'total_liabilities': ('Stock_BalanceSheet', 'A002000000'),  # 负债合计
            'current_liabilities': ('Stock_BalanceSheet',
                                    'A002100000'),  # 流动负债

            # --- 4. 利润表 (stock_ProfitSheet) ---
            # 收入与成本
            'total_revenue': ('stock_ProfitSheet',
                              'B001100000'),  # 营业总收入 (成长因子)
            'cost_of_goods_sold': ('stock_ProfitSheet', 'B001201000'),  # 营业成本

            # 利润层级
            'operating_profit': ('stock_ProfitSheet', 'B001300000'),  # 营业利润
            'total_profit': ('stock_ProfitSheet', 'B001000000'),  # 利润总额
            'net_profit_parent': ('stock_ProfitSheet',
                                  'B002000101'),  # 归母净利润 (E/P, ROE分子)

            # 费用与税
            'selling_expenses': ('stock_ProfitSheet', 'B001209000'),  # 销售费用
            'admin_expenses': ('stock_ProfitSheet', 'B001210000'),  # 管理费用
            'rd_finance_expenses': ('stock_ProfitSheet',
                                    'B001216000'),  # 研发/财务费用
            'income_tax_expense': ('stock_ProfitSheet', 'B002100000'),  # 所得税费用

            # --- 5. 现金流量表 (stock_CashFlowDirect) ---
            # 经营活动
            'net_cash_flow_ops': ('stock_CashFlowDirect',
                                  'C001000000'),  # 经营活动现金流净额 (CFO)

            # 投资活动
            'net_cash_flow_inv': ('stock_CashFlowDirect',
                                  'C002000000'),  # 投资活动现金流净额 (CFI)
            'capex': ('stock_CashFlowDirect',
                      'C002006000'),  # 购建长期资产支付 (CapEx)

            # 筹资活动
            'net_cash_flow_fin': ('stock_CashFlowDirect',
                                  'C003000000'),  # 筹资活动现金流净额 (CFF)
            'borrowing_cash': ('stock_CashFlowDirect',
                               'C003002000'),  # 取得借款收到的现金
            'dividends_paid': ('stock_CashFlowDirect',
                               'C003005000'),  # 分配股利/利息支付
            'other_fin_payment': ('stock_CashFlowDirect',
                                  'C003006000'),  # 支付其他筹资现金
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
    # 【【【核心方法 2 (升级版)：获取全市场合并数据 (支持基本面)】】】
    # ==========================================================================
    def get_all_data_for_universe(
            self,
            universe: list,
            required_columns: list = None) -> pd.DataFrame:
        logging.info(f"⚙️ [数据加载] 正在解析 {len(universe)} 只股票的数据需求...")

        # 1. 需求分析
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

        # 2. 预加载行业数据
        industry_map = {}
        if load_industry:
            logging.info("  > 正在预加载行业数据...")
            ind_query = "SELECT Stkcd, Nnindnme FROM stock_kind"
            ind_df = self.db_handler.query_data(ind_query)
            if ind_df is not None and not ind_df.empty:
                ind_df['Stkcd'] = ind_df['Stkcd'].astype(str).str.zfill(6)
                industry_map = ind_df.set_index('Stkcd')['Nnindnme'].to_dict()

        # ======================================================================
        # 【【【优化重点】】】: 基本面数据全量预加载
        # 避免在循环中进行 5000+ 次 SQL 查询
        # ======================================================================
        fundamental_cache = {
        }  # 结构: { 'Stock_BalanceSheet': { '600519': df, ... }, ... }

        for table_name, cols in table_col_map.items():
            if table_name == 'stock_daily_prices': continue

            logging.info(f"  > 🚀 [预加载] 正在全量读取 {table_name} ({cols})...")
            # 预加载该表所有股票的数据
            df_all = self._preload_fundamental_table(table_name, cols)

            if df_all is not None and not df_all.empty:
                # 按股票代码分组，存入字典
                # 假设 Stkcd 是索引的一部分或列
                # _preload_fundamental_table 返回的 df 包含 'Stkcd' 和 'date'

                # 将大表拆分为小字典，key是股票代码 (str)
                #这一步虽然耗时，但比几千次SQL快得多
                grouped = df_all.groupby('Stkcd')
                fundamental_cache[table_name] = {
                    str(k).zfill(6): v
                    for k, v in grouped
                }
                logging.info(
                    f"    > 已缓存 {len(fundamental_cache[table_name])} 只股票的 {table_name} 数据。"
                )

        # 3. 主循环
        all_dfs = []
        logging.info(f"🚀 开始合并数据...")

        # 只需要查行情的列
        price_cols = table_col_map.get('stock_daily_prices', [])

        for symbol in tqdm(universe, desc="[Data Load]"):
            # A. 获取基础行情 (这个必须逐个查，因为全量查行情表内存会爆)
            df_price = self.get_dataframe(symbol, columns=price_cols)

            if df_price is None or df_price.empty:
                continue

            # B. 内存合并基本面数据
            for table_name in fundamental_cache:
                # 从缓存字典里直接取，不需要 SQL
                stock_fund_data = fundamental_cache[table_name].get(symbol)

                if stock_fund_data is not None:
                    # stock_fund_data 已经清洗过，index是 date
                    # 删除 Stkcd 列防止重名
                    if 'Stkcd' in stock_fund_data.columns:
                        stock_fund_data = stock_fund_data.drop(
                            columns=['Stkcd'])

                    df_price = df_price.join(stock_fund_data, how='left')
                    df_price = df_price.ffill()  # 填充

            # C. 合并行业
            if load_industry:
                df_price['industry'] = industry_map.get(symbol)

            df_price['asset'] = symbol
            all_dfs.append(df_price)

        if not all_dfs:
            return pd.DataFrame()

        logging.info("⚙️ 正在堆叠数据框...")
        
        # 【【【调试日志】】】: 检查合并前的索引情况
        total_rows = sum(len(df) for df in all_dfs)
        logging.info(f"📊 合并前总行数: {total_rows}, 数据框数量: {len(all_dfs)}")
        
        # 检查是否有重复的asset
        asset_counts = {}
        for df in all_dfs:
            asset = df['asset'].iloc[0] if not df.empty else None
            if asset:
                asset_counts[asset] = asset_counts.get(asset, 0) + 1
        
        duplicates = {k: v for k, v in asset_counts.items() if v > 1}
        if duplicates:
            logging.warning(f"⚠️ 发现重复的asset: {duplicates}")
        
        final_df = pd.concat(all_dfs)
        
        # 检查合并后的索引唯一性
        if final_df.index.duplicated().any():
            dup_count = final_df.index.duplicated().sum()
            logging.warning(f"⚠️ 合并后发现 {dup_count} 个重复的date索引!")
        
        final_df.reset_index(inplace=True)
        final_df.set_index(['date', 'asset'], inplace=True)
        
        # 检查MultiIndex的唯一性并去重
        if final_df.index.duplicated().any():
            dup_count = final_df.index.duplicated().sum()
            logging.warning(f"⚠️ MultiIndex中发现 {dup_count} 个重复的(date, asset)索引!")
            dup_indices = final_df.index[final_df.index.duplicated()]
            logging.warning(f"重复索引示例: {dup_indices[:5].tolist()}")
            
            # 【【【修复】】】: 去除重复索引，保留最后一个
            final_df = final_df[~final_df.index.duplicated(keep='last')]
            logging.info(f"✅ 已去除重复索引，剩余行数: {len(final_df)}")
        
        final_df.sort_index(inplace=True)

        return final_df

    def _preload_fundamental_table(self, table_name: str,
                                   required_cols: list) -> pd.DataFrame | None:
        """
        【辅助方法】全量读取一张基本面表，并清洗列名。
        """
        db_cols = []
        rename_map = {}
        for col in required_cols:
            mapping = self.COLUMN_MAPPING.get(col)
            if mapping:
                db_cols.append(mapping[1])
                rename_map[mapping[1]] = col

        if not db_cols: return None

        # 读取 Stkcd, Accper 和所需列
        cols_str = ", ".join(['Stkcd', 'Accper'] + db_cols)
        # 限制时间范围优化性能 (可选，这里查全量比较安全)
        query = f"SELECT {cols_str} FROM {table_name} ORDER BY Accper"

        try:
            df = self.db_handler.query_data(query)
            if df is None or df.empty: return None

            # 清洗
            df['date'] = pd.to_datetime(df['Accper'])
            df.set_index('date', inplace=True)
            df.drop(columns=['Accper'], inplace=True, errors='ignore')
            df.rename(columns=rename_map, inplace=True)

            # 强制转数值
            cols_to_numeric = list(rename_map.values())
            df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric,
                                                            errors='coerce')

            # 去重 (Stkcd + Date 唯一)
            # 因为我们要 groupby Stkcd，所以这里先不 drop Stkcd
            if not df.empty:
                # reset_index 会把 date (index) 变成列，从而可以对 ['date', 'Stkcd'] 联合去重
                df = df.reset_index()
                df = df.drop_duplicates(subset=['date', 'Stkcd'], keep='last')
                df = df.set_index('date')

            return df
        except Exception as e:
            logging.error(f"全量预加载失败 {table_name}: {e}")
            return None

    def _get_fundamental_data(self, symbol: str, table_name: str,
                              required_cols: list) -> pd.DataFrame | None:
        """
        【辅助方法】查询单张基本面表的数据。
        返回: index为date(Accper)的DataFrame，列名为通用列名(如 share_capital)。
        """
        # 1. 找到这些通用列对应的数据库原始列名
        db_cols = []
        rename_map = {}

        for col in required_cols:
            mapping = self.COLUMN_MAPPING.get(col)
            if mapping:
                original_col = mapping[1]
                db_cols.append(original_col)
                rename_map[original_col] = col

        if not db_cols: return None

        # 2. 构建查询
        cols_str = ", ".join(['Accper'] + db_cols)

        # 尝试处理 symbol 格式 (CSMAR通常用整数存Stkcd)
        try:
            symbol_int = int(symbol)
        except:
            symbol_int = symbol

        query = f"SELECT {cols_str} FROM {table_name} WHERE Stkcd = ? ORDER BY Accper"

        try:
            df = self.db_handler.query_data(query, (symbol_int, ))
            if df is None or df.empty: return None

            # 3. 清洗数据
            df['date'] = pd.to_datetime(df['Accper'])
            df.set_index('date', inplace=True)
            df.drop(columns=['Accper'], inplace=True, errors='ignore')

            # 重命名回通用列名
            df.rename(columns=rename_map, inplace=True)

            # 转换数据类型为数值型 (防止字符串)
            df = df.apply(pd.to_numeric, errors='coerce')

            # 去重 (防止同一天发布两次财报导致的索引冲突)
            df = df[~df.index.duplicated(keep='last')]

            return df
        except Exception as e:
            return None

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
