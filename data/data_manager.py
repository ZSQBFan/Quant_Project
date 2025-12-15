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
    # 调试：检查输入数据的索引结构
    if hasattr(df, 'index'):
        logging.debug(f"    > 🔍 _calculate_forward_returns 输入索引: {df.index.names}")
        logging.debug(f"    > 🔍 _calculate_forward_returns 输入索引级别数: {df.index.nlevels}")
    
    # 确保按日期排序
    df = df.sort_index()
    for p in periods:
        # shift(-p) 将未来的价格向上平移，对齐到当前日期
        future_price = df['close'].shift(-p)
        df[f'forward_return_{p}d'] = (future_price / df['close']) - 1
    
    # 调试：检查输出数据的索引结构
    if hasattr(df, 'index'):
        logging.debug(f"    > 🔍 _calculate_forward_returns 输出索引: {df.index.names}")
        logging.debug(f"    > 🔍 _calculate_forward_returns 输出索引级别数: {df.index.nlevels}")
    
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
            # 从配置中获取 Tushare token，如果没有则使用环境变量或默认值
            tushare_token = None
            for cls, kwargs in self.provider_configs:
                if cls.__name__ == 'TushareDataProvider' or cls.__name__ == 'TushareTradingCalendar':
                    tushare_token = kwargs.get('token')
                    break
            
            if not tushare_token:
                # 尝试从环境变量获取
                import os
                tushare_token = os.getenv('TUSHARE_TOKEN')
            
            if tushare_token:
                self.calendar_provider = TushareTradingCalendar(token=tushare_token)
            else:
                raise ValueError("未找到 Tushare token")
        except Exception as e:
            logging.warning(f"⚠️ Tushare 初始化失败: {e}，使用 Akshare 作为备选")
            self.calendar_provider = AkshareTradingCalendar()

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

            # 添加详细调试信息
            logging.debug(f"    > 🔍 查询返回数据形状: {df.shape}")
            logging.debug(f"    > 🔍 查询返回列名: {df.columns.tolist()}")
            if hasattr(df.index, 'names'):
                logging.debug(f"    > 🔍 查询返回索引名称: {df.index.names}")
            logging.debug(f"    > 🔍 查询返回索引类型: {type(df.index)}")
            if len(df) > 0:
                logging.debug(f"    > 🔍 第一个索引值: {df.index[0]} (类型: {type(df.index[0])})")

            # 确保 date 是 datetime 类型并设为索引
            if 'date' in df.columns:
                logging.debug(f"    > 🔍 发现date列，开始转换...")
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                logging.debug(f"    > ✅ date列转换完成，索引类型: {type(df.index)}")

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

        # 3. 使用批量查询优化（新增）
        logging.info(f"🚀 开始批量合并数据...")
        return self._batch_merge_data(universe, table_col_map, fundamental_cache, industry_map, load_industry)

    def _batch_merge_data(self, universe: list, table_col_map: dict, 
                         fundamental_cache: dict, industry_map: dict, 
                         load_industry: bool) -> pd.DataFrame:
        """
        【批量查询优化版本】一次性查询所有股票数据，大幅提升5000+股票的处理速度
        """
        logging.info(f"⚡ [批量查询] 正在为 {len(universe)} 只股票执行批量数据合并...")
        
        # 1. 批量查询所有股票的行情数据（核心优化）
        price_cols = table_col_map.get('stock_daily_prices', ['close'])
        
        # 构建股票代码列表（处理SQL注入风险）
        symbols_str = "','".join([sym.replace("'", "''") for sym in universe])
        
        # 构建列查询字符串
        query_cols = ['code', 'date'] + [col for col in price_cols if col not in ['code', 'date']]
        cols_str = ", ".join(query_cols)
        
        # 批量查询所有股票数据
        batch_query = f"""
            SELECT {cols_str} 
            FROM stock_daily_prices 
            WHERE code IN ('{symbols_str}') 
            AND date BETWEEN ? AND ?
            ORDER BY code, date
        """
        
        logging.info(f"  > 🚀 执行批量行情数据查询 ({len(universe)} 只股票)...")
        all_price_df = self.db_handler.query_data(
            batch_query, 
            (self.start_date, self.end_date)
        )
        
        if all_price_df is None or all_price_df.empty:
            logging.warning("⚠️ 批量查询返回空数据")
            return pd.DataFrame()
        
        logging.info(f"  > ✅ 批量查询完成，获得 {len(all_price_df)} 行数据")
        
        # 2. 批量合并基本面数据（从预加载缓存中）
        if fundamental_cache:
            logging.info(f"  > 🚀 开始批量合并基本面数据...")
            
            # 创建一个大的基本面数据框
            all_fund_dfs = []
            for table_name, fund_data_dict in fundamental_cache.items():
                # 从缓存中提取所有股票的基本面数据
                for symbol, fund_df in fund_data_dict.items():
                    if symbol in universe:  # 只处理需要的股票
                        fund_df_copy = fund_df.copy()
                        fund_df_copy['code'] = symbol
                        fund_df_copy = fund_df_copy.reset_index()
                        all_fund_dfs.append(fund_df_copy)
            
            if all_fund_dfs:
                # 合并所有基本面数据
                combined_fund_df = pd.concat(all_fund_dfs, ignore_index=True)
                combined_fund_df['date'] = pd.to_datetime(combined_fund_df['date'])
                
                # 按code和date合并到行情数据
                merge_cols = [col for col in combined_fund_df.columns if col not in ['code', 'date']]
                if merge_cols:
                    all_price_df = all_price_df.merge(
                        combined_fund_df, 
                        on=['code', 'date'], 
                        how='left',
                        suffixes=('', '_fund')
                    )
                    
                    # 处理重复列名
                    for col in merge_cols:
                        if f"{col}_fund" in all_price_df.columns:
                            all_price_df[col] = all_price_df[col].fillna(all_price_df[f"{col}_fund"])
                            all_price_df.drop(columns=[f"{col}_fund"], inplace=True)
        
        # 3. 向量化添加行业数据
        if load_industry and industry_map:
            logging.info(f"  > 🚀 向量化添加行业数据...")
            all_price_df['industry'] = all_price_df['code'].map(industry_map)
        
        # 4. 重塑索引格式
        logging.info(f"  > ⚙️ 重塑数据索引...")
        
        # 确保有date列，如果没有则使用索引
        if 'date' not in all_price_df.columns:
            if all_price_df.index.name == 'date':
                all_price_df = all_price_df.reset_index()
            else:
                logging.warning("⚠️ 未找到date列，使用当前日期作为默认")
                all_price_df['date'] = pd.Timestamp.now()
        
        # 调试：检查重命名前的列
        logging.debug(f"  > 🔍 调试：重命名前列名: {all_price_df.columns.tolist()}")
        
        all_price_df.rename(columns={'code': 'asset'}, inplace=True)
        
        # 调试：检查重命名后的列和索引
        logging.debug(f"  > 🔍 调试：重命名后列名: {all_price_df.columns.tolist()}")
        logging.debug(f"  > 🔍 调试：当前索引名称: {all_price_df.index.names}")
        
        # 检查是否已有多级索引
        if isinstance(all_price_df.index, pd.MultiIndex):
            logging.warning(f"⚠️ 数据已有多级索引: {all_price_df.index.names}")
            # 如果已有多级索引，先重置
            all_price_df = all_price_df.reset_index()
            logging.debug(f"  > 🔍 重置后列名: {all_price_df.columns.tolist()}")
        
        all_price_df.set_index(['date', 'asset'], inplace=True)
        all_price_df.sort_index(inplace=True)
        
        # 调试：检查最终索引结构
        logging.debug(f"  > 🔍 调试：最终索引名称: {all_price_df.index.names}")
        logging.debug(f"  > 🔍 调试：最终索引级别数: {all_price_df.index.nlevels}")
        
        # 5. 前向填充缺失值（基本面数据常用）
        # 修复：使用级别号而不是名称来避免重复名称问题
        try:
            all_price_df = all_price_df.groupby(level=1, group_keys=False).apply(
                lambda x: x.ffill() if not x.empty else x
            )
        except Exception as e:
            logging.warning(f"⚠️ 使用级别号1进行前向填充失败: {e}")
            # 尝试查找asset对应的级别号
            for i, name in enumerate(all_price_df.index.names):
                if name == 'asset':
                    logging.debug(f"  > 找到asset在级别号 {i}，用于前向填充")
                    all_price_df = all_price_df.groupby(level=i, group_keys=False).apply(
                        lambda x: x.ffill() if not x.empty else x)
                    break
        
        # 【【【新增】】】: 在返回前统一处理重复索引，确保数据质量
        if isinstance(all_price_df.index, pd.MultiIndex):
            # 检查并去除重复索引
            if all_price_df.index.duplicated().any():
                dup_count = all_price_df.index.duplicated().sum()
                logging.warning(f"⚠️ [数据加载] 发现 {dup_count} 个重复索引，正在统一处理...")
                
                # 记录重复索引示例（用于调试）
                dup_indices = all_price_df.index[all_price_df.index.duplicated()]
                if len(dup_indices) > 0:
                    logging.debug(f"重复索引示例: {dup_indices[:5].tolist()}")
                
                # 去除重复索引，保留最后一个值
                original_shape = all_price_df.shape
                all_price_df = all_price_df[~all_price_df.index.duplicated(keep='last')]
                logging.info(f"✅ [数据加载] 已去除重复索引，数据形状从 {original_shape} 变为 {all_price_df.shape}")
        
        logging.info(f"  > ✅ 批量合并完成！最终数据形状: {all_price_df.shape}")
        return all_price_df

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

        # 调试：检查索引结构
        logging.debug(f"  > 🔍 调试：检查数据索引结构...")
        logging.debug(f"    > 索引名称: {all_data.index.names}")
        logging.debug(f"    > 索引级别数: {all_data.index.nlevels}")
        logging.debug(f"    > 数据形状: {all_data.shape}")
        logging.debug(f"    > 列名: {all_data.columns.tolist()}")
        
        # 检查是否有重复的索引名称
        if all_data.index.nlevels > 1:
            level_names = all_data.index.names
            name_counts = {}
            for name in level_names:
                if name in name_counts:
                    name_counts[name] += 1
                else:
                    name_counts[name] = 1
            
            for name, count in name_counts.items():
                if count > 1:
                    logging.warning(f"⚠️ 发现重复的索引名称: '{name}' 出现 {count} 次")
        
        # 使用 groupby().apply()
        # 修复：使用级别号而不是名称来避免重复名称问题
        try:
            # 尝试使用级别号（假设asset是第二级索引，级别号为1）
            returns_df = all_data.groupby(level=1, group_keys=False).apply(
                lambda x: _calculate_forward_returns(x, forward_return_periods))
        except Exception as e:
            logging.error(f"❌ 使用级别号1失败: {e}")
            # 如果级别号1不正确，尝试查找asset对应的级别号
            if all_data.index.nlevels > 1:
                for i, name in enumerate(all_data.index.names):
                    if name == 'asset':
                        logging.debug(f"  > 找到asset在级别号 {i}")
                        returns_df = all_data.groupby(level=i, group_keys=False).apply(
                            lambda x: _calculate_forward_returns(x, forward_return_periods))
                        break
                else:
                    # 如果找不到asset名称，直接使用第一级
                    logging.warning("⚠️ 未找到asset索引，使用第一级索引")
                    returns_df = all_data.groupby(level=0, group_keys=False).apply(
                        lambda x: _calculate_forward_returns(x, forward_return_periods))
            else:
                # 单级索引情况
                logging.warning("⚠️ 单级索引，不使用level参数")
                returns_df = all_data.groupby(group_keys=False).apply(
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
            # 使用 _get_provider 方法确保线程安全
            df = None
            
            # 尝试各种数据提供者
            provider_names = ['SQLiteDataProvider', 'TushareDataProvider', 'AkshareDataProvider']
            for provider_name in provider_names:
                provider = self._get_provider(provider_name)
                if provider:
                    try:
                        df = provider.fetch_data(symbol,
                                                start.strftime('%Y-%m-%d'),
                                                end.strftime('%Y-%m-%d'))
                        if df is not None and not df.empty:
                            # 添加 code 列以便后续处理
                            df['code'] = symbol
                            break
                    except Exception as e:
                        logging.debug(f"  > 📡 [{provider_name}] 获取 {symbol} 数据失败: {e}")
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
