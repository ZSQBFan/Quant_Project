# backtest/utils/roller.py

"""
回测数据滚动器

提供从 backtrader 数据对象中提取历史数据的工具函数。
用于需要历史数据的分配器（如风险平价、最大夏普比等）。
"""

import numpy as np
import pandas as pd
from typing import List, Any, Optional
import logging


class BacktestDataRoller:
    """
    回测数据滚动器

    从 backtrader 数据对象中提取历史价格数据，用于计算协方差矩阵、收益率等。

    使用示例:
        roller = BacktestDataRoller()
        returns = roller.get_returns(selected_stocks, lookback=60)
    """

    def __init__(self):
        """初始化数据滚动器"""
        self.logger = logging.getLogger(__name__)

    def get_close_prices(self,
                        data_objects: List[Any],
                        lookback: int) -> Optional[pd.DataFrame]:
        """
        获取多只股票的历史收盘价

        Args:
            data_objects: backtrader 数据对象列表
            lookback: 回看周期（天数）

        Returns:
            DataFrame: 行为日期，列为股票代码，值为收盘价
                      如果数据不足则返回 None
        """
        if not data_objects:
            self.logger.warning("数据对象列表为空")
            return None

        price_data = {}

        for data in data_objects:
            stock_name = getattr(data, '_name', 'Unknown')

            # 检查数据长度是否足够
            if len(data) < lookback:
                self.logger.warning(
                    f"股票 {stock_name} 数据不足: 需要 {lookback} 天，实际 {len(data)} 天"
                )
                return None

            # 提取历史收盘价
            # data.close[0] 是当前价格，data.close[-1] 是昨天的价格
            # 我们需要提取 [-lookback+1, 0] 这个范围
            try:
                prices = [data.close[-i] for i in range(lookback-1, -1, -1)]
                price_data[stock_name] = prices
            except Exception as e:
                self.logger.error(f"提取股票 {stock_name} 价格数据失败: {e}")
                return None

        # 转换为 DataFrame
        df = pd.DataFrame(price_data)

        self.logger.debug(f"成功提取 {len(data_objects)} 只股票的 {lookback} 天价格数据")

        return df

    def get_returns(self,
                   data_objects: List[Any],
                   lookback: int,
                   method: str = 'log') -> Optional[pd.DataFrame]:
        """
        获取多只股票的历史收益率

        Args:
            data_objects: backtrader 数据对象列表
            lookback: 回看周期（天数）
            method: 收益率计算方法，'log' 或 'simple'

        Returns:
            DataFrame: 行为日期，列为股票代码，值为收益率
                      如果数据不足则返回 None
        """
        # 获取收盘价
        prices = self.get_close_prices(data_objects, lookback + 1)  # 需要多一天来计算收益率

        if prices is None:
            return None

        # 计算收益率
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        elif method == 'simple':
            returns = prices.pct_change()
        else:
            self.logger.error(f"不支持的收益率计算方法: {method}")
            return None

        # 去掉第一行（NaN）
        returns = returns.iloc[1:]

        # 去除包含 NaN 或 inf 的行
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.dropna()

        self.logger.debug(f"成功计算 {len(data_objects)} 只股票的 {len(returns)} 天收益率")

        return returns

    def get_covariance_matrix(self,
                             data_objects: List[Any],
                             lookback: int) -> Optional[np.ndarray]:
        """
        获取股票收益率的协方差矩阵

        Args:
            data_objects: backtrader 数据对象列表
            lookback: 回看周期（天数）

        Returns:
            ndarray: 协方差矩阵，如果数据不足则返回 None
        """
        returns = self.get_returns(data_objects, lookback)

        if returns is None or len(returns) < 2:
            self.logger.warning("收益率数据不足，无法计算协方差矩阵")
            return None

        # 计算协方差矩阵
        cov_matrix = returns.cov().values

        self.logger.debug(f"成功计算 {len(data_objects)}x{len(data_objects)} 协方差矩阵")

        return cov_matrix

    def get_mean_returns(self,
                        data_objects: List[Any],
                        lookback: int) -> Optional[np.ndarray]:
        """
        获取股票的平均收益率

        Args:
            data_objects: backtrader 数据对象列表
            lookback: 回看周期（天数）

        Returns:
            ndarray: 平均收益率向量，如果数据不足则返回 None
        """
        returns = self.get_returns(data_objects, lookback)

        if returns is None:
            return None

        mean_returns = returns.mean().values

        self.logger.debug(f"成功计算 {len(data_objects)} 只股票的平均收益率")

        return mean_returns
