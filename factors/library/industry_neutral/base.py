"""
行业中性因子基类

提供行业中性化的通用功能
"""

import pandas as pd
import logging
from typing import Tuple


class IndustryNeutralFactorBase:
    """
    行业中性因子基类

    提供通用的行业中性化方法。
    复合因子不继承 BaseFactor，因为它们使用不同的计算范式（全量宽表）。
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def neutralize_by_industry(
        self,
        all_data_df: pd.DataFrame,
        base_factor_col: str,
        factor_name: str
    ) -> pd.Series:
        """
        执行行业中性化

        Args:
            all_data_df: 包含因子值和行业列的 DataFrame (MultiIndex: date, asset)
            base_factor_col: 基础因子列名
            factor_name: 中性化后的因子名称

        Returns:
            行业中性化后的因子值 Series
        """
        self.logger.info(f"      > (2/2) 正在为 {factor_name} 执行行业中性化...")

        if 'industry' not in all_data_df.columns:
            self.logger.error(f"❌ [{factor_name}] 缺少 'industry' 列")
            return None

        # 预处理：去除重复索引
        if all_data_df.index.duplicated().any():
            dup_count = all_data_df.index.duplicated().sum()
            self.logger.warning(f"⚠️ [{factor_name}] 输入数据有 {dup_count} 个重复索引，正在预处理...")
            all_data_df = all_data_df[~all_data_df.index.duplicated(keep='last')]

        # 过滤有效数据
        valid_data = all_data_df.dropna(subset=[base_factor_col, 'industry'])

        if valid_data.empty:
            self.logger.warning(f"⚠️ [{factor_name}] 没有有效数据进行中性化")
            return None

        # 计算行业均值并进行中性化
        industry_means = valid_data.groupby(['date', 'industry'])[base_factor_col].transform('mean')
        neutral_factor = valid_data[base_factor_col] - industry_means

        neutral_factor.name = factor_name

        # 最终检查
        if neutral_factor.index.duplicated().any():
            dup_count = neutral_factor.index.duplicated().sum()
            self.logger.warning(f"⚠️ [{factor_name}] 结果有 {dup_count} 个重复索引，进行最终处理...")
            neutral_factor = neutral_factor[~neutral_factor.index.duplicated(keep='last')]

        return neutral_factor.sort_index()

    def calculate(self, all_data_df: pd.DataFrame) -> pd.Series:
        """
        计算因子（子类必须实现）

        Args:
            all_data_df: 全量宽表数据 (MultiIndex: date, asset)

        Returns:
            因子值 Series
        """
        raise NotImplementedError("子类必须实现 calculate 方法")
