# core/factor_combiner.py (已重构)
import pandas as pd
import numpy as np
import logging  # <- 【【【新增】】】


class BaseFactorCombiner:
    """
    【【架构文件 - 基类】】
    因子合成器的基类。
    
    【【重构日志】】:
    - 2025-11-09:
      - 引入 'logging' 模块，替换所有 'print' 语句。
    """

    def __init__(self, **kwargs):
        pass

    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("每个合成器子类都必须实现 combine 方法")


class EqualWeightCombiner(BaseFactorCombiner):
    """
    【合成器 1: 等权重】
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 【【【新增】】】
        logging.info("ℹ️ EqualWeightCombiner 已初始化 (模式: 等权求和)。")

    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        """
        通过对所有因子得分求和来实现等权重合成。
        """
        logging.debug("  > ⚙️ [EqualWeight] 正在执行等权合成 (sum)...")
        return standardized_df.sum(axis=1)


class DynamicSignificanceCombiner(BaseFactorCombiner):
    """
    【合成器 2: 动态显著性加权】
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 【【【新增】】】
        logging.info("ℹ️ DynamicSignificanceCombiner 已初始化 (模式: 动态显著性加权)。")

    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        """
        执行动态显著性加权合成。
        """
        logging.debug("  > ⚙️ [DynamicSignificance] 正在执行动态显著性加权合成...")

        # 1. 计算每个因子的绝对值
        abs_significance = standardized_df.abs()

        # 2. 计算每行（每只股票）的“总显著性”
        total_significance = abs_significance.sum(axis=1)
        total_significance.replace(0, np.finfo(float).eps, inplace=True)

        # 3. 计算每个因子的动态权重
        dynamic_weights = abs_significance.div(total_significance, axis=0)

        # 4. 使用动态权重对原始的（带符号的）标准化因子值进行加权求和
        combined_score = (standardized_df * dynamic_weights).sum(axis=1)

        return combined_score


class DynamicWeightCombiner(BaseFactorCombiner):
    """
    【合成器 3: (原 ICIR) 动态权重合成器】
    """

    def __init__(self, factor_weights: dict, **kwargs):
        """
        初始化 动态权重 合成器。
        """
        super().__init__(**kwargs)
        if not isinstance(factor_weights, dict):
            # 【【【修改】】】
            logging.critical("⛔ 'factor_weights' 必须是一个字典。")
            raise ValueError("factor_weights 必须是一个字典")
        self.factor_weights = factor_weights
        # 【【【修改】】】
        logging.info(f"--- DynamicWeightCombiner 已初始化 ---")
        logging.info(f"    > 初始权重: {self.factor_weights}")

    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        """
        执行基于【当前内部】权重的静态加权合成。
        """
        logging.debug(
            f"  > ⚙️ [DynamicWeight] 正在使用 {len(self.factor_weights)} 个权重进行合成..."
        )

        # 1. 将权重字典转换为 Series，以便于对齐
        weights_series = pd.Series(self.factor_weights)

        # 2. 将权重 Series 与 standardized_df 的列进行对齐
        try:
            aligned_weights = weights_series.reindex(
                standardized_df.columns).fillna(0)
        except Exception as e:
            # 【【【修改】】】
            logging.error(
                f"❌ [DynamicWeight] 对齐权重时出错。DataFrame 列: {standardized_df.columns}",
                exc_info=True)
            logging.error(f"    > 权重: {self.factor_weights}")
            raise e

        # 3. 检查是否有任何在权重字典中指定的因子在数据中缺失
        missing_factors = weights_series.index.difference(
            standardized_df.columns)
        if not missing_factors.empty:
            # 【【【修改】】】
            logging.warning(
                f"  > ⚠️ [DynamicWeight] 警告: 权重字典中的因子 {list(missing_factors)} 在"
                f" 当前的 standardized_df 中未找到，它们将被忽略。")

        # 4. 执行加权求和
        combined_score = (standardized_df * aligned_weights).sum(axis=1)
        return combined_score

    def update_weights(self, new_factor_weights: dict):
        """
        【【核心方法】】
        动态更新合成器内部的因子权重。
        """
        if not isinstance(new_factor_weights, dict):
            # 【【【修改】】】
            logging.warning(
                "  > ⚠️ [DynamicWeight] 尝试更新权重失败，提供的 new_factor_weights 不是有效字典。"
            )
            return

        # 仅在权重实际发生变化时打印日志
        if self.factor_weights != new_factor_weights:
            has_changed = False
            all_keys = set(self.factor_weights.keys()) | set(
                new_factor_weights.keys())

            # 使用一个小的阈值来比较浮点数
            threshold = 1e-6
            for k in all_keys:
                if abs(
                        self.factor_weights.get(k, 0) -
                        new_factor_weights.get(k, 0)) > threshold:
                    has_changed = True
                    break

            if has_changed:
                # 【【【修改】】】
                # (使用 INFO 级别，因为这是一个关键事件)
                logging.info(f"--- 权重已在 {pd.Timestamp.now().date()} 更新 ---")
                logging.info(f"    > 📊 旧权重: {self.factor_weights}")
                self.factor_weights = new_factor_weights
                logging.info(f"    > 📊 新权重: {self.factor_weights}")
            else:
                # 权重已计算，但值与上期相同
                self.factor_weights = new_factor_weights
                logging.debug("  > ℹ️ [DynamicWeight] 权重已重新计算，但与上期相同，未更新。")


class FixedWeightCombiner(BaseFactorCombiner):
    """
    【合成器 4: 固定权重】
    """

    def __init__(self, factor_weights: dict, **kwargs):
        """
        初始化固定权重合成器。
        """
        super().__init__(**kwargs)
        if not isinstance(factor_weights, dict):
            # 【【【修改】】】
            logging.critical("⛔ 'factor_weights' 必须是一个字典。")
            raise ValueError("factor_weights 必须是一个字典")
        self.factor_weights = factor_weights
        # 【【【修改】】】
        logging.info(f"--- FixedWeightCombiner 已初始化 ---")
        logging.info(f"    > 固定权重: {self.factor_weights}")

    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        """
        执行基于固定权重的静态加权合成。
        """
        logging.debug(
            f"  > ⚙️ [FixedWeight] 正在使用 {len(self.factor_weights)} 个固定权重进行合成..."
        )

        # 1. 将权重字典转换为 Series
        weights_series = pd.Series(self.factor_weights)

        # 2. 对齐
        try:
            aligned_weights = weights_series.reindex(
                standardized_df.columns).fillna(0)
        except Exception as e:
            # 【【【修改】】】
            logging.error(
                f"❌ [FixedWeight] 对齐权重时出错。DataFrame 列: {standardized_df.columns}",
                exc_info=True)
            logging.error(f"    > 权重: {self.factor_weights}")
            raise e

        # 3. 检查缺失
        missing_factors = weights_series.index.difference(
            standardized_df.columns)
        if not missing_factors.empty:
            # 【【【修改】】】
            logging.warning(
                f"  > ⚠️ [FixedWeight] 警告: 权重字典中的因子 {list(missing_factors)} 在"
                f" 当前的 standardized_df 中未找到，它们将被忽略。")

        # 4. 执行加权求和
        combined_score = (standardized_df * aligned_weights).sum(axis=1)
        return combined_score
