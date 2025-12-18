"""固定权重合成器"""
import pandas as pd
import logging
from .base import BaseFactorCombiner
from core.registry import register_combiner

@register_combiner('FixedWeight', description='固定权重合成策略')
class FixedWeightCombiner(BaseFactorCombiner):
    """
    固定权重合成器。
    
    支持部分因子配置权重，剩余因子均分剩余权重的逻辑。
    """
    def __init__(self, factor_weights: dict = None, **kwargs):
        """
        初始化固定权重合成器。

        Args:
            factor_weights: 因子权重字典，例如 {'factor1': 0.4, 'factor2': 0.3}
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.factor_weights = factor_weights or {}
    
    def combine(self, standardized_df: pd.DataFrame) -> pd.Series:
        """
        合成因子。

        逻辑：
        1. 读取配置的因子权重。
        2. 如果实际参与计算的因子权重未在配置中定义，抛出 warning。
        3. 优先分配已配置因子的权重。
        4. 剩余未配置权重的因子均分剩余权重：(1 - 已配置因子总权重) / 未配置因子数。
        5. 如果已配置因子总权重大于 1，抛出 error 并终止计算。
        6. 允许出现负权重。

        Args:
            standardized_df: 标准化后的因子值 DataFrame，列名为因子名

        Returns:
            合成后的因子值 Series
        """
        all_factors = standardized_df.columns.tolist()
        if not all_factors:
            return pd.Series(dtype=float)

        configured_weights = {}
        unconfigured_factors = []
        
        # 1. 分类因子：已配置和未配置
        for factor in all_factors:
            if factor in self.factor_weights:
                configured_weights[factor] = self.factor_weights[factor]
            else:
                unconfigured_factors.append(factor)
        
        # 2. 如果有未配置的因子，抛出 warning
        if unconfigured_factors:
            logging.warning(f"⚠️ 因子 {unconfigured_factors} 未在配置中定义权重，将均分剩余权重。")
            
        # 3. 计算已配置因子的总权重
        total_configured_weight = sum(configured_weights.values())
        
        # 4. 检查总权重是否超过 1
        if total_configured_weight > 1.000001:  # 考虑浮点数精度
            error_msg = f"❌ 已配置因子总权重 ({total_configured_weight}) 大于 1，终止计算。"
            logging.error(error_msg)
            raise ValueError(error_msg)
            
        # 5. 分配权重
        final_weights = configured_weights.copy()
        if unconfigured_factors:
            remaining_weight = 1.0 - total_configured_weight
            avg_remaining_weight = remaining_weight / len(unconfigured_factors)
            for factor in unconfigured_factors:
                final_weights[factor] = avg_remaining_weight
        
        # 6. 执行合成
        weights_series = pd.Series(final_weights)
        aligned_weights = weights_series.reindex(all_factors).fillna(0)
        
        logging.info(f"因子合成权重分配: {final_weights}")
        
        return (standardized_df * aligned_weights).sum(axis=1)
