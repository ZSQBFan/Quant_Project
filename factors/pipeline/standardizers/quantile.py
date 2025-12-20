"""分位数标准化器"""
import pandas as pd
import numpy as np
from scipy import stats
from .base import BaseStandardizer
from core.registry import register_standardizer

@register_standardizer('Quantile', description='分位数标准化')
class CrossSectionalQuantileStandardizer(BaseStandardizer):
    def __init__(self, n_quantiles: int = 100, output_distribution: str = 'uniform', **kwargs):
        """初始化分位数标准化器
        
        Args:
            n_quantiles: 分位数数量，默认为100
            output_distribution: 输出分布类型，'uniform'（均匀分布）或 'normal'（正态分布），默认为'uniform'
            **kwargs: 其他配置参数
        """
        super().__init__(**kwargs)
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        
        if output_distribution not in ['uniform', 'normal']:
            raise ValueError(f"output_distribution must be 'uniform' or 'normal', got '{output_distribution}'")
    
    def standardize(self, raw_signals_df: pd.DataFrame) -> pd.DataFrame:
        """标准化因子数据
        
        Args:
            raw_signals_df: 原始因子数据 DataFrame
            
        Returns:
            标准化后的因子数据 DataFrame
        """
        # 计算分位数排名（0到1之间的值）
        ranks = raw_signals_df.rank(pct=True)
        
        if self.output_distribution == 'uniform':
            # 均匀分布：保持现有的逻辑，映射到[-0.5, 0.5]
            return ranks - 0.5
        elif self.output_distribution == 'normal':
            # 正态分布：将分位数通过正态分布的累积分布函数的逆函数（PPF）映射到正态分布空间
            # 处理边界值，避免0和1导致的inf值
            ranks_clipped = ranks.clip(lower=1e-8, upper=1-1e-8)
            
            # 使用正态分布的逆累积分布函数（PPF）映射到标准正态分布空间
            # ppf返回的是标准正态分布的分位数，对应于输入的概率值
            normalized = stats.norm.ppf(ranks_clipped)
            
            # 确保返回 DataFrame 格式，保持与输入相同的索引和列名
            if isinstance(normalized, pd.DataFrame):
                return normalized
            else:
                # 如果是 numpy 数组，转换为 DataFrame
                return pd.DataFrame(normalized, index=raw_signals_df.index, columns=raw_signals_df.columns)
        else:
            raise ValueError(f"Unsupported output_distribution: {self.output_distribution}")
