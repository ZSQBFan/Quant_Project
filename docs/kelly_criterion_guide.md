# 凯利公式仓位管理器使用指南

## 概述

凯利公式（Kelly Criterion）是一种动态仓位管理方法，通过最大化长期资本增长率来确定最优投资比例。本项目实现了三种凯利公式变体：

- **Full Kelly**: 完整凯利公式 (f* = 1.0)
- **Half Kelly**: 半凯利公式 (f* = 0.5)
- **Quarter Kelly**: 四分之一凯利公式 (f* = 0.25)

## 核心公式

凯利公式的基本形式：

```
f* = (μ - r) / σ²
```

其中：
- `f*` 是最优仓位比例
- `μ` 是资产的期望收益率（年化）
- `r` 是无风险利率（年化）
- `σ²` 是收益率的方差（年化）

## 文件位置

### 实现文件
- `backtest/pipeline/capital/kelly_criterion.py` - 凯利公式管理器实现
- `backtest/pipeline/capital/__init__.py` - 导出接口

### 配置文件
- `configs/backtest/pipeline/capital.yaml` - 凯利公式管理器配置

## 配置说明

在 `configs/backtest/pipeline/capital.yaml` 中配置：

```yaml
# Full Kelly 完整凯利公式
FullKelly:
  description: "完整凯利公式计算最优仓位（kelly_fraction=1.0）"
  params:
    lookback_period: 120    # 计算统计量的回看周期（天数）
    risk_free_rate: 0.02    # 年化无风险利率（默认2%）
    min_periods: 60         # 最小有效数据点数
    max_position: 0.95      # 最大仓位比例（95%）
    min_position: 0.0       # 最小仓位比例（0%）
  enabled: true

# Half Kelly 半凯利公式
HalfKelly:
  description: "半凯利公式计算仓位（kelly_fraction=0.5）"
  params:
    lookback_period: 120
    risk_free_rate: 0.02
    min_periods: 60
    max_position: 0.95
    min_position: 0.0
  enabled: true

# Quarter Kelly 四分之一凯利公式
QuarterKelly:
  description: "四分之一凯利公式计算仓位（kelly_fraction=0.25）"
  params:
    lookback_period: 120
    risk_free_rate: 0.02
    min_periods: 60
    max_position: 0.95
    min_position: 0.0
  enabled: true
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lookback_period` | int | 120 | 计算期望收益率和方差的回看周期（天数） |
| `risk_free_rate` | float | 0.02 | 年化无风险利率（2%） |
| `min_periods` | int | 60 | 最小有效数据点数，低于此值将回退到最大仓位 |
| `max_position` | float | 0.95 | 最大仓位比例（95%），防止过度杠杆 |
| `min_position` | float | 0.0 | 最小仓位比例（0%），可设置为正数以保持最低仓位 |

## 使用方式

### 1. 在代码中直接使用

```python
from backtest.pipeline.capital import (
    FullKellyManager,
    HalfKellyManager,
    QuarterKellyManager
)

# 创建管理器
full_kelly = FullKellyManager(
    lookback_period=120,
    risk_free_rate=0.02,
    max_position=0.95
)

half_kelly = HalfKellyManager(
    lookback_period=120,
    risk_free_rate=0.02,
    max_position=0.80  # 更保守的最大仓位
)

quarter_kelly = QuarterKellyManager(
    lookback_period=120,
    risk_free_rate=0.02,
    max_position=0.70
)

# 计算资金分配
# 需要提供账户总价值和数据对象列表
total_value = 100000.0  # 账户总价值
data_objects = [...]     # backtrader数据对象列表

allocation = half_kelly.get_allocation(total_value, data_objects)
print(f"分配资金: {allocation:,.2f}")
```

### 2. 使用便捷函数

```python
from backtest.pipeline.capital import (
    create_full_kelly_manager,
    create_half_kelly_manager,
    create_quarter_kelly_manager
)

# 使用默认参数创建
manager = create_half_kelly_manager()

# 自定义参数
manager = create_half_kelly_manager(
    lookback_period=90,
    risk_free_rate=0.015,
    max_position=0.85
)
```

## 三种凯利公式的对比

| 类型 | 凯利分数 | 波动性 | 回撤 | 收益率 | 适用场景 |
|------|----------|--------|------|--------|----------|
| Full Kelly | 1.0 | 最高 | 最大 | 最高 | 对模型有极高信心，能承受大波动 |
| Half Kelly | 0.5 | 中等 | 中等 | 略低于Full | **推荐使用**，平衡收益与风险 |
| Quarter Kelly | 0.25 | 最低 | 最小 | 较低 | 风险厌恶，追求稳定 |

## 特性说明

### 1. 动态仓位调整
凯利公式会根据市场状况动态调整仓位：
- 期望收益率高、波动率低 → 增加仓位
- 期望收益率低、波动率高 → 减少仓位
- 期望收益率 < 无风险利率 → 建议不持仓（仓位为0）

### 2. 仓位约束
计算出的凯利仓位会被约束在 `[min_position, max_position]` 范围内：
- 如果计算出的仓位 > `max_position`，则使用 `max_position`
- 如果计算出的仓位 < `min_position`，则使用 `min_position`
- 如果计算出负仓位，会记录警告并约束到 `min_position`

### 3. 数据不足处理
如果历史数据不足（少于 `min_periods`），管理器会：
- 记录警告日志
- 回退到使用 `max_position` 作为仓位

### 4. 异常处理
管理器会处理各种异常情况：
- 总价值无效（≤0, NaN, Inf）→ 抛出 ValueError
- 数据对象为空 → 使用 `max_position`
- 方差无效（≤0, NaN, Inf）→ 返回 None，回退到 `max_position`

## 实战建议

### 1. 参数选择建议

**保守策略（Quarter Kelly）**
```yaml
QuarterKelly:
  params:
    lookback_period: 180     # 更长的回看周期，更稳定
    risk_free_rate: 0.025    # 可以适当提高无风险利率
    max_position: 0.70       # 较低的最大仓位
```

**平衡策略（Half Kelly，推荐）**
```yaml
HalfKelly:
  params:
    lookback_period: 120     # 标准回看周期
    risk_free_rate: 0.02     # 标准无风险利率
    max_position: 0.85       # 适中的最大仓位
```

**激进策略（Full Kelly）**
```yaml
FullKelly:
  params:
    lookback_period: 60      # 较短的回看周期，更敏感
    risk_free_rate: 0.015    # 可以适当降低无风险利率
    max_position: 0.95       # 较高的最大仓位
```

### 2. 注意事项

1. **参数敏感性**: 凯利公式对期望收益率和方差的估计非常敏感。估计误差会导致仓位偏离最优值。

2. **历史数据质量**: 确保有足够长且质量好的历史数据。建议 `lookback_period` 至少为60天。

3. **市场环境**: 凯利公式假设历史统计量能够代表未来。在市场剧烈变化时需要谨慎使用。

4. **仓位上限**: 强烈建议设置合理的 `max_position`（如0.95），防止过度杠杆。

5. **分数选择**:
   - 初学者 → Quarter Kelly
   - 大多数情况 → Half Kelly（推荐）
   - 高度自信 → Full Kelly（谨慎使用）

### 3. 与其他资金管理器的配合

凯利公式管理器可以与其他组件配合使用：

```python
# 示例：结合选股器、分配器和资金管理器
from backtest.pipeline.selectors import TopNSelector
from backtest.pipeline.allocators import MaxSharpeAllocator
from backtest.pipeline.capital import HalfKellyManager

# 1. 选股：选出前10只股票
selector = TopNSelector(n=10)
selected_stocks = selector.select(all_stocks, factors)

# 2. 分配权重：使用最大夏普比分配
allocator = MaxSharpeAllocator()
weights = allocator.allocate(selected_stocks)

# 3. 资金管理：使用半凯利公式确定总仓位
capital_manager = HalfKellyManager()
total_allocation = capital_manager.get_allocation(total_value, selected_stocks)

# 4. 计算每只股票的实际分配金额
for stock, weight in weights.items():
    stock_allocation = total_allocation * weight
    # 执行交易...
```

## 理论背景

凯利公式由贝尔实验室的 John Kelly 在1956年提出，最初用于信息论中的最优信号传输。后来被应用于赌博和投资领域。

**核心思想**: 最大化长期资本增长率的期望值。

**关键性质**:
1. 理论上最优：长期来看，没有其他策略能够超越凯利公式
2. 防止破产：凯利公式永远不会导致完全亏损
3. 参数敏感：对期望收益和方差的估计误差非常敏感

**为什么使用分数凯利**:
- Full Kelly 理论最优，但实际中波动过大
- 估计误差会导致实际仓位偏离最优
- Half Kelly 或 Quarter Kelly 提供更好的风险调整收益
- 实践表明，Half Kelly 是大多数情况下的最佳选择

## 参考资料

1. Kelly, J. L. (1956). "A New Interpretation of Information Rate". Bell System Technical Journal.
2. Thorp, E. O. (2006). "The Kelly Criterion in Blackjack Sports Betting, and the Stock Market".
3. MacLean, L. C., Thorp, E. O., & Ziemba, W. T. (2011). "The Kelly Capital Growth Investment Criterion".

## 常见问题

**Q: 为什么计算出的仓位是负数？**
A: 这表明期望收益率低于无风险利率，凯利公式建议不持仓。管理器会将仓位约束到 `min_position`。

**Q: 三种凯利公式应该选哪一种？**
A: 大多数情况下推荐使用 Half Kelly，它在收益和风险之间取得了良好平衡。

**Q: 如何处理多资产组合？**
A: 当前实现简化处理为等权重组合的平均收益率和方差。未来可以扩展支持协方差矩阵。

**Q: `lookback_period` 应该设置为多少？**
A: 建议120-180天。太短会导致估计不稳定，太长会对市场变化反应迟钝。

**Q: 能否动态调整凯利分数？**
A: 可以在不同市场环境下使用不同的凯利分数。例如，市场平稳时用 Half Kelly，波动加剧时用 Quarter Kelly。
