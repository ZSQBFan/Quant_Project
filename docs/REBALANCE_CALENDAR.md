# 动态调仓日历配置指南

## 概述

动态调仓日历系统允许您根据回测的日期范围自动生成调仓日期，无需手动维护固定的日期列表。该系统支持多种调仓频率模式，提供了灵活且易于配置的调仓策略。

## 特性

- ✅ **自动生成**: 根据回测开始和结束日期自动生成调仓日历
- ✅ **多种频率**: 支持每日、每周、每月、间隔调仓
- ✅ **交易日历**: 可选使用真实交易日历（排除节假日）
- ✅ **智能调整**: 自动处理周末、月末等边界情况
- ✅ **灵活配置**: 通过 YAML 配置文件轻松修改调仓策略

## 配置方式

### 配置文件位置

调仓配置位于: `configs/backtest/triggers/rebalance.yaml`

### 配置结构

```yaml
RebalanceDay:
  description: "根据频率动态生成调仓日期"
  enabled: true
  params:
    # 必选：调仓频率
    frequency: "monthly"  # daily/weekly/monthly/interval

    # 可选：根据频率选择对应的参数
    monthday: 1        # 每月调仓时使用
    # weekday: 0       # 每周调仓时使用
    # interval_days: 20  # 间隔调仓时使用
```

## 调仓频率模式

### 1. 每日调仓 (daily)

每个交易日都进行调仓。

**配置示例:**
```yaml
RebalanceDay:
  enabled: true
  params:
    frequency: "daily"
```

**生成的调仓日期:** 回测期间的所有工作日（周一到周五）

**使用场景:** 高频策略、日内策略

---

### 2. 每周调仓 (weekly)

每周在指定的某一天进行调仓。

**配置示例:**
```yaml
RebalanceDay:
  enabled: true
  params:
    frequency: "weekly"
    weekday: 0  # 0=周一, 1=周二, ..., 4=周五, 5=周六, 6=周日
```

**weekday 参数说明:**
- `0` = 周一
- `1` = 周二
- `2` = 周三
- `3` = 周四
- `4` = 周五
- `5` = 周六
- `6` = 周日

**生成示例 (每周一):**
```
2024-01-01 (周一)
2024-01-08 (周一)
2024-01-15 (周一)
...
```

**使用场景:** 中频策略、周度轮动策略

---

### 3. 每月调仓 (monthly)

每月在指定的某一天进行调仓。

**配置示例:**
```yaml
RebalanceDay:
  enabled: true
  params:
    frequency: "monthly"
    monthday: 1  # 每月1号
```

**monthday 参数说明:**
- 取值范围: `1-31`
- 如果指定的日期大于当月天数（如2月31号），会自动使用当月最后一天
- 如果指定日期是周末，会自动调整到下一个工作日

**生成示例 (每月1号):**
```
2024-01-01
2024-02-01
2024-03-01
...
```

**特殊处理示例 (每月31号):**
```
2024-01-31  # 1月有31天
2024-02-29  # 2月最多29天(闰年)，调整为29号
2024-03-31  # 3月有31天（如果是周末会调整到周一）
2024-04-30  # 4月只有30天，调整为30号
```

**使用场景:** 月度调仓策略、因子轮动策略

---

### 4. 间隔调仓 (interval)

每隔固定数量的交易日进行调仓。

**配置示例:**
```yaml
RebalanceDay:
  enabled: true
  params:
    frequency: "interval"
    interval_days: 20  # 每20个交易日调仓一次
```

**interval_days 参数说明:**
- 取值范围: `>= 1`
- 从第一个交易日开始计数
- 每隔 N 个交易日选择一个调仓日

**生成示例 (每20个交易日):**
```
2024-01-01  # 第1个交易日
2024-01-29  # 第21个交易日
2024-02-26  # 第41个交易日
...
```

**使用场景:**
- 季度调仓（约60个交易日）
- 双周调仓（约10个交易日）
- 自定义周期策略

---

## 工作原理

### 自动调整机制

1. **周末处理**:
   - 如果调仓日是周末，自动调整到下一个周一
   - 例如: 3月31日(周日) → 4月1日(周一)

2. **月末处理**:
   - 如果指定日期大于当月天数，使用当月最后一天
   - 例如: 2月31号 → 2月28/29号

3. **日期范围**:
   - 只生成在 `[start_date, end_date]` 范围内的日期
   - 调整后超出范围的日期会被排除

### 与交易日历集成

系统支持使用真实的交易日历（排除节假日、停牌日等）:

```python
# 在代码中使用
from backtest.core.rebalance_calendar import create_rebalance_calendar
import pandas as pd

# 创建交易日历（排除周末和节假日）
trading_calendar = pd.bdate_range(start='2024-01-01', end='2024-12-31')

# 生成调仓日历
dates = create_rebalance_calendar(
    start_date='2024-01-01',
    end_date='2024-12-31',
    config={'frequency': 'monthly', 'monthday': 1},
    trading_calendar=trading_calendar
)
```

## 配置示例

### 示例 1: 每月第一个交易日调仓

```yaml
RebalanceDay:
  enabled: true
  params:
    frequency: "monthly"
    monthday: 1
```

### 示例 2: 每周五调仓

```yaml
RebalanceDay:
  enabled: true
  params:
    frequency: "weekly"
    weekday: 4  # 周五
```

### 示例 3: 每季度调仓 (约60个交易日)

```yaml
RebalanceDay:
  enabled: true
  params:
    frequency: "interval"
    interval_days: 60
```

### 示例 4: 每月最后一个交易日调仓

```yaml
RebalanceDay:
  enabled: true
  params:
    frequency: "monthly"
    monthday: 31  # 会自动调整为每月最后一天
```

## 从旧版本迁移

### 旧版本（固定日期列表）

```yaml
# configs/backtest/trading_days.json
{
  "dates": [
    "2024-01-02",
    "2024-02-01",
    "2024-03-01",
    ...
  ]
}
```

### 新版本（动态生成）

```yaml
# configs/backtest/triggers/rebalance.yaml
RebalanceDay:
  enabled: true
  params:
    frequency: "monthly"
    monthday: 1
```

### 迁移步骤

1. 确定您的调仓规律（每日/每周/每月/间隔）
2. 在 `rebalance.yaml` 中配置对应的参数
3. 删除或忽略旧的 `trading_days.json` 文件
4. 运行回测验证调仓日期生成正确

## 调试和验证

### 查看生成的调仓日期

运行回测时，日志会显示生成的调仓配置:

```
🔧 组装策略组件...
  触发器: RebalanceDay (每月1号, 12次)
```

### 使用测试脚本验证

```bash
# 运行调仓日历测试
python test/test_rebalance_calendar.py
```

### 手动验证

```python
from backtest.core.rebalance_calendar import create_rebalance_calendar

# 生成并打印调仓日历
dates = create_rebalance_calendar(
    start_date='2024-01-01',
    end_date='2024-12-31',
    config={'frequency': 'monthly', 'monthday': 1}
)

print(f"生成了 {len(dates)} 个调仓日:")
for date in dates:
    print(f"  - {date}")
```

## 常见问题

### Q: 为什么生成的日期数量和预期不同？

A: 可能的原因:
1. 周末被自动调整，导致某些日期被跳过
2. 调整后的日期超出了 `end_date` 范围
3. 月末日期被调整（如2月31号 → 2月28/29号）

### Q: 如何实现每季度调仓？

A: 使用间隔模式，设置 `interval_days: 60`（一季度约60个交易日）

### Q: 如何实现每月第一个周一调仓？

A: 目前系统不直接支持此模式，建议:
1. 使用每周调仓 (`frequency: weekly, weekday: 0`)
2. 或手动实现自定义调仓逻辑

### Q: 可以混合使用多种调仓策略吗？

A: 当前一次只能使用一种调仓频率。如需复杂策略，可以:
1. 在代码中组合多个 RebalanceCalendar 的结果
2. 实现自定义触发器

## 技术细节

### 核心类: RebalanceCalendar

位置: `backtest/core/rebalance_calendar.py`

```python
class RebalanceCalendar:
    def __init__(
        self,
        start_date: str,
        end_date: str,
        frequency: Literal["daily", "weekly", "monthly", "interval"],
        weekday: Optional[int] = None,
        monthday: Optional[int] = None,
        interval_days: Optional[int] = None,
        trading_calendar: Optional[pd.DatetimeIndex] = None
    ):
        ...

    def generate(self) -> List[str]:
        """生成调仓日期列表"""
        ...
```

### 辅助函数: create_rebalance_calendar

```python
def create_rebalance_calendar(
    start_date: str,
    end_date: str,
    config: dict,
    trading_calendar: Optional[pd.DatetimeIndex] = None
) -> List[str]:
    """根据配置创建调仓日历"""
    ...
```

## 最佳实践

1. **月度策略**: 使用 `frequency: monthly` 配合合适的 `monthday`
2. **高频策略**: 使用 `frequency: daily` 或小间隔的 `interval`
3. **避免过度交易**: 根据策略容量选择合适的调仓频率
4. **测试验证**: 修改配置后运行测试确保日期生成正确
5. **日志检查**: 运行回测时检查日志中的调仓次数是否符合预期

## 参考

- 测试文件: `test/test_rebalance_calendar.py`
- 配置文件: `configs/backtest/triggers/rebalance.yaml`
- 核心代码: `backtest/core/rebalance_calendar.py`
- 集成代码: `scripts/run_backtest.py`
