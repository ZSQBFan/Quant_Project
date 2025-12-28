# 选股器模块使用说明

## 概述

选股器模块提供了多种股票筛选策略，用于从候选股票池中选择符合条件的股票。

## 可用选股器

### 1. TopNSelector - Top N 选股器

根据因子信号强度选择排名前 N 的股票。

**特点：**
- 简单直接，按信号强度排序
- 适用于纯因子驱动策略
- 不考虑行业分布

**使用示例：**
```python
from backtest.pipeline.selectors import TopNSelector

selector = TopNSelector(top_n=10)
selected_stocks = selector.select(datas)
```

**配置文件示例：**
```yaml
pipeline:
  selector:
    class: "TopNSelector"
    params:
      n: 10
```

### 2. IndustryNeutralSelector - 行业中性化选股器

在各行业间均衡选股，降低行业集中风险。

**特点：**
- 控制行业暴露，避免行业集中风险
- 支持多种分配策略
- 自动过滤停牌和无效信号股票
- 需要行业数据支持

**三种分配策略：**

#### (1) 等权分配策略 (equal)
每个行业选择相同数量的股票。

**使用场景：**
- 追求行业完全中性
- 各行业权重相等

**示例：**
```python
selector = IndustryNeutralSelector(
    total_stocks=10,
    strategy='equal',
    stocks_per_industry=2  # 每个行业2只
)
```

**配置文件：**
```yaml
selector:
  class: "IndustryNeutralSelector"
  params:
    total_stocks: 10
    strategy: "equal"
    stocks_per_industry: 2
```

#### (2) 比例分配策略 (proportional)
按各行业候选股票数量比例分配。

**使用场景：**
- 行业权重与市场结构一致
- 大行业多选，小行业少选

**示例：**
```python
selector = IndustryNeutralSelector(
    total_stocks=10,
    strategy='proportional',
    min_stocks_per_industry=1
)
```

**配置文件：**
```yaml
selector:
  class: "IndustryNeutralSelector"
  params:
    total_stocks: 10
    strategy: "proportional"
    min_stocks_per_industry: 1
```

#### (3) 头部行业策略 (top_industries)
只在信号最强的 N 个行业中选股。

**使用场景：**
- 行业轮动策略
- 集中投资强势行业

**示例：**
```python
selector = IndustryNeutralSelector(
    total_stocks=10,
    strategy='top_industries',
    top_industries=3  # 只在前3个行业选股
)
```

**配置文件：**
```yaml
selector:
  class: "IndustryNeutralSelector"
  params:
    total_stocks: 10
    strategy: "top_industries"
    top_industries: 3
```

## 参数说明

### IndustryNeutralSelector 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `total_stocks` | int | 是 | 10 | 总共选择的股票数量 |
| `strategy` | str | 是 | 'equal' | 分配策略：equal/proportional/top_industries |
| `stocks_per_industry` | int | 否 | None | 每个行业选择的数量（仅 equal 策略） |
| `top_industries` | int | 否 | None | 选择的行业数量（仅 top_industries 策略） |
| `min_stocks_per_industry` | int | 否 | 1 | 每个行业最少选择数量 |
| `name` | str | 否 | None | 选股器名称 |

## 数据要求

### 行业数据支持

IndustryNeutralSelector 需要股票数据中包含行业信息。系统已自动配置以下功能：

1. **数据导出阶段** (`BTDataExporter`)
   - 自动从数据库加载行业信息
   - 将行业字段添加到导出的股票数据中

2. **数据加载阶段** (`FactorPandasData`)
   - 自动读取行业字段
   - 将行业信息传递给选股器

3. **选股阶段** (`IndustryNeutralSelector`)
   - 从股票数据中读取行业信息
   - 按行业分组进行选股

**无需额外配置！** 只要数据库中有行业数据（`stock_kind` 表），系统会自动处理。

## 完整使用流程

### 1. 修改配置文件

配置分为两个层级：

**第一步：选择使用哪个选股器**

编辑 `configs/backtest/config.yaml`：

```yaml
# 选股器（详见 pipeline/selectors.yaml）
# 可选值: TopN, IndustryNeutral
selector: IndustryNeutral  # 从 TopN 改为 IndustryNeutral
```

**第二步：配置选股器参数**

编辑 `configs/backtest/pipeline/selectors.yaml`：

```yaml
IndustryNeutral:
  description: "行业中性化选股器，在各行业间均衡选股"
  params:
    total_stocks: 10
    strategy: "equal"
    stocks_per_industry: 2  # 可选，仅 equal 策略需要
```

### 2. 运行回测

```bash
python -m scripts.run_backtest
```

### 3. 查看日志

系统会输出详细的选股过程，包括：
- 行业分布统计
- 每个行业选中的股票数量
- 被过滤的股票统计（停牌、NaN等）

**示例日志：**
```
🔧 组装策略组件...
  选股器: IndustryNeutralSelector (总数: 10, 策略: equal)

行业分布统计:
  总行业数: 5
  覆盖行业数: 5
    金融: 2 只 (总共 15 只)
    消费: 2 只 (总共 12 只)
    科技: 2 只 (总共 10 只)
    医药: 2 只 (总共 8 只)
    工业: 2 只 (总共 6 只)
```

## 常见问题

### Q1: 如果某个行业候选股票不足怎么办？

A: 选股器会自动调整，从其他行业补充股票，确保达到目标数量。

### Q2: 如何查看行业分布？

A: 启用 DEBUG 日志级别，会输出详细的行业分布信息。

### Q3: 停牌股票会被选中吗？

A: 不会。选股器会自动过滤停牌股票、信号为 NaN 的股票、以及缺少行业信息的股票。

### Q4: 如果数据库没有行业数据怎么办？

A: 如果缺少行业数据，IndustryNeutralSelector 会过滤掉所有无行业信息的股票。建议在这种情况下使用 TopNSelector。

## 性能优化建议

1. **数据预加载**：系统已自动批量加载行业数据，无需额外优化
2. **内存占用**：行业字段为字符串类型，占用内存较小
3. **选股速度**：行业中性化选股比 TopN 稍慢，但在可接受范围内（毫秒级）

## 测试

运行单元测试验证选股器功能：

```bash
python test/test_industry_neutral_selector.py
```

测试覆盖：
- ✅ 等权分配策略
- ✅ 比例分配策略
- ✅ 头部行业策略
- ✅ 停牌和无效信号过滤

## 扩展建议

未来可以考虑添加的选股器：

1. **市值中性化选股器**：在不同市值区间均衡选股
2. **因子得分阈值选股器**：只选择因子得分超过阈值的股票
3. **多因子组合选股器**：结合多个因子进行综合评分
4. **动态调整选股器**：根据市场状态动态调整选股策略

## 参考

- 选股器基类：`backtest/pipeline/selectors/top_n.py:SelectorBase`
- TopN 实现：`backtest/pipeline/selectors/top_n.py:TopNSelector`
- 行业中性化实现：`backtest/pipeline/selectors/industry_neutral.py:IndustryNeutralSelector`
- 配置示例：`configs/backtest/pipeline/selectors.yaml`
