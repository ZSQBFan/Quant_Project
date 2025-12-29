# 权重分配器使用指南

本文档介绍回测系统中所有可用的权重分配器及其使用场景。

---

## 概览

| 分配器名称 | 配置名称 | 核心逻辑 | 适用场景 |
|-----------|---------|---------|---------|
| **等权重** | `EqualWeight` | 所有股票权重相等 | 基准策略、简单分散 |
| **波动率倒数加权** | `RiskParity` / `InverseVolatility` | 权重与波动率成反比 | 混合波动率股票池 |
| **最大夏普比** | `MaxSharpe` | 优化风险调整后收益 | 追求收益风险比 |
| **全局最小方差** | `MinimumVariance` | 利用相关性对冲风险 | 追求低波动 |

---

## 1. EqualWeight - 等权重分配器

### 核心逻辑
所有选中的股票分配相等的权重：`weight_i = 1/N`

### 适用场景
- 作为回测的基准策略
- 简单的风险分散
- 不确定各股票风险收益特征时

### 配置示例
```yaml
# configs/backtest/config.yaml
allocator: EqualWeight
```

### 特点
- ✅ 最简单直观
- ✅ 不需要历史数据
- ✅ 计算速度快
- ⚠️ 不考虑个股风险差异
- ⚠️ 高波动股票可能主导组合风险

---

## 2. RiskParity / InverseVolatility - 波动率倒数加权

### 核心逻辑
**谁稳谁仓位重，谁浪谁仓位轻**

权重与历史波动率成反比：
```
weight_i = (1/σ_i) / Σ(1/σ_j)
```

其中 `σ_i` 是股票 i 的历史波动率（收益率标准差）。

### 适用场景

**痛点场景**：
你的 Top N 选股列表里混杂了"稳健的银行股"和"暴躁的科技股"。如果等权重买入，组合的波动率会被那几只高波动的股票主导，一旦高波动股大跌，组合回撤就很大。

**解决方案**：
- 银行股波动率 10%，科技股波动率 30%
- 模型会自动给银行股分配 3 倍于科技股的仓位
- 结果：整个组合的波动率被拉平，降低"单只妖股暴雷"带来的回撤冲击

### 配置示例
```yaml
# configs/backtest/config.yaml
allocator: RiskParity  # 或 InverseVolatility

# configs/backtest/pipeline/allocators.yaml
RiskParity:
  params:
    lookback_period: 60      # 计算波动率的回看周期
    min_periods: 30          # 最小有效数据点数
    volatility_floor: 0.0001 # 波动率下限，防止除零
```

### 参数说明
- `lookback_period`: 计算波动率使用的历史天数（默认60天）
- `min_periods`: 如果有效数据点少于此值，回退到等权重
- `volatility_floor`: 波动率的最小值，防止极低波动股票获得过高权重

### 特点
- ✅ 降低组合整体波动率
- ✅ 避免高波动股票主导组合风险
- ✅ 计算简单高效
- ✅ 适合波动率差异大的股票池
- ⚠️ 不考虑股票间相关性
- ⚠️ 可能降低收益率（低波动股票通常收益也较低）

### 使用建议
- 当选股列表中股票波动率差异较大时（如 >2倍）效果显著
- 适合风险厌恶型投资者
- 建议 `lookback_period` 设置为 30-90 天

---

## 3. MaxSharpe - 最大夏普比分配器

### 核心逻辑
通过数值优化找到使夏普比率最大化的权重组合：

```
maximize: (μ^T w - r_f) / sqrt(w^T Σ w)
```

其中：
- `μ`: 期望收益率向量
- `w`: 权重向量
- `r_f`: 无风险利率
- `Σ`: 协方差矩阵

### 适用场景
- 追求风险调整后收益最大化
- 理性的资产配置
- 均值-方差优化框架

### 配置示例
```yaml
# configs/backtest/config.yaml
allocator: MaxSharpe

# configs/backtest/pipeline/allocators.yaml
MaxSharpe:
  params:
    lookback_period: 120    # 计算统计量的回看周期
    risk_free_rate: 0.02    # 年化无风险利率（2%）
    min_periods: 60         # 最小有效数据点数
    max_weight: 0.3         # 单只股票的最大权重（30%）
```

### 参数说明
- `lookback_period`: 计算收益率和协方差矩阵的历史天数（默认120天）
- `risk_free_rate`: 年化无风险利率，用于计算超额收益（默认0.02，即2%）
- `min_periods`: 最小有效数据点数
- `max_weight`: 单只股票的最大权重，防止过度集中（默认0.3，即30%）

### 特点
- ✅ 理论基础扎实（现代投资组合理论）
- ✅ 同时考虑收益和风险
- ✅ 考虑股票间相关性
- ⚠️ 对历史数据敏感（参数估计误差）
- ⚠️ 可能出现权重集中在少数股票
- ⚠️ 计算成本较高（优化求解）

### 使用建议
- 需要较长的历史数据（建议至少90天）
- 设置合理的 `max_weight` 限制（建议 0.2-0.4）
- 适合追求收益风险比的投资者
- 在牛市中表现可能优于其他方法

---

## 4. MinimumVariance - 全局最小方差分配器 (GMV)

### 核心逻辑
**买两个负相关的股票，比不买更安全**

通过求解二次规划问题找到使组合方差最小的权重：

```
minimize: w^T Σ w
subject to: Σw_i = 1, w_i ≥ 0
```

### 适用场景

**痛点场景**：
Top N 可能会选出 10 只全是半导体的股票。虽然它们个股波动率可能不高，但因为相关性极高（齐涨齐跌），组合并没有真正分散风险。

**解决方案**：
- 模型发现：虽然股票 A 评分不是最高，但它和手里已有的股票 B 走势完全相反（负相关）
- 于是模型决定买入 A
- 这样当 B 跌的时候，A 会涨，从而抵消亏损
- 结果：净值曲线变得更加平滑

### 配置示例
```yaml
# configs/backtest/config.yaml
allocator: MinimumVariance

# configs/backtest/pipeline/allocators.yaml
MinimumVariance:
  params:
    lookback_period: 120    # 计算协方差矩阵的回看周期
    min_periods: 60         # 最小有效数据点数
    max_weight: 0.3         # 单只股票的最大权重
    regularization: 0.001   # L2正则化系数
```

### 参数说明
- `lookback_period`: 计算协方差矩阵的历史天数（默认120天）
- `min_periods`: 最小有效数据点数
- `max_weight`: 单只股票的最大权重（默认0.3，即30%）
- `regularization`: L2正则化系数，用于提高协方差矩阵的数值稳定性（默认0.001）

### 数学细节
协方差矩阵可能奇异或病态，导致优化不稳定。通过添加正则化项：
```
Σ_reg = Σ + λI
```
其中 `λ` 是 `regularization` 参数，`I` 是单位矩阵。

### 特点
- ✅ 充分利用股票间的相关性
- ✅ 负相关股票可以互相对冲
- ✅ 有效降低组合波动率
- ✅ 适合追求低波动的投资者
- ⚠️ 完全忽略期望收益（可能配置低收益股票）
- ⚠️ 对协方差矩阵估计误差敏感
- ⚠️ 计算成本较高（优化求解）
- ⚠️ 在所有股票高度相关时效果有限

### 使用建议
- 股票池中存在相关性较低的股票时效果最佳
- 适合防守型、低波动策略
- 建议 `lookback_period` 设置为 90-150 天
- 可以设置较小的 `max_weight`（如0.2）以进一步分散
- 在震荡市场中表现通常优于其他方法
- ⚠️ 在单边市场中可能跑输基准

---

## 分配器对比

### 计算复杂度
```
EqualWeight < RiskParity < MaxSharpe ≈ MinimumVariance
```

### 对历史数据的依赖
```
EqualWeight (无) < RiskParity (低) < MaxSharpe (中) < MinimumVariance (高)
```

### 风险控制能力
```
EqualWeight (弱) < RiskParity (中) < MaxSharpe (中) < MinimumVariance (强)
```

### 收益潜力
```
MinimumVariance (低) < RiskParity (中) < EqualWeight (中) < MaxSharpe (高)
```

---

## 实战案例

### 案例1：科技股 + 银行股组合

**股票池**：
- 科技股（5只）：波动率 30-40%，相关性 0.8
- 银行股（5只）：波动率 10-15%，相关性 0.7
- 科技与银行相关性：0.3

**分配器表现**：

| 分配器 | 科技股权重 | 银行股权重 | 组合波动率 | 预期特点 |
|--------|-----------|-----------|-----------|---------|
| EqualWeight | 50% | 50% | 25% | 简单分散 |
| RiskParity | 20% | 80% | 12% | 降低波动 |
| MaxSharpe | 40% | 60% | 18% | 平衡收益风险 |
| MinimumVariance | 15% | 85% | 10% | 最低波动 |

### 案例2：行业分散组合

**股票池**：
- 10只股票来自不同行业
- 波动率相近（20-25%）
- 行业间相关性低（0.2-0.4）

**分配器表现**：

| 分配器 | 权重集中度 | 组合波动率 | 预期特点 |
|--------|-----------|-----------|---------|
| EqualWeight | 低 | 22% | 均匀分散 |
| RiskParity | 低 | 20% | 略降波动 |
| MaxSharpe | 中 | 18% | 向高夏普股票倾斜 |
| MinimumVariance | 中 | 15% | 充分利用低相关性 |

---

## 快速选择指南

### 我应该用哪个分配器？

**如果你想要...**

1. **简单快速的基准策略** → `EqualWeight`

2. **降低高波动股票的影响** → `RiskParity` / `InverseVolatility`
   - 适合：选股列表波动率差异大
   - 例如：同时包含银行股和科技股

3. **追求最佳的收益风险比** → `MaxSharpe`
   - 适合：希望在控制风险的同时最大化收益
   - 例如：牛市环境，追求超额收益

4. **最低波动的稳健组合** → `MinimumVariance`
   - 适合：风险厌恶，追求平滑净值曲线
   - 例如：震荡市场，重点防守

### 市场环境选择

| 市场环境 | 推荐分配器 | 理由 |
|---------|-----------|------|
| 单边上涨（牛市） | MaxSharpe | 追求收益最大化 |
| 单边下跌（熊市） | MinimumVariance | 降低回撤 |
| 震荡市场 | MinimumVariance | 对冲波动 |
| 趋势不明 | RiskParity | 平衡风险 |
| 初次测试 | EqualWeight | 简单基准 |

---

## 配置切换示例

在 `configs/backtest/config.yaml` 中修改：

```yaml
# 选项1: 等权重（默认）
allocator: EqualWeight

# 选项2: 波动率倒数加权
allocator: RiskParity

# 选项3: 最大夏普比
allocator: MaxSharpe

# 选项4: 全局最小方差
allocator: MinimumVariance
```

详细参数在 `configs/backtest/pipeline/allocators.yaml` 中调整。

---

## 高级技巧

### 1. 组合使用（滚动切换）
可以根据市场环境动态切换分配器：
- 牛市：MaxSharpe
- 熊市：MinimumVariance
- 震荡：RiskParity

### 2. 参数优化
- `lookback_period`: 可以测试 30/60/90/120 天
- `max_weight`: 可以测试 0.2/0.3/0.4
- `regularization`: MinimumVariance 的稳定性调节

### 3. 回测验证
建议同时测试多个分配器，对比：
- 年化收益率
- 最大回撤
- 夏普比率
- 卡玛比率
- 净值曲线平滑度

---

## 注意事项

1. **数据要求**：
   - RiskParity: 至少 30 天历史数据
   - MaxSharpe: 至少 60 天历史数据
   - MinimumVariance: 至少 60 天历史数据

2. **回退机制**：
   所有需要历史数据的分配器在数据不足时会自动回退到等权重

3. **性能考虑**：
   - EqualWeight: 几乎无计算成本
   - RiskParity: 低计算成本
   - MaxSharpe / MinimumVariance: 中等计算成本（优化求解）

4. **参数敏感性**：
   - MaxSharpe 和 MinimumVariance 对历史数据敏感
   - 建议使用足够长的历史窗口（≥90天）
   - 可以通过参数扫描找到最优配置

---

## 总结

| 目标 | 推荐分配器 |
|-----|-----------|
| 基准策略 | EqualWeight |
| 降低波动 | RiskParity → MinimumVariance |
| 提升夏普 | MaxSharpe |
| 防守为主 | MinimumVariance |
| 进攻为主 | MaxSharpe |
| 平衡配置 | RiskParity |

选择合适的分配器需要结合：
- 投资目标（收益 vs 风险）
- 市场环境（牛市 vs 熊市）
- 股票池特征（波动率、相关性）
- 计算资源（实时性要求）

建议通过回测验证不同分配器的表现，选择最适合你策略的方案。
