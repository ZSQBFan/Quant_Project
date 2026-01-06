# Quant Project

> [!WARNING]
> **项目状态**：本项目目前处于**极早期开发阶段**。
> - 核心功能正在频繁重构，API 极其不稳定。
> - **存在大量已知和未知的 Bug**，计算逻辑尚未经过严格的生产环境验证。
> - 严禁直接用于实盘交易，作者不对任何因使用本项目造成的资金损失负责。

一个基于 Python 的量化交易系统，支持因子研究、策略回测和投资组合分析。

## 功能特性

- **多数据源支持** - Tushare、Akshare、本地 SQLite 数据库
- **因子系统** - 30+ 预定义因子（技术面+基本面），支持行业中性处理
- **智能因子合成** - LightGBM机器学习、滚动优化、动态加权等多种策略
- **回测引擎** - 基于 Backtrader 的事件驱动回测框架，支持自动调仓
- **Kelly Criterion资金管理** - 基于凯利公式的仓位管理系统
- **高级组合优化** - 最大夏普率、最小方差、风险平价等分配策略
- **可插拔架构** - 装饰器自动注册，易于扩展新组件
- **配置驱动** - YAML 配置文件，参数化管理

## 项目结构

```
quant_project_3.12_macmini/
├── core/                   # 核心框架层
│   ├── registry.py         # 组件注册表
│   ├── loader.py           # 动态模块加载器
│   ├── config.py           # 配置管理
│   └── abstractions.py     # 抽象基类
├── data/                   # 数据层
│   ├── manager.py          # 数据提供管理器
│   ├── calendar.py         # 交易日历
│   ├── providers/          # 数据源驱动
│   └── handlers/           # 数据库处理器
├── factors/                # 因子系统
│   ├── library/            # 因子库
│   │   ├── momentum.py     # 技术面因子（动量、反转等）
│   │   ├── rsi.py          # 技术指标（RSI、MACD、KDJ等）
│   │   ├── ep.py           # 基本面因子（EP、BP、ROE等）
│   │   ├── complex_factor_base.py  # 复合因子基类
│   │   └── industry_neutral/  # 行业中性因子目录
│   ├── pipeline/           # 处理流水线
│   │   ├── combiners/      # 因子合成器（LightGBM、滚动优化等）
│   │   ├── standardizers/  # 标准化器（ZScore、Rank等）
│   │   └── rolling/        # 滚动计算器
│   └── analysis/           # 因子分析
├── backtest/               # 回测系统
│   ├── core/               # 回测核心
│   ├── pipeline/           # 回测流水线
│   │   ├── selectors/      # 选股器（TopN、行业中性等）
│   │   ├── allocators/     # 权重分配器（最大夏普、最小方差等）
│   │   └── capital/        # 资金管理器（Kelly公式等）
│   └── triggers/           # 触发器（自动调仓、止损等）
├── configs/                # 配置文件
│   ├── universe.yaml       # 股票池配置
│   ├── data/               # 数据配置
│   ├── factors/            # 因子配置
│   └── backtest/           # 回测配置
├── scripts/                # 执行脚本
├── database/               # 数据库存储
├── output/                 # 输出目录
│   ├── logs/               # 日志文件
│   ├── factor_reports/     # 因子分析报告
│   └── bt_reports/         # 回测报告
└── test/                   # 测试文件
```

## 环境要求

- Python 3.12+
- uv (包管理器)

## 安装

```bash
# 克隆项目
git clone <repository-url>
cd quant_project_3.12_macmini

# 安装依赖
uv sync
```

## 配置

### 数据源配置

编辑 `configs/data/config.yaml`:

```yaml
start_date: 2023-01-01
end_date: 2024-01-01
database:
  db_path: "./database/quant_data.db"
download:
  num_checker_threads: 16
  num_downloader_threads: 16
  batch_size: 200
provider_priority:
  - sqlite
```

如需使用 Tushare，请设置 token:

```yaml
tushare:
  token: "your_tushare_token"
```

### 股票池配置

编辑 `configs/universe.yaml` 配置股票池（默认沪深300成分股）。

## 使用方法

### 数据下载

```bash
python main.py --mode download_data
```

### 因子分析

```bash
python main.py --mode factor_analysis
```

### 策略回测

```bash
python main.py --mode backtest
```

### 查看注册组件

```bash
python main.py --mode list_components
```

## 因子系统

### 预定义因子

**技术面因子**:
- Momentum - 动量因子（价格变化率）
- Reversal20D - 20日反转因子（加权反转）
- RSI - 相对强弱指标（14日）
- MACD - 移动平均收敛发散指标
- KDJ - 随机指标
- BollingerBands - 布林带指标
- MovingAverageCross - 均线交叉因子
- ADXDMI - ADX/DMI趋势强度指标
- VolumeSpike - 成交量突增因子

**基本面因子**:
- EP - 市盈率倒数（盈利收益率）
- BP - 市净率倒数（账面市值比）
- ROE - 净资产收益率
- GPM - 毛利率
- SalesGrowth - 营收同比增长率
- CFOP - 经营现金流市价率
- AssetTurnover - 总资产周转率
- CurrentRatio - 流动比率

**行业中性因子**（上述因子的行业中性版本）:
- IndNeu_Momentum - 行业中性动量
- IndNeu_Reversal20D - 行业中性反转
- IndNeu_EP - 行业中性市盈率倒数
- IndNeu_BP - 行业中性市净率倒数
- IndNeu_ROE - 行业中性净资产收益率
- IndNeu_GPM - 行业中性毛利率
- IndNeu_SalesGrowth - 行业中性营收增长
- IndNeu_CFOP - 行业中性经营现金流
- IndNeu_AssetTurnover - 行业中性资产周转率
- IndNeu_CurrentRatio - 行业中性流动比率
- IndNeu_VolumeCV - 行业中性成交量变异系数

### 因子合成方式

**静态策略**:
- `EqualWeights` - 等权合成（所有因子权重相同）
- `FixedWeights` - 固定权重合成（手动指定各因子权重）
- `DynamicSignificance` - 动态显著性加权（根据因子显著性自动调整）

**滚动优化策略**:
- `RollingICIR` - IC/IR滚动优化（基于历史IC/IR动态调整权重）
- `RollingRegression` - 回归滚动优化（多因子回归求解最优权重）

**机器学习策略**:
- `LightGBM_Periodic` - LightGBM模型（周期性重新训练的梯度提升树）

**LLM增强策略**:
- `AdversarialLLM` - 对抗式LLM优化（使用大语言模型辅助权重调整）

### 因子配置

编辑 `configs/factors/config.yaml` 选择使用的因子和合成策略:

```yaml
# 启用的因子列表
enabled_factors:
  # 技术面因子
  - Momentum
  - MACD
  - RSI

  # 基本面因子
  - EP  # 市盈率倒数
  - BP  # 市净率倒数
  - GPM  # 毛利率

  # 行业中性因子
  - IndNeu_Momentum
  - IndNeu_EP

# 因子合成策略
default_strategy: LightGBM_Periodic  # 使用机器学习策略

# 标准化方法
default_standardizer: ZScore  # Z-Score标准化
```

### 添加自定义因子

1. 在 `factors/library/` 中创建新文件:

```python
from factors.library.base import BaseFactor
from core.registry import register_factor

@register_factor("MyFactor")
class MyFactor(BaseFactor):
    def calculate(self, df):
        # 实现因子计算逻辑
        return df['close'].pct_change(periods=10)
```

2. 在配置文件 `configs/factors/config.yaml` 的 `enabled_factors` 列表中添加该因子名称

## 回测系统

### 回测配置

编辑 `configs/backtest/config.yaml`:

```yaml
broker:
  initial_cash: 1000000  # 初始资金
  commission:
    rate: 0.003  # 佣金费率（万分之十）

selector: TopN  # 选股器
allocator: MaxSharpe  # 权重分配器
capital_manager: FullKelly  # 资金管理器

triggers:
  - type: RebalanceDay  # 自动调仓触发器
    config_file: rebalance.yaml
  - type: StopLoss  # 止损触发器
    config_file: stop_loss.yaml

benchmark: "600519"  # 基准股票
```

### 回测组件

**选股器 (Selector)**:
- `TopN` - 选择因子值最高的 N 只股票
- `IndustryNeutral` - 行业中性选股（各行业均衡选择）

**权重分配器 (Allocator)**:
- `EqualWeight` - 等权分配
- `MaxSharpe` - 最大夏普率优化（基于历史协方差矩阵）
- `MinimumVariance` - 最小方差优化（降低组合波动性）
- `RiskParity` - 风险平价（各资产风险贡献相等）

**资金管理器 (Capital Manager)**:
- `FullPosition` - 全仓管理（固定仓位）
- `FullKelly` - 完全凯利公式（最优增长率，高风险）
- `HalfKelly` - 半凯利公式（平衡收益与风险）
- `QuarterKelly` - 四分之一凯利公式（保守策略）

**触发器 (Trigger)**:
- `RebalanceDay` - 自动定期调仓（支持按月、按周等频率）
- `StopLoss` - 止损触发（设定阈值自动止损）

### 高级特性

**Kelly Criterion 资金管理**

Kelly公式根据预期收益率和风险动态调整仓位，实现长期资本最优增长：

- `FullKelly` - 使用完整凯利公式计算的最优仓位（f* = μ/σ²），适合风险偏好型投资者
- `HalfKelly` - 使用一半凯利仓位（f*/2），在收益与风险之间取得平衡
- `QuarterKelly` - 使用四分之一凯利仓位（f*/4），更加保守稳健

**组合优化算法**

基于现代投资组合理论的多种优化策略：

- `MaxSharpe` - 最大化夏普率（收益风险比），寻找风险调整后收益最优的组合
- `MinimumVariance` - 最小化组合方差，降低投资组合波动性
- `RiskParity` - 风险平价配置，使各资产对组合风险的贡献相等

**自动调仓机制**

回测系统支持灵活的自动调仓配置：
- 无需手动指定调仓日期
- 支持按月、按周、按日等多种调仓频率
- 与触发器系统配合，可实现复杂的调仓逻辑

## 主要依赖

| 依赖 | 用途 |
|------|------|
| tushare | Tushare 金融数据 API |
| akshare | Akshare 免费财经数据 |
| backtrader | 事件驱动回测框架 |
| pandas | 数据分析和处理 |
| numpy | 数值计算 |
| lightgbm | AI 因子合成 |
| scikit-learn | 机器学习 |
| matplotlib | 数据可视化 |
| plotly | 交互式绘图 |
| pyyaml | 配置解析 |

## 输出报告

- **因子分析报告**: `output/factor_reports/`
  - IC、Rank IC分析
  - 因子收益率分析
  - 分层回测结果
  - 因子相关性矩阵
  - HTML交互式报告（可选）

- **回测报告**: `output/bt_reports/`
  - 每日持仓明细
  - 交易记录（买入/卖出）
  - 投资组合净值曲线
  - 业绩指标（年化收益率、夏普率、最大回撤等）
  - 与基准的对比分析

- **日志文件**: `output/logs/` - 详细的运行日志

## 开发

### 运行测试

```bash
pytest test/
```

### 项目架构

系统采用可插拔的组件架构:

```
┌─────────────────────────────────────────────────────────┐
│                      main.py                            │
├─────────────────────────────────────────────────────────┤
│  core/registry.py (组件注册表)                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   data/     │  │  factors/   │  │  backtest/  │     │
│  │  数据管理   │ → │  因子计算   │ → │  策略回测   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
├─────────────────────────────────────────────────────────┤
│                   configs/ (YAML)                       │
└─────────────────────────────────────────────────────────┘
```

## License

MIT
