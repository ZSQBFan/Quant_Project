# Quant Project

> [!WARNING]
> **项目状态**：本项目目前处于**极早期开发阶段**。
> - 核心功能正在频繁重构，API 极其不稳定。
> - **存在大量已知和未知的 Bug**，计算逻辑尚未经过严格的生产环境验证。
> - 严禁直接用于实盘交易，作者不对任何因使用本项目造成的资金损失负责。

一个基于 Python 的量化交易系统，支持因子研究、策略回测和投资组合分析。

## 功能特性

- **多数据源支持** - Tushare、Akshare、本地 SQLite 数据库
- **因子系统** - 20+ 预定义因子，支持行业中性处理
- **回测引擎** - 基于 Backtrader 的事件驱动回测框架
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
│   │   ├── momentum.py     # 动量因子
│   │   ├── rsi.py          # RSI 指标
│   │   ├── macd.py         # MACD 指标
│   │   └── industry_neutral/  # 行业中性因子
│   ├── pipeline/           # 处理流水线
│   │   ├── combiners/      # 因子合成器
│   │   ├── standardizers/  # 标准化器
│   │   └── rolling/        # 滚动计算器
│   └── analysis/           # 因子分析
├── backtest/               # 回测系统
│   ├── core/               # 回测核心
│   ├── pipeline/           # 回测流水线
│   │   ├── selectors/      # 选股器
│   │   ├── allocators/     # 权重分配器
│   │   └── capital/        # 资金管理器
│   └── triggers/           # 触发器
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

**简单因子**:
- Momentum - 动量因子
- Reversal20D - 反转因子
- RSI - 相对强弱指标
- MACD - 移动平均收敛发散
- KDJ - 随机指标
- BollingerBands - 布林带
- MovingAverageCross - 均线交叉
- ADXDMI - 趋势强度指标
- VolumeSpike - 成交量异常

**行业中性因子**:
- IndNeu_Momentum - 行业中性动量
- IndNeu_Reversal20D - 行业中性反转
- IndNeu_EP - 行业中性市盈率
- IndNeu_BP - 行业中性市净率
- IndNeu_ROE - 行业中性 ROE
- IndNeu_GPM - 行业中性毛利率
- IndNeu_SalesGrowth - 行业中性营收增长
- IndNeu_CFOP - 行业中性现金流
- IndNeu_VolumeCV - 行业中性成交量

### 因子合成方式

- `EqualWeightCombiner` - 等权合成
- `FixedWeightCombiner` - 固定权重
- `DynamicWeightCombiner` - 动态权重
- `DynamicSignificanceCombiner` - 基于显著性的动态权重
- `AICombiner` - LightGBM 模型动态权重

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

2. 在配置文件中启用该因子

## 回测系统

### 回测配置

编辑 `configs/backtest/config.yaml`:

```yaml
initial_cash: 1000000.0
commission: 0.0003
selector: TopN
allocator: EqualWeight
capital_manager: FullPosition
triggers:
  - type: RebalanceDay
    params:
      frequency: monthly
  - type: StopLoss
    params:
      threshold: 0.1
```

### 回测组件

**选股器 (Selector)**:
- `TopNSelector` - 选择因子值最高的 N 只股票

**权重分配器 (Allocator)**:
- `EqualWeightAllocator` - 等权分配

**资金管理器 (Capital Manager)**:
- `FullPositionManager` - 全仓管理

**触发器 (Trigger)**:
- `RebalanceDayTrigger` - 定期调仓
- `StopLossTrigger` - 止损触发

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

- **因子分析报告**: `output/factor_reports/` - IC、Rank IC、收益率分析
- **回测报告**: `output/bt_reports/` - 交易记录、投资组合历史、业绩评估
- **日志文件**: `output/logs/`

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
