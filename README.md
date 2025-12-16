# 量化投资分析系统

一个专业的股票量化投资分析框架，支持多因子模型、动态权重策略、机器学习集成和 Backtrader 事件驱动回测。

## 项目概述

本项目是一个完整的量化投资分析系统，专注于因子投资策略的研究与回测。系统采用模块化设计，支持多种因子类型和组合策略，提供全面的分析报告。

## 核心特性

### 多因子模型
- **量价因子**: 动量、反转、RSI、KDJ、布林带、MACD、ADX/DMI 等
- **基本面因子**: EP、BP、ROE、销售增长、现金流等
- **行业中性因子**: 消除行业影响的标准化因子

### 策略框架
- **静态组合策略**: 等权重、固定权重、动态显著性
- **动态滚动策略**: 基于 ICIR、回归分析的动态权重调整
- **机器学习策略**: LightGBM 模型预测
- **对抗性 LLM 策略**: 多智能体因子组合

### 分析功能
- 因子 IC 分析、分组回测、衰减分析
- 多周期收益率预测 (1/5/10/20/30 日)
- 交互式 HTML 报告生成

### 回测系统
- 基于 Backtrader 的事件驱动回测
- 支持止损、定期调仓等触发器
- Top-N 选股器、等权重分配器
- 完整的回测报告生成

## 项目结构

```
quant_project/
├── main.py                     # 统一入口
├── configs/                    # YAML 配置中心
│   ├── factors.yaml           # 因子配置
│   ├── strategies.yaml        # 策略配置
│   ├── backtest.yaml          # 回测配置
│   └── universe.yaml          # 股票池配置
├── core/                       # 框架核心
│   ├── abstractions.py        # 抽象基类
│   ├── registry.py            # 组件注册表
│   ├── loader.py              # 动态模块加载
│   ├── config.py              # 配置加载器
│   └── strategy.py            # 策略配置
├── factors/                    # 因子领域
│   ├── library/               # 因子库（单文件单因子）
│   │   ├── momentum.py        # 动量因子
│   │   ├── rsi.py             # RSI 因子
│   │   ├── macd.py            # MACD 因子
│   │   └── industry_neutral/  # 行业中性因子
│   ├── pipeline/              # 因子处理流水线
│   │   ├── combiners/         # 因子合成器
│   │   ├── standardizers/     # 标准化器
│   │   └── rolling/           # 滚动计算器
│   └── analysis/              # 因子分析
│       ├── calculator.py      # 因子计算器
│       ├── metrics.py         # 分析指标
│       └── report.py          # 报告生成
├── backtest/                   # 回测领域
│   ├── core/                  # 核心策略组件
│   ├── pipeline/              # 回测流水线
│   │   ├── selectors/         # 选股器
│   │   ├── allocators/        # 权重分配器
│   │   └── capital/           # 资金管理器
│   ├── triggers/              # 触发器
│   ├── data/                  # 数据导出
│   └── reports/               # 报告生成
├── data/                       # 数据层
│   ├── manager.py             # 数据管理器
│   ├── calendar.py            # 交易日历
│   ├── providers/             # 数据提供者
│   │   ├── sqlite.py          # SQLite 数据源
│   │   ├── akshare.py         # Akshare 数据源
│   │   └── tushare.py         # Tushare 数据源
│   └── handlers/              # 数据处理器
│       └── database.py        # 数据库处理
├── utils/                      # 通用工具
│   └── logger.py              # 日志配置
├── scripts/                    # 执行脚本
│   ├── run_factor_analysis.py # 因子分析脚本
│   ├── run_backtest.py        # 回测脚本
│   └── download_data.py       # 数据下载脚本
├── database/                   # 数据库目录
├── output/                     # 输出目录
│   ├── reports/               # 因子分析报告
│   └── logs/                  # 日志文件
├── bt_report/                  # 回测报告目录
├── bt_data/                    # Backtrader 数据目录
└── old_code/                   # 旧代码备份
```

## 快速开始

### 1. 环境配置
```bash
pip install -r requirements.txt
```

### 2. 运行模式

系统提供统一的命令行入口：

```bash
# 因子分析模式 - 计算因子、生成分析报告
python main.py --mode factor_analysis

# 回测模式 - 运行 Backtrader 事件驱动回测
python main.py --mode backtest

# 数据下载模式 - 下载/更新股票数据
python main.py --mode download_data

# 列出组件 - 显示所有已注册的因子、策略等
python main.py --mode list_components
```

### 3. 查看报告

- **因子分析报告**: `output/factor_reports/` 目录
- **回测报告**: `output/bt_reports/` 目录

## 配置文件

### 因子配置 (`configs/factors.yaml`)
```yaml
Momentum:
  enabled: true
  category: simple
  params:
    period: 20
  required_columns: [close]

IndNeu_EP:
  enabled: true
  category: complex
  required_columns: [close, net_profit_parent, share_capital]
```

### 策略配置 (`configs/strategies.yaml`)
```yaml
EqualWeights:
  enabled: true
  combiner: EqualWeight
  standardizer: ZScore

RollingICIR:
  enabled: true
  combiner: DynamicWeight
  standardizer: ZScore
  rolling_calculator: RollingICIRCalculator
  rolling_params:
    window_size: 60
    min_periods: 20
```

### 回测配置 (`configs/backtest.yaml`)
```yaml
start_date: "2024-01-01"
end_date: "2024-12-31"
initial_cash: 1000000.0
commission: 0.0003
benchmark: "600519"
```

### 股票池配置 (`configs/universe.yaml`)
```yaml
universe:
  - "600519"  # 贵州茅台
  - "000858"  # 五粮液
  # ...
```

## 组件注册机制

系统使用装饰器模式注册组件，支持动态加载：

### 注册因子
```python
from core.registry import register_factor

@register_factor("MyFactor")
def calculate_my_factor(df: pd.DataFrame, **params) -> pd.Series:
    # 因子计算逻辑
    return factor_series
```

### 注册合成器
```python
from core.registry import register_combiner

@register_combiner("MyCombiner")
class MyCombiner:
    def combine(self, factors_df, weights=None):
        # 合成逻辑
        return combined_series
```

## 数据要求

系统支持多种数据源：
- **SQLite 数据库**: 本地股票数据
- **Akshare**: 免费 A 股数据
- **Tushare**: 专业金融数据 (需要 Token)

主要数据类型：
- 股票价格数据 (日线 OHLCV)
- 财务数据 (利润表、资产负债表、现金流量表)
- 行业分类数据
- 交易日历

## 输出结果

### 因子分析报告
- 因子 IC/IR 分析
- 分组收益率表现
- 因子衰减分析
- 累计收益曲线
- 可视化图表

### 回测报告
- 策略收益曲线
- 最大回撤分析
- 夏普比率
- 交易明细
- 持仓分析

## 扩展开发

### 添加新因子
1. 在 `factors/library/` 创建新文件
2. 使用 `@register_factor` 装饰器注册
3. 在 `configs/factors.yaml` 中启用

### 添加新策略
1. 在 `factors/pipeline/combiners/` 实现合成器
2. 使用 `@register_combiner` 装饰器注册
3. 在 `configs/strategies.yaml` 中配置

### 添加新选股器/分配器
1. 在 `backtest/pipeline/` 相应目录实现
2. 继承基类并实现核心方法

## 技术栈

- **数据处理**: pandas, numpy
- **机器学习**: scikit-learn, LightGBM
- **回测引擎**: Backtrader
- **可视化**: matplotlib, plotly
- **并行计算**: joblib, multiprocessing
- **数据库**: SQLite
- **配置管理**: PyYAML

## 许可证

MIT License
