# 量化投资分析系统

一个专业的股票量化投资分析框架，支持多因子模型、动态权重策略和机器学习集成。

## 项目概述

本项目是一个完整的量化投资分析系统，专注于因子投资策略的研究与回测。系统采用模块化设计，支持多种因子类型和组合策略，提供全面的分析报告。

## 核心特性

### 🎯 多因子模型
- **量价因子**: 动量、反转、RSI、布林带等
- **基本面因子**: 估值、盈利、成长、质量等维度
- **行业中性因子**: 消除行业影响的标准化因子

### ⚙️ 策略框架
- **静态组合策略**: 等权重、固定权重、动态显著性
- **动态滚动策略**: 基于ICIR、回归分析的动态权重调整
- **机器学习策略**: LightGBM模型预测

### 📊 分析功能
- 因子IC分析、分组回测、衰减分析
- 多周期收益率预测 (1/5/10/20/30/90日)
- 交互式HTML报告生成

## 项目结构

```
quant_project_3.12/
├── core/                    # 核心框架
│   ├── abstractions.py     # 抽象基类
│   └── strategy.py         # 策略配置
├── data/                   # 数据管理
│   ├── data_manager.py     # 数据管理器
│   ├── data_providers.py   # 数据源接口
│   └── database_handler.py # 数据库处理
├── factor_analysis/        # 因子分析
│   ├── factor_calculator.py # 因子计算器
│   ├── factor_report.py    # 报告生成
│   ├── factors.py          # 基础因子
│   └── factors_complex.py  # 复合因子
├── strategies/             # 策略实现
│   ├── combiners.py        # 因子组合器
│   ├── rolling_calculators.py # 滚动计算
│   └── ai_trainers.py      # AI训练器
├── logger/                 # 日志系统
├── factor_reports/         # 报告输出目录
└── test/                   # 测试脚本
```

## 快速开始

### 1. 环境配置
```bash
pip install -r requirements.txt
```

### 2. 配置因子和策略
编辑 [`factor_configs.py`](factor_configs.py:1) 和 [`strategy_configs.py`](strategy_configs.py:1) 文件，配置需要分析的因子和策略。

### 3. 运行分析
```bash
python main_analyzer.py
```

### 4. 查看报告
分析完成后，在 `factor_reports/` 目录下查看生成的HTML报告。

## 核心配置文件

### 因子配置 ([`factor_configs.py`](factor_configs.py:1))
定义所有可用的因子，包括：
- 量价因子 (Momentum, Reversal20D, RSI等)
- 基本面因子 (EP, BP, ROE等)
- 行业中性因子 (IndNeu_*)

### 策略配置 ([`strategy_configs.py`](strategy_configs.py:1))
注册可用的策略：
- RollingICIR: 基于ICIR的动态权重
- RollingRegression: 基于回归的动态权重  
- FixedWeights: 固定权重组合
- EqualWeights: 等权重组合
- LightGBM_Periodic: 机器学习策略

### 主分析器 ([`main_analyzer.py`](main_analyzer.py:1))
系统的主要入口，协调整个分析流程：
1. 数据准备和验证
2. 因子计算和标准化
3. 策略执行和组合
4. 报告生成和输出

## 数据要求

系统支持多种数据源，主要通过SQLite数据库获取：
- 股票价格数据 (日线)
- 财务数据 (净利润、净资产等)
- 行业分类数据
- 交易日历

## 输出结果

系统生成详细的HTML报告，包含：
- 因子IC分析
- 分组收益率表现
- 因子衰减分析  
- 组合策略回测结果
- 可视化图表

## 扩展开发

### 添加新因子
1. 在 [`factor_configs.py`](factor_configs.py:13) 中注册因子配置
2. 在 [`factor_analysis/factors.py`](factor_analysis/factors.py:1) 或 [`factor_analysis/factors_complex.py`](factor_analysis/factors_complex.py:1) 中实现因子逻辑

### 添加新策略
1. 在 [`strategies/combiners.py`](strategies/combiners.py:1) 中实现组合器
2. 在 [`strategy_configs.py`](strategy_configs.py:94) 中注册策略配置

## 技术栈

- **数据处理**: pandas, numpy
- **机器学习**: scikit-learn, LightGBM
- **可视化**: plotly
- **并行计算**: joblib
- **数据库**: SQLite