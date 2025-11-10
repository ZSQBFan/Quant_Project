# quant_project
FactorLab: 模块化量化因子回测框架
FactorLab 是一个为量化研究人员设计的、高度模块化、可扩展的Python因子回测框架。

它的核心目标是速度与灵活性：

快速迭代: 允许研究员在 main_analyzer.py 的“控制面板”中快速配置、测试新的因子和策略。

灵活扩展: 所有关键组件（数据、因子、标准化、合成、策略）都被解耦为独立的模块，可以像乐高积木一样轻松替换或添加。

核心特性
多源数据管理 (data_manager.py): 采用生产者-消费者多线程模型，自动从多个数据源（如 Akshare, Tushare, 或本地 SQLite 数据库）高效地检查、下载和清洗数据，并存入统一的回测数据库。

模块化因子库 (factors.py, factors_complex.py):

Type 1 (简单因子): 在 factors.py 中定义仅依赖单股票价格的因子 (如 RSI, 动量)。

Type 2 (复合因子): 在 factors_complex.py 中定义需要跨股票或外部数据（如行业、基本面）的因子 (如 行业中性化动量)。

可插拔处理层 (factor_standardizer.py, factor_combiner.py):

标准化: 在截面上对原始因子值应用 Z-Score, Quantile 排名或不进行处理。

合成: 使用等权重、固定权重、动态显著性权重或自定义权重将多个因子合成为一个 Alpha 信号。

动态滚动策略 (rolling_weight_calculator.py):

内置两种先进的动态权重计算逻辑：滚动ICIR加权 (RollingICIRCalculator) 和 滚动回归加权 (RollingRegressionCalculator / Pooled OLS)。

权重计算与因子合成（DynamicWeightCombiner）解耦，易于扩展AI/ML模型（如随机森林）。

自动化分析报告 (factor_report.py, analysis_metrics.py):

自动计算所有核心绩效指标：IC序列、ICIR、分层收益、多空组合净值等。

一键生成包含图表的 HTML 报告，并与基准（Benchmark）进行对比。

项目结构
.
├── database/
│   ├── JY_database/          # (示例) 您的原始数据源 (如 聚源/Tushare/Wind)
│   │   └── JY_database.sqlite
│   └── quant_data.db         # 【核心】回测框架使用的【专用】数据库 (由 data_manager 自动写入)
│
├── factor_reports/           # 自动生成的 HTML 报告输出目录
│   └── report_...html
│
├── logger/                   # 日志文件生成
│   └── logger_config.py      #【日志记录】
│
├── logs/                     # 日志文件目录
│
├── main_analyzer.py          # 【【主入口 & 控制面板】】
├── strategy_configs.py       # (必需) 【策略注册中心】，定义滚动配置
├── universe_config.py        # (必需) 【股票池配置文件】
│
├── core/
│   ├── analysis_metrics.py   # 【核心数学】定义 IC, IR, 分层收益等计算
│   ├── factor_combiner.py    # 【核心组件】定义因子合成器 (等权, 固定, 动态)
│   ├── rolling_weight_calculator.py # 【滚动大脑】ICIR/OLS 权重计算器
│   └── factor_standardizer.py# 【核心组件】定义标准化器 (Z-Score, Quantile)
│
│
├── data/
│   ├── data_manager.py       # 【数据引擎】负责ETL、多线程下载和数据缓存
│   ├── data_providers.py     # 【数据接口】定义 Akshare, Tushare, SQLite 适配器
│   ├── database_handler.py   # 【数据工具】线程安全的 SQLite 处理器
│   └── trading_calendars.py  # 【数据工具】交易日历获取器
│
└── factor_analysis/
    ├── factors.py            # 【因子库-Type 1】简单因子 (RSI, 动量)
    ├── factors_complex.py    # 【因子库-Type 2】复合因子 (行业中性化)
    ├── factor_calculator.py  # 【执行器】Type 1 因子的计算引擎
    └── factor_report.py      # 【报告器】生成 HTML 报告
     
如何运行一次回测
运行一次完整的因子分析流程非常简单，所有高频操作都在 main_analyzer.py 的顶部“控制面板”中完成：

(可选) 配置数据准备:

在 main_analyzer.py 中设置 SKIP_DATA_PREPARATION = True 可跳过数据下载（用于快速迭代）。

在 2. 基础回测与路径配置 中设置 START_DATE, END_DATE 和数据库路径。

在 universe_config.py 中定义 UNIVERSE 股票池（如果列表为空，data_manager.py 会自动加载全市场股票）。

选择策略 (1a):

在 strategy_configs.py (您未提供) 中定义您的策略（例如 RollingICIR）。

在 main_analyzer.py 中设置 STRATEGY_NAME 为您选择的策略名。

选择因子 (1b, 1c):

Type 1: 在 FACTORS_TO_ANALYZE 列表中添加 factors.py 中注册的因子及其参数。

Type 2: 在 COMPLEX_FACTORS_TO_RUN 列表中添加 factors_complex.py 中注册的因子名。

设置 LOAD_INDUSTRY_DATA = True 来为 Type 2 因子启用外部数据合并。

选择标准化器 (1e):

设置 STANDARDIZER_CLASS 为 CrossSectionalZScoreStandardizer (推荐) 或其他。

运行:

直接在命令行中运行主文件：

Bash

python main_analyzer.py
程序将自动执行所有步骤（数据、计算、合成、评估），并在 factor_reports/ 目录下生成最终的 HTML 报告。

如何扩展
A. 如何添加一个新的简单因子 (Type 1)
factor_analysis/factors.py:

编写您的因子计算函数（例如 calculate_my_factor），它必须接收 df (pd.DataFrame) 和 **kwargs。

在文件底部的 FACTOR_REGISTRY 中注册您的函数：'MyFactor': calculate_my_factor。

main_analyzer.py:

在 FACTORS_TO_ANALYZE 列表中添加 ('MyFactor', {'param1': 10})。

运行程序。

B. 如何添加一个新的复合因子 (Type 2, 如基本面)
(ETL): 确保您的外部数据（如E/P, FCF）已被存入 quant_data.db 中的一个新表（例如 stock_fundamentals）。

data/data_manager.py:

模仿 get_industry_mapping()，添加一个新函数 get_fundamental_data()，用于从 stock_fundamentals 表中读取数据。

main_analyzer.py:

在 "控制面板" 添加一个新开关 LOAD_FUNDAMENTAL_DATA = True。

在 "步骤 2.5" 中，调用 data_manager.get_fundamental_data() 并将其 merge 到 all_data_df 中（注意使用 ffill() 处理季报数据）。

factor_analysis/factors_complex.py:

编写您的因子函数（例如 calculate_ep_factor），它接收 all_data_df 作为参数（此时 all_data_df 已包含 ep_ratio 列）。

在 COMPLEX_FACTOR_REGISTRY 中注册 'EP_Factor': calculate_ep_factor。

main_analyzer.py:

在 COMPLEX_FACTORS_TO_RUN 列表中添加 'EP_Factor'。

运行程序。

C. 如何添加一个新的滚动策略 (如 AI 因子)
factor_analysis/rolling_weight_calculator.py:

创建一个新类 RollingMLCalculator，它必须实现 calculate_new_weights 方法。

在此方法中，您可以像 RollingRegressionCalculator 一样准备 X 和 y，但调用的是（例如）RandomForest.fit(X, y)。

返回从 model.feature_importances_ 提取的权重字典。

strategy_configs.py:

导入您的 RollingMLCalculator。

注册一个新策略（例如 "RollingRandomForest"），将其 roller_class 指向 RollingMLCalculator。

main_analyzer.py:

设置 STRATEGY_NAME = "RollingRandomForest"。

运行程序。

依赖库
pandas

numpy

pandas-ta (用于 factors.py)

statsmodels (用于 Fama-MacBeth, alpha_icir.py 中提及)

scikit-learn (用于 RollingRegressionCalculator)

matplotlib (用于 factor_report.py)

tqdm

akshare (用于 data_providers.py)

tushare (用于 data_providers.py)
