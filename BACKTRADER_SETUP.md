# Backtrader 回测框架 MVP 目录结构说明

## 创建时间
2025-12-14

## 架构说明
本目录结构遵循**架构优先，算法最简**原则，旨在建立从 Pandas 因子计算到 Backtrader 事件驱动回测的闭环链路。

## 目录结构

### config/ - 配置目录
存放所有与回测相关的配置文件，与原有的 factor_configs.py 和 strategy_configs.py 解耦，专注于 Backtrader 运行时配置。

### bt/ - Backtrader 扩展核心
主回测引擎目录，包含以下子模块：

#### bt/core/
**Backtrader 核心组件**
- DataFeed 实现：从 Pandas DataFrame 读取预计算的因子数据
- 策略引擎：中央策略大脑，协调所有触发器和交易决策
- 指令执行器：管理订单生成、风控检查和交易执行

#### bt/triggers/
**事件触发器系统**（独立的探测器模块）
- 风控触发器：止损、止盈、回撤控制（最简实现：固定百分比）
- 调仓触发器：定期调仓、信号阈值触发（最简实现：固定日期）
- 流动性触发器：成交量、流动性检查（最简实现：固定阈值）

**关键原则**：触发器不直接交易，只向中央缓冲区提交"意图(Intent)"。

#### bt/pipeline/
**数据处理流水线**
- PandasToBacktrader：将因子分析结果转换为 Backtrader 可识别的数据格式
- 数据验证器：检查数据完整性、对齐时间索引、处理停牌等情况
- 信号标准化：确保权重和信号格式符合 Backtrader 要求

#### bt/utils/
**工具函数**
- Backtrader 适配器：通用工具函数和辅助类
- 性能监控：回测性能统计和分析工具
- 可视化辅助：回测结果的可视化支持

### bt_report/ - 报告输出目录
存放 Backtrader 回测生成的报告和结果，与 factor_reports/ 解耦，专注于交易层面的分析。

## 与现有系统的集成点

1. **数据流集成**：从 factor_analysis/ 输出的 combined_signal 直接进入 bt/pipeline/
2. **日志集成**：复用 logger/ 模块，保持一致的日志格式和级别控制
3. **配置集成**：新的策略配置可以在 strategy_configs.py 中注册，或创建独立的 bt_config.py
4. **报告集成**：回测报告输出到 bt_report/，与因子分析报告分离但保持统一风格

## 下一步开发任务（按优先级）

### 阶段 1：数据流闭环（MVP核心）
- [ ] 实现 bt/pipeline/pandas_feed.py：Pandas DataFrame 转 Backtrader DataFeed
- [ ] 创建 bt/core/base_strategy.py：中央策略基类，实现 pending_actions 缓冲区
- [ ] 实现 bt_report/analyzer.py：基础回测结果分析器

### 阶段 2：触发器系统
- [ ] 实现 bt/triggers/risk_manager.py：固定百分比止损触发器
- [ ] 实现 bt/triggers/rebalance_trigger.py：定期调仓触发器
- [ ] 实现 bt/triggers/signal_validator.py：信号有效性验证

### 阶段 3：执行与风控
- [ ] 完善 bt/core/order_executor.py：订单优先级裁决逻辑
- [ ] 实现 bt/core/position_manager.py：仓位管理（等权重最简实现）
- [ ] 添加 bt/config/bt_config.py：回测参数配置文件

### 阶段 4：报告与验证
- [ ] 扩展 bt_report/analyzer.py：生成交易明细、持仓分析
- [ ] 实现回测与因子分析结果的一致性验证
- [ ] 创建 main_backtrader.py：独立的回测入口脚本

## 示例数据流

```
Pandas 因子分析
    ↓ (combined_signal DataFrame)
bt/pipeline/pandas_feed.py
    ↓ (Backtrader DataFeed)
bt/core/base_strategy.next()
    ↓
bt/triggers/* (多个触发器并行检测)
    ↓ (提交 Intents 到 pending_actions)
bt/core/order_executor (优先级裁决)
    ↓ (生成 Orders)
Backtrader 引擎执行
    ↓ (回测结果)
bt_report/analyzer.py
    ↓ (HTML报告)
bt_report/
```

## 重要约定

1. **禁止在 Backtrader 中重复计算因子**：DataFeed 只负责搬运 Pandas 结果
2. **触发器必须解耦**：每个触发器独立运作，只提交意图不执行交易
3. **优先级顺序**：风控 > 清仓 > 建仓（具体数值在 bt/core/order_executor.py 中定义）
4. **最简算法原则**：所有算法优先使用固定阈值/等权重等最简单实现

## 联系方式
如有关于架构的疑问，请参考项目根目录的 README.md 和 AGENTS.md 文件。
