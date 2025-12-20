"""
对抗式 LLM 因子合成器

通过多智能体对抗（Multi-Agent Adversarial Debate）机制动态决定因子权重。
"""

import pandas as pd
import numpy as np
import logging
import json
import re
import requests
from typing import Any, Dict, List

from factors.core.abstractions import RollingCalculatorBase
from factors.analysis import metrics


class AdversarialLLMCombiner(RollingCalculatorBase):
    """
    通过多智能体对抗（Multi-Agent Adversarial Debate）机制来动态决定因子的权重。

    该类使用两个具有不同投资理念的AI代理进行辩论：
    - Agent A (激进型投资经理): 关注Alpha收益最大化
    - Agent B (保守型风险经理): 关注风险控制和回撤管理

    通过多轮辩论达到最优权重分配，支持正负权重以表达对因子的不同观点。
    """

    def __init__(self,
                 api_url: str = "https://api.openai.com/v1/chat/completions",
                 api_key: str | None = None,
                 max_rounds: int = 2,
                 include_factor_values: bool = False,
                 include_conversation_history: bool = False,
                 allow_negative_weights: bool = True,
                 **kwargs):
        """
        初始化 AdversarialLLMCombiner。

        Args:
            api_url: OpenAI 兼容 API 的 URL
            api_key: API 密钥
            max_rounds: 最大辩论轮数
            include_factor_values: 是否包含当前因子值
            include_conversation_history: 是否包含对话历史
            allow_negative_weights: 是否允许负权重（做空因子观点）
        """
        super().__init__(**kwargs)
        self.api_url = api_url
        self.api_key = api_key
        self.max_rounds = max_rounds
        self.include_factor_values = include_factor_values
        self.include_conversation_history = include_conversation_history
        self.allow_negative_weights = allow_negative_weights
        self.conversation_history = []

    def _call_llm(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        调用 LLM API 获取响应。

        Args:
            prompt: 用户提示
            system_prompt: 系统提示

        Returns:
            LLM 的响应文本
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 添加对话历史（如果需要）
        if self.include_conversation_history and self.conversation_history:
            messages.extend(self.conversation_history)

        payload = {
            "model": "qwen-flash",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 5000
        }

        # 如果允许负权重，在提示中明确说明
        if self.allow_negative_weights:
            payload["temperature"] = 0.8  # 稍微增加随机性以促进创新思维

        try:
            response = requests.post(self.api_url,
                                     headers=headers,
                                     json=payload,
                                     timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logging.error(f"LLM调用失败: {e}")
            raise

    def _parse_json_response(self,
                             response_text: str,
                             max_retries: int = 3) -> Dict[str, float]:
        """
        解析 LLM 的 JSON 响应，包含错误处理和重试机制。

        Args:
            response_text: LLM 的响应文本
            max_retries: 最大重试次数

        Returns:
            解析后的权重字典
        """
        for attempt in range(max_retries):
            try:
                # 尝试从响应中提取 JSON 部分
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1

                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    parsed_response = json.loads(json_str)

                    # 如果响应包含理由字段，提取权重部分
                    if isinstance(parsed_response, dict):
                        # 检查是否是带有理由的格式
                        if all(
                                isinstance(v, dict) and 'weight' in v
                                for v in parsed_response.values()):
                            # 提取权重
                            weights = {
                                k: v['weight']
                                for k, v in parsed_response.items()
                            }
                            # 记录理由信息
                            reasons = {
                                k: v.get('reason', '未提供理由')
                                for k, v in parsed_response.items()
                            }
                            logging.info(
                                f"[AdversarialLLM] 权重理由: {json.dumps(reasons, ensure_ascii=False)}"
                            )
                        else:
                            # 标准格式
                            weights = parsed_response

                        # 验证权重字典的格式
                        if isinstance(weights, dict) and all(
                                isinstance(k, str)
                                and isinstance(v, (int, float))
                                for k, v in weights.items()):
                            return weights

                # 如果解析失败，记录日志并准备重试
                logging.warning(
                    f"JSON解析失败 (尝试 {attempt + 1}/{max_retries}): {response_text}"
                )

                if attempt < max_retries - 1:  # 不是最后一次尝试
                    # 请求LLM重新格式化输出
                    retry_prompt = f"请将以下内容重新格式化为标准JSON格式的字典，键为因子名称，值为权重数值（可正可负）：\n\n{response_text}"
                    response_text = self._call_llm(retry_prompt)

            except json.JSONDecodeError as e:
                logging.warning(
                    f"JSON解码错误 (尝试 {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:  # 不是最后一次尝试
                    retry_prompt = f"请将以下内容重新格式化为标准JSON格式的字典，键为因子名称，值为权重数值：\n\n{response_text}"
                    response_text = self._call_llm(retry_prompt)

            except Exception as e:
                logging.error(f"解析过程中发生未知错误: {e}")
                break

        # 如果所有重试都失败，抛出异常
        raise ValueError(f"无法解析LLM响应为有效的JSON格式: {response_text}")

    def _calculate_payload_for_day(
            self, historical_data_window: pd.DataFrame, current_date: pd.Timestamp = None) -> Dict[str, float]:
        """
        使用多智能体对抗机制计算因子权重。

        Args:
            historical_data_window: 历史数据窗口
            current_date: 当前计算日期（本方法中暂未使用，保留接口兼容性）

        Returns:
            因子权重字典
        """
        # 准备因子指标数据
        factor_metrics = {}
        for fname in self.factor_names:
            # 获取因子数据和对应的 forward return
            factor_cols = [
                col for col in historical_data_window.columns
                if col.startswith(fname)
            ]
            if not factor_cols:
                continue

            # 获取默认的 forward return 周期（如果有配置的话）
            return_period = 30  # 默认30天周期
            return_col = f'forward_return_{return_period}d'

            # 检查是否有该周期的 forward return 数据
            if return_col not in historical_data_window.columns:
                # 尝试找到最接近的 forward return 列
                forward_cols = [
                    col for col in historical_data_window.columns
                    if col.startswith('forward_return_')
                ]
                if forward_cols:
                    return_col = forward_cols[0]
                    # 从列名中提取实际的周期数，如 'forward_return_5d' -> 5
                    match = re.search(r'forward_return_(\d+)d', return_col)
                    if match:
                        return_period = int(match.group(1))
                else:
                    continue

            # 安全提取因子值 Series (处理重复列)
            if isinstance(historical_data_window[fname], pd.DataFrame):
                logging.warning(f"因子 {fname} 存在重复列，将使用第一列。")
                factor_series = historical_data_window[fname].iloc[:, 0]
            else:
                factor_series = historical_data_window[fname]

            # 安全提取收益率 Series (处理重复列)
            if isinstance(historical_data_window[return_col], pd.DataFrame):
                logging.warning(f"收益列 {return_col} 存在重复列，将使用第一列。")
                return_series = historical_data_window[return_col].iloc[:, 0]
            else:
                return_series = historical_data_window[return_col]

            # 计算 IC 序列和统计指标
            ic_data = pd.DataFrame({
                'factor_value': factor_series,
                return_col: return_series
            })
            
            ic_series = metrics.calculate_rank_ic_series(
                ic_data.dropna(), return_period)
            ic_stats = metrics.analyze_ic_statistics(ic_series)

            factor_metrics[fname] = ic_stats

        # 如果没有因子指标数据，返回等权分配
        if not factor_metrics:
            return {
                fname: 1.0 / len(self.factor_names)
                for fname in self.factor_names
            }

        # 构建 Agent A (Portfolio Manager) 的提示
        agent_a_system_prompt = (
            "你是一名为顶级对冲基金工作的量化投资组合经理(Alpha PM)。"
            "你的核心目标是最大化投资组合的预期信息比率(IR)。"
            "【决策逻辑】："
            "1. **方向性判断**：根据 IC 均值的正负决定权重的正负。若因子与未来收益负相关(IC<0)，应分配负权重(做空该因子)。"
            "2. **权重分配**：根据 IC_IR (信息比率) 和 Rank_IC 的稳定性分配权重大小。高 IR 的因子应获得更高的绝对权重暴露。"
            "3. **多空策略**：你被允许构建多空组合。不要局限于纯多头。"
            "【输出约束】："
            "请以严格的 JSON 格式返回结果。JSON 结构必须包含 'weight' (float) 和 'reason' (string)。"
            "约束条件：所有因子权重的**绝对值之和 (Sum of Absolute Weights)** 应接近 1.0 (即 Gross Exposure = 100%)。"
        )

        # 构建因子指标描述
        metrics_description = ""
        for fname, stats in factor_metrics.items():
            metrics_description += f"\n因子 '{fname}':\n"
            for stat_name, stat_value in stats.items():
                metrics_description += f"  {stat_name}: {stat_value:.4f}\n"

        agent_a_prompt = (
            f"请基于以下因子的历史回测绩效指标，构建最优的因子权重向量：\n"
            f"{metrics_description}\n"
            f"【分析要求】\n"
            f"1. 识别有效因子：重点关注 IC Mean 的绝对值大小和 IC IR。\n"
            f"2. 确定方向：如果 IC Mean 为负且显著，请毫不犹豫地给予负权重。\n"
            f"3. 剔除噪音：对于 IC 接近 0 或 IR 极低的因子，应给予 0 或极低的权重。\n"
            f"【输出示例】\n"
            f"{{\"FactorA\": {{\"weight\": 0.4, \"reason\": \"Strong positive correlation (IC=0.05), high stability\"}}, "
            f"\"FactorB\": {{\"weight\": -0.3, \"reason\": \"Consistent negative correlation, used as reverse signal\"}}}}"
        )

        # Agent A 的第一轮建议
        try:
            agent_a_response = self._call_llm(agent_a_prompt,
                                              agent_a_system_prompt)
            agent_a_weights = self._parse_json_response(agent_a_response)

            # 记录原始模型输出作为 info 信息
            logging.debug(f"[AdversarialLLM] Agent A原始输出: {agent_a_response}")
            logging.debug(
                f"[AdversarialLLM] Agent A解析权重: {json.dumps(agent_a_weights)}")

            # 更新对话历史
            if self.include_conversation_history:
                self.conversation_history.append({
                    "role":
                    "assistant",
                    "content":
                    f"Agent A建议: {json.dumps(agent_a_weights)}"
                })
        except Exception as e:
            logging.error(f"Agent A响应失败: {e}")
            # 回退到等权分配
            return {
                fname: 1.0 / len(self.factor_names)
                for fname in self.factor_names
            }

        current_weights = agent_a_weights
        previous_weights = None

        # 多轮对抗辩论
        for round_num in range(self.max_rounds):
            # 构建 Agent B (Risk Manager) 的提示
            agent_b_system_prompt = (
                "你是一名为顶级对冲基金工作的首席风控官(CRO)。"
                "你的职责是审查 PM 提交的因子权重方案，识别潜在的过度拟合风险、拥挤风险和尾部风险。"
                "【审查维度】："
                "1. **方向性风险**：检查 PM 是否错误地做多了一个长期衰减的因子，或者做空了一个虽然近期回撤但长期有效的因子。"
                "2. **过度集中**：警惕单一因子（无论方向）的绝对权重过大（例如 > 40%），除非其 IR 极高。"
                "3. **历史表现**：如果一个因子的近期表现（如最近1个月 IC）与其长期均值背离，需考虑是否是风格切换（Regime Shift），并建议降低暴露。"
                "【输出约束】："
                "请以严格的 JSON 格式返回修改后的权重。保持权重的**绝对值之和**接近 1.0。"
                "必须提供专业的风控修正理由。")

            # 构建 Agent B 的提示
            weights_description = ""
            for fname, weight in current_weights.items():
                weights_description += f"  {fname}: {weight:.4f}\n"

            agent_b_prompt = (
                f"投资组合经理提出了以下权重分配建议：\n"
                f"{weights_description}\n"
                f"请基于以下因子风险指标对其进行批判和修改：\n"
                f"{metrics_description}\n\n"
                f"请特别关注以下几点：\n"
                f"1. 是否有因子权重绝对值过高导致过度集中？\n"
                f"2. 是否有因子近期表现不佳但仍被赋予高正权重？\n"
                f"3. 是否存在因子拥挤风险需要通过对冲（负权重）来管理？\n"
                f"4. 权重分配是否符合风险调整后收益最大化的投资目标？\n"
                f"【重要要求】：请为每个权重分配提供详细的理由说明，解释为什么做出这样的修改（包括正负符号）。\n"
                f"请以严格的JSON格式返回修改后的权重分配，所有权重绝对值之和必须为1.0。")

            try:
                agent_b_response = self._call_llm(agent_b_prompt,
                                                  agent_b_system_prompt)
                agent_b_weights = self._parse_json_response(agent_b_response)

                # 记录原始模型输出作为 info 信息
                logging.debug(
                    f"[AdversarialLLM] Agent B原始输出: {agent_b_response}")
                logging.debug(
                    f"[AdversarialLLM] Agent B解析权重: {json.dumps(agent_b_weights)}"
                )

                # 更新对话历史
                if self.include_conversation_history:
                    self.conversation_history.append({
                        "role":
                        "assistant",
                        "content":
                        f"Agent B建议: {json.dumps(agent_b_weights)}"
                    })

                # 检查权重变化是否足够小（收敛条件）
                if previous_weights is not None:
                    weight_diff = sum(
                        abs(
                            current_weights.get(fname, 0) -
                            agent_b_weights.get(fname, 0))
                        for fname in set(current_weights.keys())
                        | set(agent_b_weights.keys()))
                    if weight_diff < 0.01:  # 如果权重变化小于1%，认为已收敛
                        logging.info(f"权重已收敛，停止辩论。轮数: {round_num + 1}")
                        return agent_b_weights

                previous_weights = current_weights
                current_weights = agent_b_weights

            except Exception as e:
                logging.error(f"Agent B响应失败: {e}")
                # 返回当前最好的权重分配
                break

        return current_weights

    def _combine_factors_for_day(self, payload: Dict[str, float],
                                 daily_factors: pd.DataFrame) -> pd.Series:
        """
        根据计算出的权重合成因子。

        Args:
            payload: 因子权重字典
            daily_factors: 当日因子值

        Returns:
            合成后的因子值 Series
        """
        try:
            # 处理日度因子切片退化为 Series 的场景，确保保持行向量形态
            if isinstance(daily_factors, pd.Series):
                daily_factors = daily_factors.to_frame().T
                # Series.name 常用于资产ID，尽量保持索引含义
                if daily_factors.index.name is None:
                    daily_factors.index.name = 'asset'

            if daily_factors.empty:
                logging.warning("因子合成失败: 当日因子为空，跳过。")
                return None

            # 将权重载荷安全转换为数值型 Series，过滤掉非数值/无穷/NaN
            weights = pd.Series(payload)
            weights = pd.to_numeric(weights, errors="coerce")
            weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0)
            weights = weights.reindex(daily_factors.columns, fill_value=0).astype(float)

            # 归一化权重确保绝对值总和为 1（使用显式标量避免 Series 布尔歧义）
            weight_abs_sum = float(weights.abs().sum())
            if weight_abs_sum > 0:
                weights = weights / weight_abs_sum
            else:
                # 权重为空或全 0 时回退等权
                logging.warning("权重载荷为空或全为零，回退为等权合成。")
                if len(daily_factors.columns) == 0:
                    return None
                weights = pd.Series(1.0, index=daily_factors.columns, dtype=float)
                weights = weights / float(weights.abs().sum())

            # 计算加权合成因子
            combined_factor = (daily_factors * weights).sum(axis=1)
            return combined_factor
        except Exception as e:
            logging.error(f"因子合成失败: {e}")
            # 回退到等权合成
            return daily_factors.sum(axis=1)
