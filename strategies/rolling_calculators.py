# strategies/rolling_calculators.py

import pandas as pd
import logging
import json
import requests
from typing import Any, Dict, List
from sklearn.linear_model import LinearRegression
from core.abstractions import RollingCalculatorBase, AITrainerBase
from factor_analysis import analysis_metrics as metrics


class RollingICIRCalculator(RollingCalculatorBase):

    def __init__(self, factor_weight_config: Dict[str, str],
                 forward_return_periods: List[int], **kwargs):
        super().__init__(**kwargs)
        self.config, self.periods = factor_weight_config, forward_return_periods

    def _parse_metric_str(self, s: str) -> tuple[str, int]:
        p = s.split('_')[-1]
        k = '_'.join(s.split('_')[:-1]) or 'ir'
        if p.endswith('d'): return k, int(p[:-1])
        return s, self.periods[0]

    def _calculate_payload_for_day(self,
                                   hist_df: pd.DataFrame) -> Dict[str, float]:
        w = {}
        
        # 【关键调试】打印所有可用的列名
        available_cols = hist_df.columns.tolist()
        logging.debug(f"🔍 [RollingICIR] 历史数据窗口可用列: {available_cols}")
        logging.debug(f"🔍 [RollingICIR] 期望的因子名称: {self.factor_names}")
        
        for fname in self.factor_names:
            m_str = self.config.get(fname, f'ir_{self.periods[0]}d')
            m_key, p = self._parse_metric_str(m_str)
            
            # 【调试】检查输入数据
            logging.debug(f"🔍 [RollingICIR] 因子 {fname}: 计算 {p}d IC")
            logging.debug(f"🔍 [RollingICIR] 历史数据形状: {hist_df.shape}")
            
            # 【关键修复】更加严格的列存在性检查
            if fname not in hist_df.columns:
                # 查找匹配相似度的列
                similar_cols = [col for col in available_cols if fname.lower() in col.lower() or col.lower() in fname.lower()]
                logging.error(f"❌ [RollingICIR] 因子 {fname}: 在历史数据中未找到！")
                logging.error(f"❌ [RollingICIR] 可用列包括: {available_cols}")
                if similar_cols:
                    logging.warning(f"  > ⚠️ 发现相似列: {similar_cols}，检查是否命名不一致")
                w[fname] = 0.0
                continue
                
            return_col_p = f'forward_return_{p}d'
            if return_col_p not in hist_df.columns:
                logging.error(f"❌ [RollingICIR] 收益率列 {return_col_p}: 在历史数据中未找到！")
                available_return_cols = [col for col in available_cols if col.startswith('forward_return_')]
                logging.error(f"❌ [RollingICIR] 可用的收益率列: {available_return_cols}")
                w[fname] = 0.0
                continue
            
            # 验证数据
            factor_data = hist_df[fname]
            return_data = hist_df[return_col_p]
            logging.debug(f"🔍 [RollingICIR] 因子 {fname}: 非空值数={factor_data.count()}/{len(factor_data)}, 均值={factor_data.mean():.4f}")
            logging.debug(f"🔍 [RollingICIR] 收益率 {return_col_p}: 非空值数={return_data.count()}/{len(return_data)}, 均值={return_data.mean():.4f}")
            
            # 如果大多数值为0或NA，提示警告
            if factor_data.count() < len(factor_data) * 0.1:
                logging.warning(f"  > ⚠️ [RollingICIR] 因子 {fname}: 超过90%的值为空或NA！")
                
            ic_data = hist_df[[fname, return_col_p
                               ]].rename(columns={fname: 'factor_value'})
            logging.debug(f"🔍 [RollingICIR] IC计算数据形状: {ic_data.shape}")
            logging.debug(f"🔍 [RollingICIR] 数据样例:\n{ic_data.head()}")
            
            ic_s = metrics.calculate_rank_ic_series(ic_data.dropna(), p)
            logging.debug(f"🔍 [RollingICIR] 计算得到的IC序列: {len(ic_s)} 个值")
            if len(ic_s) > 0:
                logging.debug(f"🔍 [RollingICIR] IC统计: {metrics.analyze_ic_statistics(ic_s)}")
            
            w[fname] = metrics.analyze_ic_statistics(ic_s).get(m_key, 0.0)
            logging.debug(f"🔍 [RollingICIR] 因子 {fname} 权重: {w[fname]:.4f}")
            
        tot_s = sum(abs(v) for v in w.values())
        final_weights = {f: v / tot_s if tot_s else 0.0 for f, v in w.items()}
        logging.debug(f"🔍 [RollingICIR] 最终权重: {final_weights}")
        
        # 如果所有权重都是0，记录重大警告
        if all(v == 0.0 for v in final_weights.values()):
            logging.warning(f"⚠️ [RollingICIR] 所有因子权重都为0！使用等权回退")
            # 临时等权
            equal_weight = 1.0 / len(self.factor_names) if self.factor_names else 1.0
            final_weights = {f: equal_weight for f in self.factor_names}
            
        return final_weights

    def _combine_factors_for_day(self, payload: Dict[str, float],
                                 daily_df: pd.DataFrame) -> pd.Series:
        weights = pd.Series(payload).reindex(daily_df.columns, fill_value=0)
        return (daily_df * weights).sum(axis=1)


class RollingRegressionCalculator(RollingCalculatorBase):

    def __init__(self, target_return_period: int, **kwargs):
        super().__init__(**kwargs)
        self.return_col = f'forward_return_{target_return_period}d'

    def _calculate_payload_for_day(
            self, historical_data_window: pd.DataFrame) -> Dict[str, float]:
        data = historical_data_window[self.factor_names +
                                      [self.return_col]].dropna()
        if len(data) < len(self.factor_names) + 2:
            return {f: 0.0 for f in self.factor_names}
        X, y = data[self.factor_names], data[self.return_col]
        model = LinearRegression().fit(X, y)
        w = {n: c for n, c in zip(self.factor_names, model.coef_)}
        tot_s = sum(abs(v) for v in w.values())
        return {f: v / tot_s if tot_s else 0.0 for f, v in w.items()}

    def _combine_factors_for_day(self, payload: Dict[str, float],
                                 daily_factors: pd.DataFrame) -> pd.Series:
        weights = pd.Series(payload).reindex(daily_factors.columns,
                                             fill_value=0)
        return (daily_factors * weights).sum(axis=1)


class RollingAITrainer(RollingCalculatorBase):

    def __init__(self, training_calculator: AITrainerBase, **kwargs):
        super().__init__(**kwargs)
        self.trainer = training_calculator

    def _calculate_payload_for_day(
            self, historical_data_window: pd.DataFrame) -> Any:
        return self.trainer.train_model(historical_data_window,
                                        self.factor_names)

    def _combine_factors_for_day(self, payload: Any,
                                 daily_factors: pd.DataFrame) -> pd.Series:
        model = payload
        try:
            features = model.feature_name_ if hasattr(
                model, 'feature_name_') else daily_factors.columns
            X_today = daily_factors[features]
            return pd.Series(model.predict(X_today), index=X_today.index)
        except Exception as e:
            logging.error(f"  > ❌ [RollingAITrainer] 预测失败: {e}")
            return pd.Series(dtype=float)


class AdversarialLLMCombiner(RollingCalculatorBase):
    """
    通过多智能体对抗（Multi-Agent Adversarial Debate）机制来动态决定因子的权重。
    
    该类使用两个具有不同投资理念的AI代理进行辩论：
    - Agent A (激进型投资经理): 关注Alpha收益最大化
    - Agent B (保守型风险经理): 关注风险控制和回撤管理
    
    通过多轮辩论达到最优权重分配，支持正负权重以表达对因子的不同观点。
    
    注意：这里的"调仓"实际上是指因子权重的更新频率，而非投资组合的实际调仓。
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
        初始化AdversarialLLMCombiner
        
        Args:
            api_url: OpenAI兼容API的URL
            api_key: API密钥
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
        调用LLM API获取响应
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            
        Returns:
            LLM的响应文本
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
        解析LLM的JSON响应，包含错误处理和重试机制
        
        Args:
            response_text: LLM的响应文本
            max_retries: 最大重试次数
            
        Returns:
            解析后的权重字典
        """
        for attempt in range(max_retries):
            try:
                # 尝试从响应中提取JSON部分
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
            self, historical_data_window: pd.DataFrame) -> Dict[str, float]:
        """
        使用多智能体对抗机制计算因子权重
        
        Args:
            historical_data_window: 历史数据窗口
            
        Returns:
            因子权重字典
        """
        # 准备因子指标数据
        factor_metrics = {}
        for fname in self.factor_names:
            # 获取因子数据和对应的forward return
            factor_cols = [
                col for col in historical_data_window.columns
                if col.startswith(fname)
            ]
            if not factor_cols:
                continue

            # 获取默认的forward return周期（如果有配置的话）
            return_period = 30  # 默认30天周期
            return_col = f'forward_return_{return_period}d'

            # 检查是否有该周期的forward return数据
            if return_col not in historical_data_window.columns:
                # 尝试找到最接近的forward return列
                forward_cols = [
                    col for col in historical_data_window.columns
                    if col.startswith('forward_return_')
                ]
                if forward_cols:
                    return_col = forward_cols[0]
                else:
                    continue

            # 计算IC序列和统计指标
            ic_data = historical_data_window[[
                fname, return_col
            ]].rename(columns={fname: 'factor_value'})
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
        """
        使用多智能体对抗机制计算因子权重
        
        Args:
            historical_data_window: 历史数据窗口
            
        Returns:
            因子权重字典
        """
        # 准备因子指标数据
        factor_metrics = {}
        for fname in self.factor_names:
            # 获取因子数据和对应的forward return
            factor_cols = [
                col for col in historical_data_window.columns
                if col.startswith(fname)
            ]
            if not factor_cols:
                continue

            # 获取默认的forward return周期（如果有配置的话）
            return_period = 30  # 默认30天周期
            return_col = f'forward_return_{return_period}d'

            # 检查是否有该周期的forward return数据
            if return_col not in historical_data_window.columns:
                # 尝试找到最接近的forward return列
                forward_cols = [
                    col for col in historical_data_window.columns
                    if col.startswith('forward_return_')
                ]
                if forward_cols:
                    return_col = forward_cols[0]
                else:
                    continue

            # 计算IC序列和统计指标
            ic_data = historical_data_window[[
                fname, return_col
            ]].rename(columns={fname: 'factor_value'})
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

        # 构建Agent A (Portfolio Manager)的提示
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

        # Agent A的第一轮建议
        try:
            agent_a_response = self._call_llm(agent_a_prompt,
                                              agent_a_system_prompt)
            agent_a_weights = self._parse_json_response(agent_a_response)

            # 记录原始模型输出作为info信息
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
            # 构建Agent B (Risk Manager)的提示
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

            # 构建Agent B的提示
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

                # 记录原始模型输出作为info信息
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
        根据计算出的权重合成因子
        
        Args:
            payload: 因子权重字典
            daily_factors: 当日因子值
            
        Returns:
            合成后的因子值Series
        """
        try:
            # 将权重字典转换为Series并对其索引
            weights = pd.Series(payload).reindex(daily_factors.columns,
                                                 fill_value=0)

            # 归一化权重确保绝对值总和为1
            weight_abs_sum = weights.abs().sum()
            if weight_abs_sum > 0:
                weights = weights / weight_abs_sum

            # 计算加权合成因子
            combined_factor = (daily_factors * weights).sum(axis=1)
            return combined_factor
        except Exception as e:
            logging.error(f"因子合成失败: {e}")
            # 回退到等权合成
            return daily_factors.sum(axis=1)
