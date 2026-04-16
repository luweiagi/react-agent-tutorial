import re
from llm_client import LLMClient
from tools import ToolExecutor


# ReAct 的系统提示词：定义了 Thought-Action-Observation 的交互格式
# 这是整个 ReAct 机制的"灵魂"——通过精心设计的 prompt 引导 LLM
# 在每一步显式地输出推理过程（Thought）和要执行的动作（Action）
REACT_SYSTEM_PROMPT = """\
你是一个能够使用工具来解决问题的智能助手。

可用工具：
{tools}

你必须严格按照以下格式回复，每次只输出一组 Thought 和 Action：

Thought: <你对当前情况的分析和推理>
Action: <工具名>[<输入参数>]

当你已经收集到足够信息可以给出最终答案时，使用：

Thought: <总结推理过程>
Action: Finish[<最终答案>]

示例 1 —— 需要搜索的问题：

问题：爱因斯坦是哪一年获得诺贝尔奖的？

Thought: 我需要查找爱因斯坦获得诺贝尔奖的年份，让我搜索一下。
Action: Search[爱因斯坦 诺贝尔奖]

Observation: 阿尔伯特·爱因斯坦于1921年获得诺贝尔物理学奖...

Thought: 根据搜索结果，爱因斯坦在1921年获得诺贝尔物理学奖。我可以给出答案了。
Action: Finish[爱因斯坦于1921年获得诺贝尔物理学奖。]

示例 2 —— 需要计算的问题：

问题：一个圆的半径是7厘米，它的面积是多少？

Thought: 圆的面积公式是 π * r²，半径r=7，我用计算器算一下。
Action: Calculator[3.14159 * 7 * 7]

Observation: 153.93791

Thought: 计算结果是约153.94平方厘米，可以给出答案了。
Action: Finish[半径为7厘米的圆，面积约为153.94平方厘米。]

规则：
1. 每次只输出一组 Thought + Action，然后等待 Observation
2. 不要自己编造 Observation
3. Action 格式必须是 工具名[输入]
4. 尽量在 {max_steps} 步内完成
5. 事实类问题：先 Search 获取候选网页，再用 ReadWeb 读取 1-5 个高相关链接
6. 使用 Search 时，这是真实的网页搜索（聚合多个搜索源），支持任意自然语言。
   你可以输入完整的短语或问题，也可以输入核心关键词。
7. 禁止重复：不要用相同关键词重复搜索，不要重读同一 URL；未命中时改写关键词
8. 始终围绕用户原问题作答；信息不足时基于已有信息给出答案并注明不确定性
9. 计算/公式类问题优先使用 Calculator，不要先去 Search
10. 天气类问题优先使用 GetWeather 工具获取实时天气和温度等，而不是用 Search 搜索。
"""


def _build_user_prompt(question: str, history: list[str]) -> str:
    """构建每一轮发给 LLM 的 user prompt"""
    parts = [f"问题：{question}"]
    parts.append("当前唯一任务：只回答上面的原始问题，这是唯一的目的，不要扩展到其他主题。")
    if history:
        parts.append("\n下面是之前的推理过程：")
        parts.append("\n".join(history))
        parts.append("\n请继续输出下一步的 Thought 和 Action：")
    else:
        parts.append("请先输出第一步的 Thought 和 Action：")
    return "\n".join(parts)


def _extract_bracket_content(text: str) -> str:
    """提取最外层方括号内的内容，正确处理嵌套括号

    例如: "Finish[答案是 f(x) = x[0] + 1]" -> "答案是 f(x) = x[0] + 1"
    """
    start = text.find("[")
    # 未找到左括号，说明不是 ToolName[...] 结构，直接原样返回
    if start == -1:
        return text

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return text[start + 1 :]


class ReActAgent:
    """ReAct (Reasoning + Acting) 智能体

    核心循环:
        1. 将问题 + 历史轨迹发送给 LLM
        2. LLM 输出 Thought（推理）和 Action（动作）
        3. 解析 Action，调用对应工具，获得 Observation（观察）
        4. 将 Thought + Action + Observation 追加到历史轨迹
        5. 重复以上步骤，直到 LLM 输出 Finish[最终答案] 或达到最大步数
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: ToolExecutor,
        max_steps: int = 10,
        verbose: bool = True,
    ):
        self.llm = llm_client
        self.tools = tool_executor
        self.max_steps = max_steps
        self.verbose = verbose

    def run(self, question: str) -> str:
        """
        run() 不是“直接求答案”，而是一个受控的 ReAct 闭环执行器：
            让模型分步决策，程序负责执行与约束，历史负责记忆与纠偏。
        """
        # 初始化状态
        history: list[str] = []
        seen_actions: set[str] = set()
        invalid_action_count = 0

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  问题: {question}")
            print(f"{'='*60}")

        # 准备系统提示词
        system_prompt = REACT_SYSTEM_PROMPT.format(
            tools=self.tools.get_tools_description(),
            max_steps=self.max_steps,
        )

        # 进入循环（最多 max_steps 轮）
        for step in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"\n--- Step {step}/{self.max_steps} ---")

            # 构建每一轮发给 LLM 的 user prompt
            user_prompt = _build_user_prompt(question, history)
            # 构建 messages 列表，包含系统提示词和 user prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            try:
                # 调 self.llm.chat(...) 拿模型输出
                response = self.llm.chat(messages, verbose=self.verbose)
            except Exception as e:
                if self.verbose:
                    print(f"Observation: LLM 请求失败: {e}")
                return "LLM 请求失败，可能是网络波动或模型服务超时，请稍后重试。"

            # 解析模型输出，提取 Thought 和 Action
            thought, action_raw = self._parse_response(response)

            # 解析失败就给“纠错 Observation”，继续下一轮
            if not thought and not action_raw:
                if self.verbose:
                    print(f"[警告] 无法解析 LLM 输出，原始内容:\n{response}")
                history.append(
                    "Thought: (系统提示：上一次输出格式不正确，请严格按 Thought/Action 格式输出)"
                )
                continue

            # 执行动作或结束

            # 如果 Action 是 Finish[...]：直接返回最终答案
            # 否则校验格式、查重、执行工具，得到 Observation
            if self._is_finish(action_raw):
                final_answer = _extract_bracket_content(action_raw)
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"  最终答案: {final_answer}")
                    print(f"  (共经历 {step} 步推理)")
                    print(f"{'='*60}")
                return final_answer

            # 若格式不正确，则给“纠错 Observation”，继续下一轮
            if not self._is_valid_action_format(action_raw):
                invalid_action_count += 1
                observation = "Action 格式错误。请严格使用 ToolName[input]，例如 Search[爱因斯坦 出生日期]。"
                if self.verbose:
                    print(f"Observation: {observation}")
                history.append(f"Thought: {thought}")
                history.append(f"Action: {action_raw}")
                history.append(f"Observation: {observation}")
                if invalid_action_count >= 2:
                    if self.verbose:
                        print("[提示] 连续两次非法 Action，已提前终止以避免无意义循环。")
                    return "模型连续输出非法 Action，已提前停止。请重试或更换问题。"
                continue

            # 解析 Action，得到 tool_name 和 tool_input
            tool_name, tool_input = self._parse_action(action_raw)
            invalid_action_count = 0

            # 重复动作检测：归一化后用 set 去重，覆盖全部历史且 O(1) 查找
            # 归一化工具名和输入，忽略大小写/空白/标点，便于识别“本质重复”的 Action
            action_key = tool_name + ":" + re.sub(r"[\s\W]+", "", tool_input.lower())
            if action_key in seen_actions:
                observation = "检测到重复 Action，请更换策略：改写关键词、切换语言、或换一个不同 URL。"
                if self.verbose:
                    print(f"Observation: {observation}")
                history.append(f"Thought: {thought}")
                history.append(f"Action: {action_raw}")
                history.append(f"Observation: {observation}")
                continue
            seen_actions.add(action_key)

            # 执行工具，得到 Observation
            observation = self.tools.execute(tool_name, tool_input)

            if self.verbose:
                obs_display = observation[:200] + "..." if len(observation) > 200 else observation
                print(f"Observation: {obs_display}")

            # 把本轮结果写回 history
            # Thought、Action、Observation 全都追加
            # 下一轮模型就能“看到上一轮发生了什么”
            history.append(f"Thought: {thought}")
            history.append(f"Action: {action_raw}")
            history.append(f"Observation: {observation}")

        if self.verbose:
            print(f"\n[警告] 达到最大步数 ({self.max_steps})，强制结束。")
        return "未能在最大步数内完成推理。"

    # ==================== 输出解析 ====================
    # 这是 ReAct 工程实现中最脆弱的一环：
    # LLM 返回的是自由文本，需要用正则从中提取结构化的 Thought 和 Action

    def _parse_response(self, response: str) -> tuple[str, str]:
        """
        从 LLM 的自由文本响应中提取 Thought 和 Action
        
        例如：
        若 response 是：
            Thought: 我需要先搜索奥巴马出生年份
            Action: Search[奥巴马 出生年份]
        返回就是：
            thought = "我需要先搜索奥巴马出生年份"
            action = "Search[奥巴马 出生年份]"
        """
        # 提取 Thought
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=\nAction:|\Z)", response, re.DOTALL
        )
        action_match = re.search(r"Action:\s*(.+)", response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else ""
        return thought, action

    def _is_finish(self, action: str) -> bool:
        """
        判断 action 是否为 Finish[...] 形式的结束指令。
        """
        return bool(re.match(r"Finish\s*\[", action, re.IGNORECASE))

    def _is_valid_action_format(self, action: str) -> bool:
        """
        校验 action 是否符合 ToolName[...] 的基础格式。
        """
        return bool(re.match(r"^\w+\s*\[.*\]\s*$", action, re.DOTALL))

    def _parse_action(self, action_str: str) -> tuple[str, str]:
        """
        将 'ToolName[input]' 解析为 (tool_name, tool_input)

        例如：
            action_str = "Search[奥巴马 出生年份]"
        返回就是：
            tool_name = "Search"
            tool_input = "奥巴马 出生年份"
        """
        match = re.match(r"(\w+)\s*\[", action_str)
        if match:
            return match.group(1), _extract_bracket_content(action_str)
        return action_str, ""
