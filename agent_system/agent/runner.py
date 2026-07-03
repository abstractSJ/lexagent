"""
Agent 工具调用执行循环。

Runner 负责一轮“模型请求 -> 本地工具调用 -> 工具结果回传 -> 最终回答”的闭环。
它不保存长期历史，原因是历史属于 AgentSession；Runner 只负责把当前 messages 跑到最终结果。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable

from agent_system.agent.events import AgentEvent
from agent_system.agent.tools import ToolRegistry
from agent_system.config import LLMCallOptions
from agent_system.llm.openai_client import Message, OpenAIChatClient


@dataclass(frozen=True)
class AgentRunOptions:
    """
    Agent 单轮执行时的分阶段 LLM 参数。

    Attributes:
        initial_options: 首次模型请求使用的单次调用覆盖参数。
            这一阶段通常负责理解用户问题、决定是否调用工具以及生成工具参数。
        tool_result_options: 工具结果回传后的后续模型请求参数。
            法律 RAG 场景下，这一阶段更适合低温、受约束的总结参数，
            避免把工具选择阶段的高温误带到最终回答。
    """

    initial_options: LLMCallOptions | None = None
    tool_result_options: LLMCallOptions | None = None


class AgentRunner:
    """
    非流式 Agent 工具调用 Runner。

    Args:
        llm: 底层 LLM 客户端。
        tools: 本地工具注册表。
        max_tool_steps: 单轮最多允许多少轮工具调用，防止模型反复调用工具导致死循环。
    """

    def __init__(
        self,
        llm: OpenAIChatClient,
        tools: ToolRegistry,
        *,
        max_tool_steps: int = 5,
    ) -> None:
        """
        初始化 AgentRunner。

        Args:
            llm: 底层 LLM 客户端。
            tools: 本地工具注册表。
            max_tool_steps: 最大工具调用轮数。
        """

        if max_tool_steps < 1:
            raise ValueError("max_tool_steps 必须大于等于 1。")

        self.llm = llm
        self.tools = tools
        self.max_tool_steps = max_tool_steps

    def run(
        self,
        messages: list[Message],
        *,
        options: AgentRunOptions | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> tuple[str, list[AgentEvent]]:
        """
        执行一轮 Agent 工具调用。

        Args:
            messages: 当前会话历史，使用项目内部 Chat-style message 格式。
            options: 可选分阶段 LLM 参数。为 None 时沿用底层客户端默认配置。
            on_delta: 可选文本增量回调。只有注册表没有任何工具时才走真正的流式路径；
                有工具时仍执行非流式工具循环并忽略该回调。原因是工具调用需要读取完整
                response 里的 function_call item，与逐字流式输出是两种互斥的请求形态。

        Returns:
            tuple[str, list[AgentEvent]]: 最终文本和过程事件列表。

        Raises:
            RuntimeError: 模型没有返回最终文本，或工具调用超过最大步数时抛出。
        """

        openai_tools = self.tools.to_openai_tools()
        if on_delta is not None and not openai_tools:
            return self._run_streaming(messages, options=options, on_delta=on_delta)

        instructions, response_input = self.llm.build_responses_request(messages)
        events: list[AgentEvent] = []

        # conversation_input 是手动维护的 Responses input 链。
        # 原因是有些 OpenAI-compatible 服务虽然支持 function_call，
        # 但不能可靠用 previous_response_id 关联 function_call_output。
        # 把原始 input、function_call item、function_call_output 都显式传回去，兼容性更好，也更方便调试。
        conversation_input: list[dict[str, Any]] = list(response_input)

        response = self.llm.create_response(
            input=conversation_input,
            instructions=instructions,
            tools=openai_tools,
            options=options.initial_options if options is not None else None,
        )

        for step_index in range(self.max_tool_steps + 1):
            function_calls = self._extract_function_calls(response)
            if not function_calls:
                final_text = self._extract_output_text(response)
                if not final_text:
                    raise RuntimeError("模型没有返回最终文本。")

                events.append(
                    AgentEvent(
                        type="message_done",
                        data={
                            "response_id": self._get_attr(response, "id"),
                            "text": final_text,
                        },
                    )
                )
                return final_text, events

            if step_index >= self.max_tool_steps:
                raise RuntimeError(f"工具调用超过最大步数：{self.max_tool_steps}")

            tool_outputs = []
            for function_call in function_calls:
                conversation_input.append(self._function_call_to_input_item(function_call))
                tool_output = self._run_function_call(function_call, events)
                tool_outputs.append(tool_output)

            conversation_input.extend(tool_outputs)

            response = self.llm.create_response(
                input=conversation_input,
                instructions=instructions,
                tools=openai_tools,
                options=options.tool_result_options if options is not None else None,
            )

        raise RuntimeError(f"工具调用超过最大步数：{self.max_tool_steps}")

    def _run_streaming(
        self,
        messages: list[Message],
        *,
        options: AgentRunOptions | None,
        on_delta: Callable[[str], None],
    ) -> tuple[str, list[AgentEvent]]:
        """
        执行一轮无工具的流式回答生成。

        Args:
            messages: 当前会话历史。
            options: 可选分阶段 LLM 参数；流式路径没有工具阶段，只使用 initial_options。
            on_delta: 文本增量回调，每个片段生成时立即调用。

        Returns:
            tuple[str, list[AgentEvent]]: 完整最终文本和过程事件列表。

        Raises:
            RuntimeError: 流式请求失败或模型没有返回任何文本时抛出。
        """

        text_parts: list[str] = []
        stream = self.llm.chat(
            messages,
            options=options.initial_options if options is not None else None,
        )
        for delta in stream:
            text_parts.append(delta)
            on_delta(delta)

        final_text = "".join(text_parts)
        if not final_text:
            raise RuntimeError("模型没有返回最终文本。")

        # 流式事件接口不保证提供 response id；这里保持 message_done 数据结构和非流式路径一致，
        # 让上层消费者不需要区分两种执行路径。
        events = [
            AgentEvent(
                type="message_done",
                data={"response_id": None, "text": final_text},
            )
        ]
        return final_text, events

    def _function_call_to_input_item(self, function_call: dict[str, Any]) -> dict[str, Any]:
        """
        将归一化 function_call 转回 Responses input item。

        Args:
            function_call: 已归一化的工具调用数据。

        Returns:
            dict[str, Any]: 可放回 Responses input 的 function_call item。
        """

        item = {
            "type": "function_call",
            "name": function_call.get("name", ""),
            "arguments": function_call.get("arguments", "{}"),
            "call_id": function_call.get("call_id"),
        }

        # id/status 是可选字段。保留它们有助于兼容更严格的 Responses 实现和后续调试。
        if function_call.get("id"):
            item["id"] = function_call["id"]
        if function_call.get("status"):
            item["status"] = function_call["status"]

        return item

    def _run_function_call(
        self,
        function_call: dict[str, Any],
        events: list[AgentEvent],
    ) -> dict[str, Any]:
        """
        执行一个 function_call，并构造 function_call_output。

        Args:
            function_call: 已归一化的 function_call 数据。
            events: 过程事件列表，会在内部追加 tool_call 和 tool_result。

        Returns:
            dict[str, Any]: Responses API 的 function_call_output item。

        Raises:
            RuntimeError: function_call 缺少 call_id 时抛出。
        """

        name = str(function_call.get("name", ""))
        call_id = function_call.get("call_id")
        arguments_text = function_call.get("arguments", "{}")

        if not call_id:
            raise RuntimeError(f"工具调用缺少 call_id：{name}")

        arguments, parse_error = self._parse_tool_arguments(arguments_text)

        events.append(
            AgentEvent(
                type="tool_call",
                data={
                    "name": name,
                    "call_id": call_id,
                    "arguments": arguments,
                    "raw_arguments": arguments_text,
                },
            )
        )

        if parse_error:
            result = {"ok": False, "error": parse_error}
        else:
            result = self.tools.run(name, arguments)

        events.append(
            AgentEvent(
                type="tool_result",
                data={
                    "name": name,
                    "call_id": call_id,
                    "result": result,
                },
            )
        )

        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(result, ensure_ascii=False, default=str),
        }

    def _parse_tool_arguments(self, arguments_text: Any) -> tuple[dict[str, Any], str | None]:
        """
        解析模型返回的工具参数。

        Args:
            arguments_text: function_call.arguments，通常是 JSON 字符串。

        Returns:
            tuple[dict[str, Any], str | None]: 参数字典和错误信息；解析成功时错误信息为 None。
        """

        if arguments_text is None:
            return {}, None

        if not isinstance(arguments_text, str):
            return {}, "工具参数必须是 JSON 字符串。"

        try:
            arguments = json.loads(arguments_text or "{}")
        except json.JSONDecodeError as error:
            return {}, f"工具参数不是合法 JSON：{error}"

        if not isinstance(arguments, dict):
            return {}, "工具参数必须解析为 JSON object。"

        return arguments, None

    def _extract_function_calls(self, response: Any) -> list[dict[str, Any]]:
        """
        从 Responses response.output 中提取 function_call item。

        Args:
            response: OpenAI SDK response 对象或兼容 dict。

        Returns:
            list[dict[str, Any]]: 归一化后的工具调用列表。
        """

        function_calls: list[dict[str, Any]] = []
        for item in self._get_output_items(response):
            if self._get_attr(item, "type") != "function_call":
                continue

            function_calls.append(
                {
                    "name": self._get_attr(item, "name", ""),
                    "arguments": self._get_attr(item, "arguments", "{}"),
                    "call_id": self._get_attr(item, "call_id"),
                    "id": self._get_attr(item, "id"),
                    "status": self._get_attr(item, "status"),
                }
            )

        return function_calls

    def _extract_output_text(self, response: Any) -> str:
        """
        从 Responses response 中提取最终文本。

        Args:
            response: OpenAI SDK response 对象或兼容 dict。

        Returns:
            str: 最终文本。没有文本时返回空字符串。
        """

        output_text = self._get_attr(response, "output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        text_parts: list[str] = []
        for item in self._get_output_items(response):
            if self._get_attr(item, "type") != "message":
                continue

            content_items = self._get_attr(item, "content", []) or []
            for content_item in content_items:
                if self._get_attr(content_item, "type") == "output_text":
                    text = self._get_attr(content_item, "text", "")
                    if isinstance(text, str):
                        text_parts.append(text)

        return "".join(text_parts)

    def _get_output_items(self, response: Any) -> list[Any]:
        """
        读取 response.output。

        Args:
            response: OpenAI SDK response 对象或兼容 dict。

        Returns:
            list[Any]: output item 列表。
        """

        output = self._get_attr(response, "output", [])
        if output is None:
            return []
        return list(output)

    def _get_attr(self, obj: Any, name: str, default: Any = None) -> Any:
        """
        兼容 dict 与 SDK 对象的属性读取。

        Args:
            obj: dict 或 SDK 对象。
            name: 字段名。
            default: 默认值。

        Returns:
            Any: 字段值。
        """

        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)
