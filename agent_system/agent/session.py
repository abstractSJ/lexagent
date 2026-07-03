"""
Agent 会话状态封装。

AgentSession 是 ChatSession 的上层版本：它仍然维护用户和助手的文本历史，
但单轮回复允许模型调用本地工具。工具调用过程通过 AgentEvent 暴露，
不直接写入 Chat-style messages，避免把 Responses API 的工具 item 和聊天历史混在一起。
"""

from agent_system.agent.events import AgentEvent
from agent_system.agent.runner import AgentRunOptions, AgentRunner
from agent_system.llm.messages import assistant_message, system_message, user_message
from agent_system.llm.openai_client import Message


class AgentSession:
    """
    支持本地工具调用的 Agent 会话。

    Args:
        runner: AgentRunner，负责执行单轮工具调用循环。
        system_prompt: 可选系统提示词。传入后会作为第一条 system message 保存。
    """

    def __init__(self, runner: AgentRunner, system_prompt: str | None = None) -> None:
        """
        初始化 Agent 会话。

        Args:
            runner: AgentRunner 实例。
            system_prompt: 可选系统提示词。
        """

        self.runner = runner
        self.messages: list[Message] = []

        if system_prompt:
            self.messages.append(system_message(system_prompt))

    def ask(self, text: str, *, options: AgentRunOptions | None = None) -> str:
        """
        发送一轮用户输入，允许模型调用工具，并返回最终文本。

        Args:
            text: 用户输入。
            options: 可选分阶段 LLM 参数。为 None 时沿用底层客户端默认配置。

        Returns:
            str: 模型最终回复文本。
        """

        answer, _ = self.ask_with_events(text, options=options)
        return answer

    def ask_with_events(
        self,
        text: str,
        *,
        options: AgentRunOptions | None = None,
    ) -> tuple[str, list[AgentEvent]]:
        """
        发送一轮用户输入，返回最终文本和过程事件。

        Args:
            text: 用户输入。
            options: 可选分阶段 LLM 参数。为 None 时沿用底层客户端默认配置。

        Returns:
            tuple[str, list[AgentEvent]]: 最终文本和 Agent 过程事件。

        Raises:
            Exception: 底层模型调用或工具调用循环失败时向上抛出。
        """

        current_user_message = user_message(text)
        self.messages.append(current_user_message)

        try:
            answer, events = self.runner.run(self.messages, options=options)
        except Exception:
            # 请求失败时回滚当前 user 消息。
            # 原因是失败请求没有形成有效 assistant 回复，保留它会污染后续 Agent 上下文。
            if self.messages and self.messages[-1] is current_user_message:
                self.messages.pop()
            raise

        self.messages.append(assistant_message(answer))
        return answer, events

    def history(self) -> list[Message]:
        """
        返回当前会话历史的浅拷贝。

        Returns:
            list[Message]: 当前 messages 历史。
        """

        return [dict(message) for message in self.messages]

    def clear(self, keep_system: bool = True) -> None:
        """
        清空会话历史。

        Args:
            keep_system: 是否保留开头的 system message。
        """

        if not keep_system:
            self.messages.clear()
            return

        if self.messages and self.messages[0].get("role") == "system":
            self.messages[:] = [self.messages[0]]
        else:
            self.messages.clear()
