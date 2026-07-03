"""
LLM 会话封装。

OpenAIChatClient 负责“单次请求怎么发给模型”，ChatSession 负责“多轮对话怎么维护历史”。
这样 main.py 或未来的 Agent 层就不需要自己 append user、拼接 assistant、失败回滚。
"""

from typing import Iterator, List, Sequence

from agent_system.config import LLMCallOptions
from agent_system.llm.messages import assistant_message, system_message, user_message
from agent_system.llm.openai_client import ImagePath, Message, OpenAIChatClient


class ChatSession:
    """
    一个轻量级多轮聊天会话。

    ChatSession 保存当前会话的 messages 历史，并把用户输入、图片输入、模型回复写回历史。
    它不负责长期记忆、工具调用或任务规划；这些能力以后可以在更高一层 Agent 中组合。

    Args:
        llm: 底层 LLM 客户端。
        system_prompt: 可选系统提示词。传入后会作为第一条 system message 保存。
    """

    def __init__(self, llm: OpenAIChatClient, system_prompt: str | None = None) -> None:
        """
        初始化聊天会话。

        Args:
            llm: 已经初始化好的 OpenAIChatClient。
            system_prompt: 可选系统提示词。
        """

        self.llm = llm
        self.messages: List[Message] = []

        if system_prompt:
            self.messages.append(system_message(system_prompt))

    def ask(
        self,
        text: str,
        image_paths: ImagePath | Sequence[ImagePath] | None = None,
        *,
        image_detail: str = "auto",
        options: LLMCallOptions | None = None,
    ) -> str:
        """
        使用流式接口发送一轮用户输入，并返回完整模型回复。

        Args:
            text: 用户输入文本。
            image_paths: 可选图片路径。需要模型看图时传入；不需要时保持 None。
            image_detail: 图片理解精度。常见值是 auto、low、high。
            options: 可选单次调用覆盖参数。为 None 时沿用当前会话绑定的默认 LLM 配置。

        Returns:
            str: 模型完整回复。
        """

        return "".join(
            self.stream_ask(
                text,
                image_paths=image_paths,
                image_detail=image_detail,
                options=options,
            )
        )

    def ask_non_stream(
        self,
        text: str,
        image_paths: ImagePath | Sequence[ImagePath] | None = None,
        *,
        image_detail: str = "auto",
        options: LLMCallOptions | None = None,
    ) -> str:
        """
        使用非流式接口发送一轮用户输入，并返回完整模型回复。

        Args:
            text: 用户输入文本。
            image_paths: 可选图片路径。需要模型看图时传入；不需要时保持 None。
            image_detail: 图片理解精度。常见值是 auto、low、high。
            options: 可选单次调用覆盖参数。为 None 时沿用当前会话绑定的默认 LLM 配置。

        Returns:
            str: 模型完整回复。
        """

        current_user_message = user_message(text)
        self.messages.append(current_user_message)

        try:
            answer = self.llm.complete_non_stream(
                self.messages,
                image_paths=image_paths,
                image_detail=image_detail,
                options=options,
            )
        except Exception:
            # 非流式失败时同样回滚当前 user 消息。
            # 原因是失败请求没有形成有效 assistant 回复，保留它会污染后续上下文。
            if self.messages and self.messages[-1] is current_user_message:
                self.messages.pop()
            raise

        self.messages.append(assistant_message(answer))
        return answer

    def stream_ask(
        self,
        text: str,
        image_paths: ImagePath | Sequence[ImagePath] | None = None,
        *,
        image_detail: str = "auto",
        options: LLMCallOptions | None = None,
    ) -> Iterator[str]:
        """
        发送一轮用户输入，并流式返回模型回复片段。

        这个方法会在模型完整回复结束后，把 assistant 消息写回历史。
        如果调用失败，或者调用方中途停止消费流式输出，会回滚刚加入的 user 消息。

        Args:
            text: 用户输入文本。
            image_paths: 可选图片路径。图片不会被永久写入 messages，只会在本次请求中临时传给 LLM。
            image_detail: 图片理解精度。常见值是 auto、low、high。
            options: 可选单次调用覆盖参数。为 None 时沿用当前会话绑定的默认 LLM 配置。

        Yields:
            str: 模型流式输出的文本片段。
        """

        current_user_message = user_message(text)
        self.messages.append(current_user_message)

        answer_parts: list[str] = []
        completed = False

        try:
            for chunk in self.llm.chat(
                self.messages,
                image_paths=image_paths,
                image_detail=image_detail,
                options=options,
            ):
                answer_parts.append(chunk)
                yield chunk

            completed = True
        except Exception:
            # 请求失败时必须回滚当前 user 消息。
            # 原因是失败请求没有形成有效的 assistant 回复，保留它会让历史变成“有问无答”的半截状态。
            if self.messages and self.messages[-1] is current_user_message:
                self.messages.pop()
            raise
        finally:
            # 如果调用方中途 break，没有消费完整个生成器，也不能把半截 user 消息留在历史中。
            if not completed and self.messages and self.messages[-1] is current_user_message:
                self.messages.pop()

        self.messages.append(assistant_message("".join(answer_parts)))

    def history(self) -> List[Message]:
        """
        返回当前会话历史的浅拷贝。

        Returns:
            List[Message]: 当前 messages 历史。返回拷贝是为了避免外部代码误改会话内部状态。
        """

        return [dict(message) for message in self.messages]

    def clear(self, keep_system: bool = True) -> None:
        """
        清空会话历史。

        Args:
            keep_system: 是否保留开头的 system message。默认保留，因为系统提示通常代表会话身份设定。
        """

        if not keep_system:
            self.messages.clear()
            return

        if self.messages and self.messages[0].get("role") == "system":
            self.messages[:] = [self.messages[0]]
        else:
            self.messages.clear()
