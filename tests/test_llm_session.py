"""
ChatSession 的本地单元测试。

重点验证两件事：
1. 单次调用 options 会继续透传到底层 llm.chat()。
2. 新增参数后，原有历史写入和失败回滚行为仍保持正确。
"""

from __future__ import annotations

import unittest

from agent_system.config import LLMCallOptions
from agent_system.llm.session import ChatSession


class FakeLLM:
    """
    仅用于测试的假 LLM。
    """

    def __init__(self, chunks: list[str] | None = None, *, error: Exception | None = None) -> None:
        self.chunks = list(chunks or [])
        self.error = error
        self.seen_calls: list[dict[str, object]] = []

    def chat(self, messages, image_paths=None, *, image_detail="auto", options=None):
        self.seen_calls.append(
            {
                "method": "chat",
                "messages": [dict(message) for message in messages],
                "image_paths": image_paths,
                "image_detail": image_detail,
                "options": options,
            }
        )
        if self.error is not None:
            raise self.error
        for chunk in self.chunks:
            yield chunk

    def complete_non_stream(self, messages, image_paths=None, *, image_detail="auto", options=None):
        self.seen_calls.append(
            {
                "method": "complete_non_stream",
                "messages": [dict(message) for message in messages],
                "image_paths": image_paths,
                "image_detail": image_detail,
                "options": options,
            }
        )
        if self.error is not None:
            raise self.error
        return "".join(self.chunks)


class ChatSessionTests(unittest.TestCase):
    """
    测试 ChatSession 的 options 透传与回滚逻辑。
    """

    def test_ask_passes_options_to_llm(self) -> None:
        """
        ask() 应通过 stream_ask() 把 options 继续传给底层 llm.chat()。
        """

        llm = FakeLLM(["法", "律"])
        session = ChatSession(llm=llm, system_prompt="你是助手。")
        options = LLMCallOptions(temperature=0.25, max_tokens=200)

        answer = session.ask("请解释法条。", options=options)

        self.assertEqual("法律", answer)
        self.assertEqual(1, len(llm.seen_calls))
        self.assertEqual(options, llm.seen_calls[0]["options"])
        self.assertEqual(3, len(session.history()))
        self.assertEqual("assistant", session.history()[-1]["role"])

    def test_ask_non_stream_passes_options_and_writes_history(self) -> None:
        """
        ask_non_stream() 应走底层非流式接口，并在成功后写回 assistant 历史。
        """

        llm = FakeLLM(["非", "流", "式"])
        session = ChatSession(llm=llm, system_prompt="你是助手。")
        options = LLMCallOptions(temperature=0.1, max_tokens=300)

        answer = session.ask_non_stream("请用非流式回答。", options=options)

        self.assertEqual("非流式", answer)
        self.assertEqual("complete_non_stream", llm.seen_calls[0]["method"])
        self.assertEqual(options, llm.seen_calls[0]["options"])
        self.assertEqual(3, len(session.history()))
        self.assertEqual("assistant", session.history()[-1]["role"])

    def test_stream_ask_rolls_back_user_message_when_llm_fails(self) -> None:
        """
        即使新增了 options，底层失败时也不能把半截 user message 留在历史中。
        """

        llm = FakeLLM(error=RuntimeError("boom"))
        session = ChatSession(llm=llm, system_prompt="你是助手。")
        options = LLMCallOptions(temperature=0.9, reasoning_effort="low")

        with self.assertRaises(RuntimeError):
            list(session.stream_ask("请补充案情叙述。", options=options))

        self.assertEqual(1, len(session.history()))
        self.assertEqual("system", session.history()[0]["role"])
        self.assertEqual(options, llm.seen_calls[0]["options"])

    def test_ask_non_stream_rolls_back_user_message_when_llm_fails(self) -> None:
        """
        非流式接口失败时，也不能把当前 user message 留在历史中。
        """

        llm = FakeLLM(error=RuntimeError("boom"))
        session = ChatSession(llm=llm, system_prompt="你是助手。")

        with self.assertRaises(RuntimeError):
            session.ask_non_stream("请测试非流式。")

        self.assertEqual(1, len(session.history()))
        self.assertEqual("system", session.history()[0]["role"])


if __name__ == "__main__":
    unittest.main()
