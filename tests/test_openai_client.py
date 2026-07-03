"""
OpenAIChatClient 的本地单元测试。

这些测试不会发起真实网络请求，而是通过替换 `client.responses.create` 捕获最终请求参数，
验证“默认配置 + 单次调用覆盖参数”的合并逻辑是否正确。
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from agent_system.config import LLMCallOptions, LLMConfig
from agent_system.llm.openai_client import OpenAIChatClient


class OpenAIChatClientTests(unittest.TestCase):
    """
    测试 OpenAIChatClient 的单次调用参数覆盖能力。
    """

    def build_client(
        self,
        *,
        model: str = "default-model",
        temperature: float = 0.7,
        reasoning_effort: str | None = "medium",
        max_tokens: int | None = 256,
    ) -> tuple[OpenAIChatClient, MagicMock]:
        """
        创建一个不会真实请求网络的测试客户端。

        Returns:
            tuple[OpenAIChatClient, MagicMock]: 客户端实例和被替换的 create mock。
        """

        config = LLMConfig(
            api_key="test-key",
            model=model,
            base_url="http://example.com/v1",
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            timeout=12.0,
            max_tokens=max_tokens,
        )
        client = OpenAIChatClient(config=config)
        create_mock = MagicMock()
        client.client = SimpleNamespace(
            responses=SimpleNamespace(create=create_mock)
        )
        return client, create_mock

    def test_chat_uses_default_config_params(self) -> None:
        """
        不传 options 时，应完整沿用客户端默认配置。
        """

        client, create_mock = self.build_client()
        create_mock.return_value = [
            {"type": "response.output_text.delta", "delta": "法"},
            {"type": "response.output_text.delta", "delta": "条"},
        ]

        chunks = list(
            client.chat(
                [{"role": "user", "content": "你好"}],
            )
        )

        self.assertEqual(["法", "条"], chunks)
        request = create_mock.call_args.kwargs
        self.assertEqual("default-model", request["model"])
        self.assertEqual(0.7, request["temperature"])
        self.assertEqual(256, request["max_output_tokens"])
        self.assertEqual({"effort": "medium"}, request["reasoning"])
        self.assertTrue(request["stream"])

    def test_chat_options_override_and_can_disable_reasoning(self) -> None:
        """
        单次调用应能覆盖 model/temperature/max_tokens，并显式关闭 reasoning。
        """

        client, create_mock = self.build_client(reasoning_effort="high", max_tokens=512)
        create_mock.return_value = []

        list(
            client.chat(
                [{"role": "user", "content": "请总结。"}],
                options=LLMCallOptions(
                    model="override-model",
                    temperature=0.2,
                    max_tokens=128,
                    disable_reasoning=True,
                ),
            )
        )

        request = create_mock.call_args.kwargs
        self.assertEqual("override-model", request["model"])
        self.assertEqual(0.2, request["temperature"])
        self.assertEqual(128, request["max_output_tokens"])
        self.assertNotIn("reasoning", request)

    def test_get_usage_respects_disable_max_tokens_and_reasoning_override(self) -> None:
        """
        usage 请求也应使用同一套覆盖逻辑，避免和 chat()/create_response() 行为不一致。
        """

        client, create_mock = self.build_client(reasoning_effort="medium", max_tokens=300)
        create_mock.return_value = SimpleNamespace(
            usage={"input_tokens": 10, "output_tokens": 5}
        )

        usage = client.get_usage(
            [{"role": "user", "content": "统计一下。"}],
            options=LLMCallOptions(
                reasoning_effort="low",
                disable_max_tokens=True,
            ),
        )

        request = create_mock.call_args.kwargs
        self.assertEqual({"effort": "low"}, request["reasoning"])
        self.assertNotIn("max_output_tokens", request)
        self.assertEqual(10, usage["prompt_tokens"])
        self.assertEqual(5, usage["completion_tokens"])
        self.assertEqual(15, usage["total_tokens"])

    def test_create_response_uses_override_and_preserves_tool_fields(self) -> None:
        """
        Agent 路径使用 create_response()，因此这里必须同时保留工具相关字段和覆盖参数。
        """

        client, create_mock = self.build_client(max_tokens=None, reasoning_effort=None)
        create_mock.return_value = {"id": "resp-1"}

        response = client.create_response(
            input=[{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            instructions="你是助手。",
            tools=[{"type": "function", "name": "demo", "parameters": {}}],
            previous_response_id="prev-1",
            options=LLMCallOptions(
                temperature=0.05,
                reasoning_effort="high",
                max_tokens=900,
            ),
        )

        self.assertEqual({"id": "resp-1"}, response)
        request = create_mock.call_args.kwargs
        self.assertEqual(0.05, request["temperature"])
        self.assertEqual({"effort": "high"}, request["reasoning"])
        self.assertEqual(900, request["max_output_tokens"])
        self.assertEqual("你是助手。", request["instructions"])
        self.assertEqual("prev-1", request["previous_response_id"])
        self.assertEqual("demo", request["tools"][0]["name"])

    def test_complete_passes_options_through_to_chat(self) -> None:
        """
        complete() 是 chat() 的便捷封装，因此也必须透传 options。
        """

        client, _ = self.build_client()
        client.chat = MagicMock(return_value=iter(["A", "B"]))
        options = LLMCallOptions(temperature=0.33, max_tokens=77)

        answer = client.complete(
            [{"role": "user", "content": "请回答。"}],
            options=options,
        )

        self.assertEqual("AB", answer)
        self.assertEqual(options, client.chat.call_args.kwargs["options"])

    def test_complete_non_stream_uses_non_stream_request(self) -> None:
        """
        complete_non_stream() 应直接发起非流式 Responses 请求，并提取 output_text。
        """

        client, create_mock = self.build_client(max_tokens=None, reasoning_effort=None)
        create_mock.return_value = {"output_text": "完整回答", "output": []}
        options = LLMCallOptions(temperature=0.11, max_tokens=123)

        answer = client.complete_non_stream(
            [{"role": "system", "content": "你是助手。"}, {"role": "user", "content": "请回答。"}],
            options=options,
        )

        self.assertEqual("完整回答", answer)
        request = create_mock.call_args.kwargs
        self.assertNotIn("stream", request)
        self.assertEqual(0.11, request["temperature"])
        self.assertEqual(123, request["max_output_tokens"])
        self.assertEqual("你是助手。", request["instructions"])


if __name__ == "__main__":
    unittest.main()
