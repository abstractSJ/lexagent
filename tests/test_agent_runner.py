"""
AgentRunner / AgentSession 的本地单元测试。

重点验证：
1. Agent 工具循环能按阶段传不同的 LLM options。
2. AgentSession 会把 options 继续传给 Runner，并保持历史回滚逻辑正确。
"""

from __future__ import annotations

import unittest

from agent_system.agent.events import AgentEvent
from agent_system.agent.runner import AgentRunOptions, AgentRunner
from agent_system.agent.session import AgentSession
from agent_system.agent.tools import LocalTool, ToolRegistry
from agent_system.config import LLMCallOptions


class FakeRunner:
    """
    仅用于测试 AgentSession 的假 Runner。
    """

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.seen_calls: list[dict[str, object]] = []

    def run(self, messages, *, options=None):
        self.seen_calls.append(
            {
                "messages": [dict(message) for message in messages],
                "options": options,
            }
        )
        if self.error is not None:
            raise self.error
        return "完成", [AgentEvent(type="message_done", data={"text": "完成"})]


class FakeLLM:
    """
    仅用于测试 AgentRunner 的假 LLM。
    """

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.seen_create_calls: list[dict[str, object]] = []

    def build_responses_request(self, messages):
        return "你是助手。", [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": str(messages[-1].get("content", ""))}],
            }
        ]

    def create_response(self, *, input, instructions=None, tools=None, previous_response_id=None, options=None):
        self.seen_create_calls.append(
            {
                "input": input,
                "instructions": instructions,
                "tools": tools,
                "previous_response_id": previous_response_id,
                "options": options,
            }
        )
        if not self.responses:
            raise AssertionError("FakeLLM 没有更多预设 response。")
        return self.responses.pop(0)


class AgentRunnerTests(unittest.TestCase):
    """
    测试 AgentRunner 的分阶段 options 透传能力。
    """

    def build_tools(self) -> ToolRegistry:
        """
        创建最小可用工具注册表。
        """

        return ToolRegistry(
            [
                LocalTool(
                    name="echo",
                    description="返回传入文本。",
                    parameters={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                        },
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {
                        "ok": True,
                        "echo": str(arguments.get("text", "")),
                    },
                )
            ]
        )

    def test_runner_uses_stage_specific_options(self) -> None:
        """
        首轮工具规划和工具结果总结应分别使用 initial_options / tool_result_options。
        """

        llm = FakeLLM(
            responses=[
                {
                    "output": [
                        {
                            "type": "function_call",
                            "name": "echo",
                            "arguments": '{"text": "法条"}',
                            "call_id": "call-1",
                        }
                    ]
                },
                {
                    "output_text": "最终答案",
                    "output": [],
                },
            ]
        )
        runner = AgentRunner(llm=llm, tools=self.build_tools())
        run_options = AgentRunOptions(
            initial_options=LLMCallOptions(temperature=0.4, max_tokens=500),
            tool_result_options=LLMCallOptions(temperature=0.0, reasoning_effort="high", max_tokens=900),
        )

        answer, events = runner.run(
            [{"role": "user", "content": "请检索并回答。"}],
            options=run_options,
        )

        self.assertEqual("最终答案", answer)
        self.assertEqual(2, len(llm.seen_create_calls))
        self.assertEqual(run_options.initial_options, llm.seen_create_calls[0]["options"])
        self.assertEqual(run_options.tool_result_options, llm.seen_create_calls[1]["options"])
        self.assertIn("tool_call", [event.type for event in events])
        self.assertIn("tool_result", [event.type for event in events])
        self.assertTrue(
            any(item.get("type") == "function_call_output" for item in llm.seen_create_calls[1]["input"])
        )


class AgentSessionTests(unittest.TestCase):
    """
    测试 AgentSession 对 options 的透传与失败回滚。
    """

    def test_ask_with_events_passes_options_to_runner(self) -> None:
        """
        AgentSession 不应吞掉上层传入的分阶段 options。
        """

        runner = FakeRunner()
        session = AgentSession(runner=runner, system_prompt="你是助手。")
        options = AgentRunOptions(
            initial_options=LLMCallOptions(temperature=0.3),
            tool_result_options=LLMCallOptions(temperature=0.0),
        )

        answer, events = session.ask_with_events("请回答。", options=options)

        self.assertEqual("完成", answer)
        self.assertEqual(1, len(events))
        self.assertEqual(options, runner.seen_calls[0]["options"])
        self.assertEqual(3, len(session.history()))
        self.assertEqual("assistant", session.history()[-1]["role"])

    def test_ask_with_events_rolls_back_when_runner_fails(self) -> None:
        """
        即使新增了 options，Runner 失败时也不能污染后续 Agent 上下文。
        """

        runner = FakeRunner(error=RuntimeError("runner boom"))
        session = AgentSession(runner=runner, system_prompt="你是助手。")
        options = AgentRunOptions(initial_options=LLMCallOptions(temperature=0.8))

        with self.assertRaises(RuntimeError):
            session.ask_with_events("请补充案情。", options=options)

        self.assertEqual(1, len(session.history()))
        self.assertEqual("system", session.history()[0]["role"])
        self.assertEqual(options, runner.seen_calls[0]["options"])


if __name__ == "__main__":
    unittest.main()
