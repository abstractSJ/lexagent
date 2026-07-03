"""
法律 query planner 的本地单元测试。

这些测试使用 fake LLM，不会调用真实模型接口。
测试重点是 JSON 提取、字段校验、规范化和一次修复重试逻辑。
"""

from __future__ import annotations

import unittest

from agent_system.config import LLMCallOptions
from agent_system.planning.legal_query_planner import (
    DEFAULT_PLANNER_LLM_OPTIONS,
    LegalQueryPlanError,
    extract_json_object,
    plan_legal_queries,
    validate_and_normalize_plan,
)


class FakeLLM:
    """
    仅用于测试的假 LLM。

    它会按顺序返回预设文本，便于验证 planner 的修复重试逻辑。
    """

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.seen_messages: list[list[dict[str, str]]] = []
        self.seen_options: list[LLMCallOptions | None] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        options: LLMCallOptions | None = None,
    ) -> str:
        if self.calls >= len(self.responses):
            raise AssertionError("FakeLLM 没有更多预设响应。")
        self.seen_messages.append(messages)
        self.seen_options.append(options)
        response = self.responses[self.calls]
        self.calls += 1
        return response


class ExtractJsonObjectTests(unittest.TestCase):
    """
    测试从 LLM 输出中提取 JSON object 的容错逻辑。
    """

    def test_extracts_plain_json_object(self) -> None:
        text = '{"global_queries": [], "issues": []}'
        self.assertEqual(text, extract_json_object(text))

    def test_extracts_json_inside_markdown_fence(self) -> None:
        text = '下面是结果：\n```json\n{"global_queries": [], "issues": []}\n```'
        self.assertEqual('{"global_queries": [], "issues": []}', extract_json_object(text))

    def test_raises_when_no_json_object_exists(self) -> None:
        with self.assertRaises(ValueError):
            extract_json_object("这里没有 JSON")


class ValidateAndNormalizePlanTests(unittest.TestCase):
    """
    测试本地字段校验和规范化逻辑。
    """

    def test_normalizes_and_deduplicates_query_plan(self) -> None:
        data = {
            "global_queries": [
                "  故意伤害他人身体  ",
                "故意伤害他人身体",
                "",
            ],
            "issues": [
                {
                    "issue": "故意伤害行为的刑事责任",
                    "facts": ["伤害他人身体", "伤害他人身体", " "],
                    "preferred_legal_names": ["刑法", "刑法"],
                    "queries": [
                        "故意伤害他人身体",
                        "故意伤害如何处罚",
                        "故意伤害罪 刑事责任",
                        "故意伤害他人身体",
                    ],
                    "positive_terms": ["故意伤害", "伤害他人身体", "故意伤害"],
                    "negative_terms": ["过失伤害", "过失伤害"],
                }
            ],
        }

        plan = validate_and_normalize_plan(data)
        self.assertEqual(["故意伤害他人身体"], plan.global_queries)
        self.assertEqual(1, len(plan.issues))
        issue = plan.issues[0]
        self.assertEqual(["伤害他人身体"], issue.facts)
        self.assertEqual(["刑法"], issue.preferred_legal_names)
        self.assertEqual(
            ["故意伤害他人身体", "故意伤害如何处罚", "故意伤害罪 刑事责任"],
            issue.queries,
        )
        self.assertEqual(["故意伤害", "伤害他人身体"], issue.positive_terms)
        self.assertEqual(["过失伤害"], issue.negative_terms)

    def test_filters_forbidden_queries_and_terms(self) -> None:
        data = {
            "global_queries": ["故意伤害他人身体"],
            "issues": [
                {
                    "issue": "故意伤害行为的刑事责任",
                    "facts": ["伤害他人身体"],
                    "preferred_legal_names": ["刑法"],
                    "queries": [
                        "故意伤害他人身体",
                        "故意伤害 处三年以下有期徒刑",
                        "刑法第二百三十四条 故意伤害",
                    ],
                    "positive_terms": ["故意伤害", "第八十二条"],
                    "negative_terms": ["过失伤害", "罚金"],
                }
            ],
        }

        plan = validate_and_normalize_plan(data)
        issue = plan.issues[0]
        self.assertEqual(["故意伤害他人身体"], issue.queries)
        self.assertEqual(["故意伤害"], issue.positive_terms)
        self.assertEqual(["过失伤害"], issue.negative_terms)
        self.assertTrue(plan.warnings)

    def test_raises_when_all_queries_become_invalid(self) -> None:
        data = {
            "global_queries": ["故意伤害他人身体"],
            "issues": [
                {
                    "issue": "故意伤害行为的刑事责任",
                    "facts": ["伤害他人身体"],
                    "preferred_legal_names": ["刑法"],
                    "queries": [
                        "故意伤害 处三年以下有期徒刑",
                        "刑法第二百三十四条 故意伤害",
                    ],
                    "positive_terms": ["故意伤害"],
                    "negative_terms": [],
                }
            ],
        }

        with self.assertRaises(ValueError):
            validate_and_normalize_plan(data)


class PlannerFlowTests(unittest.TestCase):
    """
    测试 planner 的一次调用、修复重试和最终失败路径。
    """

    def test_plan_succeeds_on_first_response(self) -> None:
        llm = FakeLLM(
            [
                '{'
                '"global_queries": ["开设赌场 诈骗 故意伤害 离婚后财产未分割"], '
                '"issues": ['
                '{'
                '"issue": "开设赌场行为的刑事责任", '
                '"facts": ["存在开设赌场行为"], '
                '"preferred_legal_names": ["刑法"], '
                '"queries": ["开设赌场", "开设赌场罪 刑事责任", "开设赌场如何处罚"], '
                '"positive_terms": ["开设赌场"], '
                '"negative_terms": []'
                '}'
                ']'
                '}'
            ]
        )

        plan = plan_legal_queries("我开设赌场。", llm=llm)
        self.assertEqual(1, llm.calls)
        self.assertEqual(1, len(plan.issues))
        self.assertEqual("开设赌场行为的刑事责任", plan.issues[0].issue)
        self.assertEqual([DEFAULT_PLANNER_LLM_OPTIONS], llm.seen_options)

    def test_plan_repairs_after_invalid_first_response(self) -> None:
        llm = FakeLLM(
            [
                "这不是合法 JSON",
                '{'
                '"global_queries": ["故意伤害他人身体 轻伤 处罚"], '
                '"issues": ['
                '{'
                '"issue": "故意伤害行为的刑事责任", '
                '"facts": ["伤害他人身体"], '
                '"preferred_legal_names": ["刑法"], '
                '"queries": ["故意伤害他人身体", "故意伤害如何处罚", "故意伤害罪 刑事责任"], '
                '"positive_terms": ["故意伤害", "伤害他人身体"], '
                '"negative_terms": ["过失伤害"]'
                '}'
                ']'
                '}'
            ]
        )

        plan = plan_legal_queries("故意伤害他人身体。", llm=llm, max_repair_attempts=1)
        self.assertEqual(2, llm.calls)
        self.assertEqual("故意伤害行为的刑事责任", plan.issues[0].issue)
        self.assertEqual([DEFAULT_PLANNER_LLM_OPTIONS, DEFAULT_PLANNER_LLM_OPTIONS], llm.seen_options)

    def test_plan_uses_custom_llm_options_in_all_attempts(self) -> None:
        llm = FakeLLM(
            [
                "这不是合法 JSON",
                '{'
                '"global_queries": ["故意伤害他人身体"], '
                '"issues": ['
                '{'
                '"issue": "故意伤害行为的刑事责任", '
                '"facts": ["伤害他人身体"], '
                '"preferred_legal_names": ["刑法"], '
                '"queries": ["故意伤害他人身体", "故意伤害如何处罚", "故意伤害罪 刑事责任"], '
                '"positive_terms": ["故意伤害"], '
                '"negative_terms": ["过失伤害"]'
                '}'
                ']'
                '}'
            ]
        )
        custom_options = LLMCallOptions(
            temperature=0.35,
            reasoning_effort="medium",
            max_tokens=900,
        )

        plan = plan_legal_queries(
            "故意伤害他人身体。",
            llm=llm,
            max_repair_attempts=1,
            llm_options=custom_options,
        )
        self.assertEqual(2, llm.calls)
        self.assertEqual("故意伤害行为的刑事责任", plan.issues[0].issue)
        self.assertEqual([custom_options, custom_options], llm.seen_options)

    def test_plan_raises_after_repair_failure(self) -> None:
        llm = FakeLLM([
            "这不是合法 JSON",
            '{"global_queries": [], "issues": []}',
        ])

        with self.assertRaises(LegalQueryPlanError):
            plan_legal_queries("故意伤害他人身体。", llm=llm, max_repair_attempts=1)
        self.assertEqual(2, llm.calls)


if __name__ == "__main__":
    unittest.main()
