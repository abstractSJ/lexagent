"""
法律咨询业务链路的本地单元测试。

这些测试使用 fake LLM、fake planner、fake retriever 和 fake runner，不调用真实模型接口，
也不加载本地 BGE-M3。测试重点是多轮状态、案情拆解 + 多 query RAG、主会话不污染和失败回滚。
"""

from __future__ import annotations

import json
import threading
import time
import unittest
from unittest.mock import patch

from agent_system.agent.events import AgentEvent
from agent_system.config import LLMCallOptions
from agent_system.legal_consultation.models import (
    LegalAnalysisCatalog,
    LegalArticleEvidence,
    LegalCaseRagResult,
    LegalCaseState,
    LegalNextAction,
    LegalRiskFinding,
    LegalWebSearchItem,
    LegalWebSearchQueryResult,
    LegalWebSearchResearchResult,
)
from agent_system.legal_consultation.session import (
    DEFAULT_LEGAL_CONSULTATION_SYSTEM_PROMPT,
    LegalConsultationSession,
    build_reference_materials,
)
from agent_system.legal_consultation.subtasks import (
    LegalCaseAnalyzer,
    LegalCaseRagSubtask,
    LegalCaseStateUpdater,
    LegalDeterministicWebSearchSubtask,
    build_deterministic_web_search_queries,
    classify_web_authority_level,
    evidence_rank_score,
    legal_case_state_from_dict,
    merge_state_with_analysis,
    merge_text_lists,
    sort_evidences,
    web_search_items_from_tool_result,
)
from agent_system.planning.legal_query_planner import LegalIssueQuery, LegalQueryPlan


class FakeLLM:
    """
    仅用于测试结构化子任务的假 LLM。

    它按顺序返回预设响应，并记录 messages 与 options，便于验证内部 prompt 不会进入公开 history。
    响应条目可以是 Exception 实例：轮到该条目时抛出，用于模拟网络/接口传输失败。
    """

    def __init__(self, responses: list, *, usage_per_call: dict | None = None) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.seen_messages: list[list[dict[str, str]]] = []
        self.seen_options: list[LLMCallOptions | None] = []
        # 模拟真实客户端的 usage 累计接口，供轮级 metrics 测试读取调用次数和 token 消耗。
        self.usage_per_call = dict(usage_per_call or {})
        self.usage_totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def snapshot_usage_totals(self) -> dict[str, int]:
        """
        返回累计 usage 快照，与 OpenAIChatClient 的可观测性接口保持一致。
        """

        return dict(self.usage_totals)

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
        if isinstance(response, Exception):
            raise response
        self.usage_totals["calls"] += 1
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            self.usage_totals[key] += int(self.usage_per_call.get(key) or 0)
        return response


class FakePlanner:
    """
    仅用于测试 RAG 子任务的假 query planner。

    Args:
        plan: 预设返回的检索计划。
        error: 可选异常；传入后 plan() 抛出，用于验证规划失败时的链路降级。
    """

    def __init__(self, plan: LegalQueryPlan, *, error: Exception | None = None) -> None:
        self.plan_result = plan
        self.error = error
        self.seen_case_texts: list[str] = []

    def plan(self, case_text: str) -> LegalQueryPlan:
        self.seen_case_texts.append(case_text)
        if self.error is not None:
            raise self.error
        return self.plan_result


class FakeRetriever:
    """
    仅用于测试 RAG 子任务的假法条检索器。
    """

    def __init__(self) -> None:
        self.semantic_calls: list[dict[str, object]] = []
        self.keyword_calls: list[dict[str, object]] = []
        self.preload_calls: list[dict[str, object]] = []

    def preload(self, *, include_keyword_index: bool = True) -> None:
        """
        记录预热调用，不加载真实向量模型或关键词索引。
        """

        self.preload_calls.append({"include_keyword_index": include_keyword_index})

    def search_legal_articles(self, **kwargs):
        self.semantic_calls.append(dict(kwargs))
        return {
            "ok": True,
            "results": [
                {
                    "citation": "《劳动合同法》第八十二条",
                    "legal_name": "劳动合同法",
                    "article_no": "第八十二条",
                    "text": "用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。",
                    "score": 0.92,
                },
                {
                    "citation": "《劳动合同法》第三十九条",
                    "legal_name": "劳动合同法",
                    "article_no": "第三十九条",
                    "text": "劳动者存在特定过错时，用人单位可以解除劳动合同。",
                    "score": 0.65,
                },
            ],
        }

    def search_legal_articles_by_keyword(self, **kwargs):
        self.keyword_calls.append(dict(kwargs))
        return {
            "ok": True,
            "results": [
                {
                    "citation": "《劳动合同法》第八十二条",
                    "legal_name": "劳动合同法",
                    "article_no": "第八十二条",
                    "text": "用人单位未签书面劳动合同的二倍工资责任。",
                    "score": 1.0,
                }
            ],
        }


class FailingSemanticRetriever(FakeRetriever):
    """
    语义检索失败、关键词检索正常的假检索器。
    """

    def search_legal_articles(self, **kwargs):
        self.semantic_calls.append(dict(kwargs))
        return {"ok": False, "error": "semantic boom", "results": []}


class FakeRunner:
    """
    仅用于测试 LegalConsultationSession 的假最终回答 Runner。

    Args:
        error: 可选异常；传入后 run() 会抛出，用于验证失败回滚。
        answer_text: 预设最终回答文本。
        stream_chunk_size: 模拟流式时每个增量的字符数。使用一个和行边界无关的小块长，
            可以覆盖“一行被拆成多个 delta”的真实流式场景。
    """

    def __init__(
        self,
        *,
        error: Exception | None = None,
        answer_text: str | None = None,
        stream_chunk_size: int = 7,
        fail_times: int = 0,
    ) -> None:
        self.error = error
        self.answer_text = answer_text or "这是阶段性答复。以下内容仅作一般信息参考，不构成正式法律意见。"
        self.stream_chunk_size = max(1, int(stream_chunk_size))
        # fail_times 表示“前 N 次调用失败、之后成功”，用于验证最终回答的自动重试；
        # error 则表示每次调用都失败，用于验证重试耗尽后的回滚。
        self.fail_times = int(fail_times)
        self.seen_calls: list[dict[str, object]] = []

    def run(self, messages, *, options=None, on_delta=None):
        self.seen_calls.append(
            {
                "messages": [dict(message) for message in messages],
                "options": options,
                "streaming": on_delta is not None,
            }
        )
        if self.error is not None:
            raise self.error
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("模拟最终回答暂时失败")
        if on_delta is not None:
            for start in range(0, len(self.answer_text), self.stream_chunk_size):
                on_delta(self.answer_text[start : start + self.stream_chunk_size])
        return self.answer_text, [AgentEvent(type="message_done", data={"text": self.answer_text})]


class FakeWebSearchSubtask:
    """
    仅用于测试会话编排的假公网案例与司法实践检索子任务。
    """

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.seen_calls: list[dict[str, object]] = []

    def run(self, *, user_input, state, rag, risks, catalog, next_action):
        self.seen_calls.append(
            {
                "user_input": user_input,
                "state": state,
                "rag": rag,
                "risks": risks,
                "catalog": catalog,
                "next_action": next_action,
            }
        )
        if self.error is not None:
            raise self.error
        return LegalWebSearchResearchResult(
            query_results=[
                LegalWebSearchQueryResult(
                    purpose="similar_cases",
                    query="公司未签劳动合同 判决 典型案例 裁判文书",
                    ok=True,
                    results=[
                        LegalWebSearchItem(
                            title="未签劳动合同二倍工资案例",
                            url="https://example.test/case",
                            snippet="法院支持劳动者二倍工资请求。",
                            summary="公开案例摘要",
                            site_name="示例案例库",
                        )
                    ],
                ),
                LegalWebSearchQueryResult(
                    purpose="judicial_practice",
                    query="劳动合同法第八十二条 裁判规则 裁判要旨 司法实践",
                    ok=True,
                    results=[
                        LegalWebSearchItem(
                            title="二倍工资裁判规则",
                            url="https://example.test/practice",
                            snippet="司法实践关注入职时间、工资标准和仲裁时效。",
                            summary="裁判规则摘要",
                            site_name="示例法律站点",
                        )
                    ],
                ),
            ]
        )


class FakeWebSearchToolRunner:
    """
    仅用于测试确定性 web_search 子任务的假工具注册表。
    """

    def __init__(self, result: dict[str, object] | None = None, results_by_call: list[dict[str, object]] | None = None) -> None:
        self.result = result or {
            "ok": True,
            "results": [
                {
                    "title": "示例案例",
                    "url": "https://example.test/a",
                    "display_url": "example.test/a",
                    "snippet": "示例摘要",
                    "summary": "示例总结",
                    "site_name": "示例站点",
                    "date_published": "2026-01-01",
                    "date_last_crawled": "2026-07-01",
                }
            ],
        }
        self.results_by_call = list(results_by_call or [])
        self.calls: list[dict[str, object]] = []
        self.lock = threading.Lock()

    def run(self, name: str, arguments: dict[str, object]) -> object:
        with self.lock:
            call_index = len(self.calls)
            self.calls.append({"name": name, "arguments": dict(arguments)})
        if self.results_by_call:
            return self.results_by_call[min(call_index, len(self.results_by_call) - 1)]
        return self.result


class LegalCaseStateUpdaterTests(unittest.TestCase):
    """
    测试案件状态更新子任务。
    """

    def test_updates_empty_state_and_records_changes(self) -> None:
        """
        空状态应能根据用户输入初始化，并记录新增事实和修正事实。
        """

        llm = FakeLLM(
            [
                json.dumps(
                    {
                        "state": {
                            "summary": "用户主张公司未签劳动合同并被辞退。",
                            "parties": ["劳动者", "公司"],
                            "timeline": ["入职两年"],
                            "confirmed_facts": ["未签书面劳动合同"],
                            "disputed_facts": [],
                            "adverse_facts": [],
                            "contradictions": [],
                            "evidence_gaps": ["工资流水", "辞退通知"],
                            "user_goals": ["要求补偿"],
                            "legal_concepts": ["可能涉及二倍工资"],
                            "follow_up_questions": ["入职时间是什么时候？"],
                        },
                        "newly_added_facts": ["未签书面劳动合同"],
                        "changed_facts": ["从无状态初始化"],
                        "warnings": [],
                    },
                    ensure_ascii=False,
                )
            ]
        )
        updater = LegalCaseStateUpdater(llm)

        update = updater.update(
            previous_state=LegalCaseState(),
            public_messages=[],
            user_input="公司没签合同。",
        )

        self.assertEqual(1, update.state.version)
        self.assertEqual("用户主张公司未签劳动合同并被辞退。", update.state.summary)
        self.assertEqual(["未签书面劳动合同"], update.newly_added_facts)
        self.assertEqual(["从无状态初始化"], update.changed_facts)
        self.assertEqual(1, llm.calls)

    def test_parses_supplement_pause_fields(self) -> None:
        """
        状态更新器应能解析需要暂停补充的控制字段。
        """

        llm = FakeLLM(
            [
                json.dumps(
                    {
                        "state": {
                            "summary": "用户咨询交通事故赔偿，但缺少责任和损失信息。",
                            "parties": ["伤者", "对方司机"],
                            "timeline": [],
                            "confirmed_facts": ["发生交通事故"],
                            "disputed_facts": [],
                            "adverse_facts": [],
                            "contradictions": [],
                            "evidence_gaps": ["事故认定书", "医疗票据"],
                            "user_goals": ["了解赔偿范围"],
                            "legal_concepts": ["可能涉及侵权责任"],
                            "follow_up_questions": ["事故责任如何认定？"],
                        },
                        "newly_added_facts": ["发生交通事故"],
                        "changed_facts": [],
                        "warnings": [],
                        "should_pause_for_supplement": True,
                        "pause_reason": "缺少责任比例和损失信息，无法判断赔偿范围。",
                        "supplement_questions": ["事故责任认定如何？", "医疗费和误工损失是多少？"],
                        "supplement_evidence_gaps": ["事故认定书", "医疗票据"],
                    },
                    ensure_ascii=False,
                )
            ]
        )
        updater = LegalCaseStateUpdater(llm)

        update = updater.update(previous_state=LegalCaseState(), public_messages=[], user_input="交通事故怎么赔？")

        self.assertTrue(update.should_pause_for_supplement)
        self.assertEqual("缺少责任比例和损失信息，无法判断赔偿范围。", update.pause_reason)
        self.assertEqual(["事故责任认定如何？", "医疗费和误工损失是多少？"], update.supplement_questions)
        self.assertEqual(["事故认定书", "医疗票据"], update.supplement_evidence_gaps)


class CaseStateFieldMergeTests(unittest.TestCase):
    """
    测试案件状态字段解析与合并的“明确清空 / 缺失兜底 / 滚动窗口”语义。

    这三组行为共同保证长咨询下状态不会“错的删不掉、新的进不来”。
    """

    def test_explicit_empty_list_clears_previous_value(self) -> None:
        """
        模型明确输出空列表时应清空旧值，不得让上一轮旧条目复活。
        """

        previous = LegalCaseState(disputed_facts=["欠款金额有争议"], version=1)

        state = legal_case_state_from_dict(
            {"summary": "双方已确认欠款金额为五万元。", "disputed_facts": []},
            fallback=previous,
            version=2,
        )

        self.assertEqual([], state.disputed_facts)

    def test_missing_or_null_field_falls_back_to_previous_value(self) -> None:
        """
        字段缺失或值为 null 时视为模型没有更新该字段，应沿用上一轮旧值。
        """

        previous = LegalCaseState(
            disputed_facts=["金额有争议"],
            adverse_facts=["用户先动手"],
            version=1,
        )

        state = legal_case_state_from_dict(
            {"summary": "用户补充了新情况。", "adverse_facts": None},
            fallback=previous,
            version=2,
        )

        # disputed_facts 字段整体缺失，adverse_facts 值为 null，两者都应回退旧值。
        self.assertEqual(["金额有争议"], state.disputed_facts)
        self.assertEqual(["用户先动手"], state.adverse_facts)

    def test_merge_keeps_new_items_when_list_is_full(self) -> None:
        """
        旧列表攒满上限后，新条目仍应能进入，被挤出的是最老的条目。
        """

        old_items = [f"旧事实{index}" for index in range(20)]

        merged = merge_text_lists(old_items, ["本轮新发现"], limit=20)

        self.assertEqual(20, len(merged))
        self.assertIn("本轮新发现", merged)
        self.assertNotIn("旧事实0", merged)

    def test_merge_refreshes_duplicate_items_to_latest_position(self) -> None:
        """
        重复条目应按最后一次出现的位置保留：本轮再次提到的旧事实被“刷新”，不容易被挤出。
        """

        merged = merge_text_lists(
            ["旧事实0", "旧事实1", "旧事实2"],
            ["旧事实0", "新事实"],
            limit=3,
        )

        # “旧事实0”本轮再次出现，应移到新位置；被挤出的是最久没被提到的“旧事实1”。
        self.assertEqual(["旧事实2", "旧事实0", "新事实"], merged)

    def test_merge_state_with_analysis_admits_new_risks_when_full(self) -> None:
        """
        状态列表满载时，本轮风险识别的新发现仍应合并进案件状态。
        """

        state = LegalCaseState(
            adverse_facts=[f"历史不利事实{index}" for index in range(20)],
            version=3,
        )
        risks = [
            LegalRiskFinding(
                type="adverse_fact",
                severity="high",
                fact="本轮新识别的不利事实",
                reason="与在案证据矛盾",
                suggestion="尽快核实",
            )
        ]

        merged_state = merge_state_with_analysis(
            state=state,
            risks=risks,
            catalog=LegalAnalysisCatalog(),
        )

        self.assertIn("本轮新识别的不利事实", merged_state.adverse_facts)
        self.assertEqual(20, len(merged_state.adverse_facts))


class LegalCaseRagSubtaskTests(unittest.TestCase):
    """
    测试“案情拆解 + 多重 query RAG”合并子任务。
    """

    def test_runs_multi_query_rag_and_deduplicates_evidence(self) -> None:
        """
        子任务应复用 planner 的 issue/query，并对语义检索和关键词兜底结果去重合并。
        """

        plan = LegalQueryPlan(
            global_queries=["劳动争议 未签合同"],
            issues=[
                LegalIssueQuery(
                    issue="未签书面劳动合同的责任",
                    facts=["公司未签书面劳动合同"],
                    preferred_legal_names=["劳动合同法"],
                    queries=["未签劳动合同 二倍工资", "书面劳动合同 用人单位 责任", "劳动合同 未订立"],
                    positive_terms=["未签", "劳动合同", "二倍工资"],
                    negative_terms=[],
                )
            ],
            warnings=["planner warning"],
        )
        planner = FakePlanner(plan)
        retriever = FakeRetriever()
        subtask = LegalCaseRagSubtask(planner=planner, retriever=retriever)

        result = subtask.run(
            case_text="公司没签合同。",
            state=LegalCaseState(summary="未签劳动合同", confirmed_facts=["公司未签书面劳动合同"]),
        )

        self.assertEqual(1, len(planner.seen_case_texts))
        self.assertEqual([{"include_keyword_index": True}], retriever.preload_calls)
        self.assertEqual(3, len(retriever.semantic_calls))
        self.assertEqual(1, len(retriever.keyword_calls))
        self.assertEqual("劳动合同法", retriever.semantic_calls[0]["legal_name"])
        self.assertEqual(["未签", "劳动合同", "二倍工资"], retriever.keyword_calls[0]["keywords"])
        self.assertEqual(2, len(result.evidences))
        first = result.evidences[0]
        self.assertEqual("《劳动合同法》第八十二条", first.citation)
        self.assertGreater(first.hit_count, 1)
        self.assertIn("planner warning", result.warnings)

    def test_keyword_fallback_fans_out_to_multiple_preferred_legal_names(self) -> None:
        """
        关键词兜底应覆盖 planner 推荐的多部法律，避免只检索第一部法律导致跨法域条文漏召回。
        """

        plan = LegalQueryPlan(
            global_queries=["劳动争议 社保 仲裁"],
            issues=[
                LegalIssueQuery(
                    issue="未签合同、社保与仲裁程序的综合处理",
                    facts=["公司未签合同", "公司未缴社保"],
                    preferred_legal_names=["劳动合同法", "社会保险法", "劳动争议调解仲裁法"],
                    queries=["未签劳动合同 未缴社保 劳动仲裁"],
                    positive_terms=["未签", "社保", "仲裁"],
                    negative_terms=[],
                )
            ],
            warnings=[],
        )
        retriever = FakeRetriever()
        subtask = LegalCaseRagSubtask(planner=FakePlanner(plan), retriever=retriever, retrieval_workers=1)

        subtask.run(
            case_text="公司没签合同，也没交社保，我要仲裁。",
            state=LegalCaseState(summary="未签合同、未缴社保和劳动仲裁"),
        )

        # 关键词兜底沿用检索任务的法律名上限，避免复杂案情下 job 数量失控；关键是不能只取第一个。
        self.assertEqual(
            ["劳动合同法", "社会保险法"],
            [call["legal_name"] for call in retriever.keyword_calls],
        )
        self.assertTrue(all(call["keywords"] == ["未签", "社保", "仲裁"] for call in retriever.keyword_calls))

    def test_rag_collects_retrieval_warnings(self) -> None:
        """
        单条检索失败时，子任务应保留 warning 并继续合并其他可用证据。
        """

        plan = LegalQueryPlan(
            global_queries=["劳动争议"],
            issues=[
                LegalIssueQuery(
                    issue="未签书面劳动合同的责任",
                    facts=["公司未签书面劳动合同"],
                    preferred_legal_names=["劳动合同法"],
                    queries=["未签劳动合同 二倍工资"],
                    positive_terms=["未签", "劳动合同"],
                    negative_terms=[],
                )
            ],
            warnings=[],
        )
        retriever = FailingSemanticRetriever()
        subtask = LegalCaseRagSubtask(planner=FakePlanner(plan), retriever=retriever, retrieval_workers=1)

        result = subtask.run(
            case_text="公司没签合同。",
            state=LegalCaseState(summary="未签劳动合同"),
        )

        self.assertEqual(1, len(retriever.semantic_calls))
        self.assertEqual(1, len(retriever.keyword_calls))
        self.assertIn("语义检索未成功：未签劳动合同 二倍工资；原因：semantic boom", result.warnings)
        self.assertEqual(["《劳动合同法》第八十二条"], [item.citation for item in result.evidences])


class LegalDeterministicWebSearchSubtaskTests(unittest.TestCase):
    """
    测试确定性公网案例与司法实践检索子任务。
    """

    def build_inputs(self) -> dict[str, object]:
        """
        构造一组最小但完整的法律咨询上下文。
        """

        return {
            "user_input": "公司没签合同怎么办？",
            "state": LegalCaseState(
                summary="用户主张公司未签书面劳动合同。",
                confirmed_facts=["公司未签书面劳动合同"],
                legal_concepts=["可能涉及二倍工资"],
                user_goals=["要求补偿"],
            ),
            "rag": LegalCaseRagResult(
                query_plan=LegalQueryPlan(global_queries=["劳动争议"], issues=[], warnings=[]),
                issue_results=[],
                evidences=[
                    LegalArticleEvidence(
                        citation="《劳动合同法》第八十二条",
                        legal_name="劳动合同法",
                        article_no="第八十二条",
                        text="未签书面劳动合同二倍工资。",
                        issue="未签书面劳动合同责任",
                        source_query="未签劳动合同",
                        retrieval_type="semantic",
                    )
                ],
            ),
            "risks": [],
            "catalog": LegalAnalysisCatalog(
                case_points=["未签书面劳动合同"],
                legal_concepts=["二倍工资"],
                follow_up_questions=[],
            ),
            "next_action": LegalNextAction(action="ask_followup"),
        }

    def test_runs_three_fixed_purpose_queries_with_site_filters(self) -> None:
        """
        子任务每轮应检索相似案例、司法解释和裁判实务三类固定目的，并携带站点过滤参数。
        """

        tool_runner = FakeWebSearchToolRunner()
        subtask = LegalDeterministicWebSearchSubtask(tool_runner=tool_runner)

        result = subtask.run(**self.build_inputs())

        self.assertEqual(3, len(tool_runner.calls))
        self.assertEqual(
            ["similar_cases", "judicial_interpretation", "judicial_practice"],
            [item.purpose for item in result.query_results],
        )
        self.assertTrue(all(item.ok for item in result.query_results))
        self.assertEqual("示例案例", result.query_results[0].results[0].title)
        first_arguments = tool_runner.calls[0]["arguments"]
        self.assertEqual("web_search", tool_runner.calls[0]["name"])
        # 每条 query 多取候选给权威度重排留空间。
        self.assertEqual(10, first_arguments["count"])
        self.assertEqual("noLimit", first_arguments["freshness"])
        self.assertTrue(first_arguments["summary"])
        # 相似案例与裁判实务 query 服务端排除低质站点。
        self.assertIn("裁判文书", first_arguments["query"])
        self.assertIn("66law.cn", first_arguments["exclude"])
        # 司法解释 query 限定官方与专业法律站点，不再叠加 exclude。
        interpretation_arguments = tool_runner.calls[1]["arguments"]
        self.assertIn("司法解释", interpretation_arguments["query"])
        self.assertIn("gov.cn", interpretation_arguments["include"])
        self.assertNotIn("exclude", interpretation_arguments)
        practice_arguments = tool_runner.calls[2]["arguments"]
        self.assertIn("司法实践", practice_arguments["query"])
        # 案情材料出现“补偿”诉求时，实务 query 应追加赔偿口径词。
        self.assertIn("赔偿标准", practice_arguments["query"])

    def test_query_material_strips_speculative_concept_prefix(self) -> None:
        """
        query 核心词应剥掉“可能涉及”等推测性前缀，避免稀释搜索关键词。
        """

        specs = build_deterministic_web_search_queries(
            user_input="公司没签合同怎么办？",
            state=LegalCaseState(
                summary="用户主张公司未签书面劳动合同。",
                confirmed_facts=["公司未签书面劳动合同"],
                legal_concepts=["可能涉及二倍工资"],
                user_goals=["要求补偿"],
            ),
        )

        self.assertEqual(3, len(specs))
        self.assertIn("二倍工资", specs[0]["query"])
        self.assertNotIn("可能涉及", specs[0]["query"])
        self.assertIn("典型案例", specs[0]["query"])

    def test_collects_warning_when_tool_returns_error(self) -> None:
        """
        web_search 工具失败时应转成 warning，不抛异常中断主链路。
        """

        tool_runner = FakeWebSearchToolRunner(result={"ok": False, "error": "quota exceeded", "results": []})
        subtask = LegalDeterministicWebSearchSubtask(tool_runner=tool_runner, max_queries=2)

        result = subtask.run(**self.build_inputs())

        self.assertEqual(2, len(result.query_results))
        self.assertFalse(result.query_results[0].ok)
        self.assertIn("quota exceeded", result.query_results[0].error)
        self.assertGreaterEqual(len(result.warnings), 1)

    def test_run_accepts_preliminary_context_for_parallel_start(self) -> None:
        """
        主链路应能在风险和下一步判断完成前启动公网检索。
        """

        tool_runner = FakeWebSearchToolRunner()
        subtask = LegalDeterministicWebSearchSubtask(tool_runner=tool_runner, max_queries=2)
        inputs = self.build_inputs()

        result = subtask.run(
            user_input=inputs["user_input"],
            state=inputs["state"],
            rag=inputs["rag"],
            risks=None,
            catalog=None,
            next_action=None,
        )

        self.assertEqual(2, len(tool_runner.calls))
        self.assertEqual(
            ["similar_cases", "judicial_interpretation"],
            [item.purpose for item in result.query_results],
        )

    def test_parallel_web_search_keeps_order_and_deduplicates_urls(self) -> None:
        """
        多条公网检索可并发执行，但合并结果仍应按 query 顺序稳定去重。
        """

        tool_runner = FakeWebSearchToolRunner(
            results_by_call=[
                {
                    "ok": True,
                    "results": [
                        {
                            "title": "第一条案例",
                            "url": "https://example.test/shared",
                            "snippet": "第一条摘要",
                            "summary": "第一条总结",
                            "site_name": "站点一",
                        }
                    ],
                },
                {
                    "ok": True,
                    "results": [
                        {
                            "title": "重复案例",
                            "url": "https://example.test/shared",
                            "snippet": "重复摘要",
                            "summary": "重复总结",
                            "site_name": "站点二",
                        },
                        {
                            "title": "第二条案例",
                            "url": "https://example.test/second",
                            "snippet": "第二条摘要",
                            "summary": "第二条总结",
                            "site_name": "站点二",
                        },
                    ],
                },
            ]
        )
        subtask = LegalDeterministicWebSearchSubtask(tool_runner=tool_runner, max_queries=2, web_search_workers=2)

        result = subtask.run(**self.build_inputs())

        self.assertEqual(
            ["similar_cases", "judicial_interpretation"],
            [item.purpose for item in result.query_results],
        )
        self.assertEqual(["第一条案例"], [item.title for item in result.query_results[0].results])
        self.assertEqual(["第二条案例"], [item.title for item in result.query_results[1].results])


class WebSearchAuthorityRankingTests(unittest.TestCase):
    """
    测试公网结果的权威度分级、重排和低置信度过滤。
    """

    def build_raw_item(self, *, title: str, url: str, snippet: str = "", site_name: str = "") -> dict[str, object]:
        """
        构造一条最小的 web_search 工具原始结果。
        """

        return {
            "title": title,
            "url": url,
            "snippet": snippet,
            "summary": "",
            "site_name": site_name,
        }

    def test_classify_web_authority_level_by_domain(self) -> None:
        """
        域名分级应整段匹配后缀：官方站 high、专业库 medium、内容农场 low、拼接伪装域名不得蹭分。
        """

        self.assertEqual("high", classify_web_authority_level("https://wenshu.court.gov.cn/website/wenshu/181107"))
        self.assertEqual("high", classify_web_authority_level("https://www.chinacourt.org/article/detail/2026/01/id/1.shtml"))
        self.assertEqual("medium", classify_web_authority_level("https://www.pkulaw.com/case/12345.html"))
        self.assertEqual("low", classify_web_authority_level("https://www.66law.cn/question/1.html"))
        self.assertEqual("low", classify_web_authority_level("https://baijiahao.baidu.com/s?id=1"))
        self.assertEqual("normal", classify_web_authority_level("https://www.some-law-blog.example/a"))
        # fakegov.cn 结尾包含 gov.cn 字样，但不是 .gov.cn 子域，不能判为权威来源。
        self.assertEqual("normal", classify_web_authority_level("https://www.fakegov.cn/a"))

    def test_ranking_prefers_authoritative_and_drops_low_quality(self) -> None:
        """
        候选充足时，官方来源应排最前，低置信度站点应被直接淘汰。
        """

        result = {
            "ok": True,
            "results": [
                self.build_raw_item(title="华律网解读劳动合同纠纷问题", url="https://www.66law.cn/question/1.html"),
                self.build_raw_item(title="某博客谈未签合同", url="https://blog.example/a"),
                self.build_raw_item(
                    title="最高人民法院发布劳动争议典型案例",
                    url="https://www.court.gov.cn/zixun/xiangqing/1.html",
                ),
            ],
        }

        items = web_search_items_from_tool_result(result, seen_keys=set(), limit=5)

        self.assertEqual(
            ["最高人民法院发布劳动争议典型案例", "某博客谈未签合同"],
            [item.title for item in items],
        )
        self.assertEqual(["high", "normal"], [item.authority_level for item in items])

    def test_low_quality_backfills_only_when_results_scarce(self) -> None:
        """
        只有低置信度结果时允许兜底回填，避免资料栏整栏空白。
        """

        result = {
            "ok": True,
            "results": [
                self.build_raw_item(title="华律网文章一号解读内容", url="https://www.66law.cn/question/1.html"),
                self.build_raw_item(title="找法网文章二号解读内容", url="https://china.findlaw.cn/ask/2.html"),
                self.build_raw_item(title="律图文章三号解读内容", url="https://www.64365.com/ask/3.html"),
            ],
        }

        items = web_search_items_from_tool_result(result, seen_keys=set(), limit=5)

        # 兜底最多保留两条低置信度结果，而不是填满 limit。
        self.assertEqual(2, len(items))
        self.assertTrue(all(item.authority_level == "low" for item in items))

    def test_authority_content_keywords_boost_normal_sites(self) -> None:
        """
        同为一般站点时，标题含指导案例、规范案号等权威特征的结果应排在前面。
        """

        result = {
            "ok": True,
            "results": [
                self.build_raw_item(title="劳动纠纷经验分享长文", url="https://blog-a.example/a"),
                self.build_raw_item(
                    title="最高人民法院指导案例评析：（2023）京01民终1234号判决书要点",
                    url="https://blog-b.example/b",
                ),
            ],
        }

        items = web_search_items_from_tool_result(result, seen_keys=set(), limit=5)

        self.assertEqual(
            [
                "最高人民法院指导案例评析：（2023）京01民终1234号判决书要点",
                "劳动纠纷经验分享长文",
            ],
            [item.title for item in items],
        )

    def test_cross_query_title_dedup_removes_syndicated_copies(self) -> None:
        """
        同题文章被多站转载时，第二条 query 不应再次收录相同标题的结果。
        """

        seen_keys: set[str] = set()
        first = web_search_items_from_tool_result(
            {
                "ok": True,
                "results": [
                    self.build_raw_item(title="未签劳动合同二倍工资裁判规则综述", url="https://site-a.example/1"),
                ],
            },
            seen_keys=seen_keys,
            limit=5,
        )
        second = web_search_items_from_tool_result(
            {
                "ok": True,
                "results": [
                    self.build_raw_item(title="未签劳动合同二倍工资裁判规则综述", url="https://site-b.example/2"),
                    self.build_raw_item(title="另一篇不同标题的实务文章", url="https://site-c.example/3"),
                ],
            },
            seen_keys=seen_keys,
            limit=5,
        )

        self.assertEqual(1, len(first))
        self.assertEqual(["另一篇不同标题的实务文章"], [item.title for item in second])


class EvidenceRankingTests(unittest.TestCase):
    """
    测试法条证据的融合重排。
    """

    def build_evidence(
        self,
        *,
        citation: str,
        text: str = "",
        score: float | None = None,
        hit_count: int = 1,
    ) -> LegalArticleEvidence:
        """
        构造最小法条证据。
        """

        return LegalArticleEvidence(
            citation=citation,
            legal_name="测试法",
            article_no="第一条",
            text=text,
            issue="测试事项",
            source_query="测试query",
            retrieval_type="semantic",
            score=score,
            hit_count=hit_count,
        )

    def test_single_high_score_beats_repeated_low_scores(self) -> None:
        """
        单次高分精准命中应排在多次低分泛化命中之前；重复命中只作加分项。
        """

        precise = self.build_evidence(citation="《A法》第一条", score=0.9, hit_count=1)
        generic = self.build_evidence(citation="《B法》第二条", score=0.4, hit_count=2)

        ordered = sort_evidences([generic, precise])

        self.assertEqual(["《A法》第一条", "《B法》第二条"], [item.citation for item in ordered])
        self.assertGreater(
            evidence_rank_score(precise),
            evidence_rank_score(generic),
        )

    def test_positive_and_negative_terms_adjust_order(self) -> None:
        """
        planner 输出的正/反向词应参与重排：同分证据按加权词调序。
        """

        wanted = self.build_evidence(citation="《A法》第一条", text="竞业限制补偿金的支付标准", score=0.5)
        confusing = self.build_evidence(citation="《B法》第二条", text="保密义务的一般规定", score=0.5)

        ordered = sort_evidences(
            [confusing, wanted],
            positive_terms=["竞业限制"],
            negative_terms=["保密义务"],
        )

        self.assertEqual(["《A法》第一条", "《B法》第二条"], [item.citation for item in ordered])

    def test_semantic_low_score_tail_is_filtered_with_min_keep(self) -> None:
        """
        语义检索的低分长尾应被剔除，但每条 query 前两名保底保留。
        """

        class TailNoiseRetriever(FakeRetriever):
            """
            返回一条高分、一条保底低分和两条长尾结果的假检索器。
            """

            def search_legal_articles(self, **kwargs):
                self.semantic_calls.append(dict(kwargs))
                return {
                    "ok": True,
                    "results": [
                        {"citation": "《A法》第一条", "legal_name": "A法", "article_no": "第一条", "text": "高分命中", "score": 0.9},
                        {"citation": "《B法》第二条", "legal_name": "B法", "article_no": "第二条", "text": "保底低分", "score": 0.2},
                        {"citation": "《C法》第三条", "legal_name": "C法", "article_no": "第三条", "text": "长尾噪声", "score": 0.25},
                        {"citation": "《D法》第四条", "legal_name": "D法", "article_no": "第四条", "text": "长尾可用", "score": 0.5},
                    ],
                }

        plan = LegalQueryPlan(
            global_queries=[],
            issues=[
                LegalIssueQuery(
                    issue="测试事项",
                    facts=["事实"],
                    preferred_legal_names=[],
                    queries=["测试query"],
                    positive_terms=[],
                    negative_terms=[],
                )
            ],
            warnings=[],
        )
        retriever = TailNoiseRetriever()
        subtask = LegalCaseRagSubtask(planner=FakePlanner(plan), retriever=retriever, retrieval_workers=1)

        result = subtask.run(case_text="测试案情", state=LegalCaseState(summary="测试"))

        citations = [item.citation for item in result.evidences]
        self.assertIn("《A法》第一条", citations)
        self.assertIn("《D法》第四条", citations)
        self.assertIn("《B法》第二条", citations)
        # 排名第三之后且低于分数线的长尾噪声应被剔除。
        self.assertNotIn("《C法》第三条", citations)
        # 融合分数排序：0.9 > 0.5 > 0.2。
        self.assertEqual(["《A法》第一条", "《D法》第四条", "《B法》第二条"], citations)


class LegalCaseAnalyzerTests(unittest.TestCase):
    """
    测试风险、目录和下一步动作的合并分析子任务。
    """

    def build_empty_rag(self) -> LegalCaseRagResult:
        """
        构造无证据的最小 RAG 结果。
        """

        return LegalCaseRagResult(
            query_plan=LegalQueryPlan(global_queries=[], issues=[], warnings=[]),
            issue_results=[],
            evidences=[],
        )

    def test_parses_combined_analysis_with_single_llm_call(self) -> None:
        """
        一次 LLM 调用应同时解析出风险项、案情目录和下一步动作。
        """

        llm = FakeLLM(
            [
                json.dumps(
                    {
                        "risks": [
                            {
                                "type": "missing_evidence",
                                "severity": "high",
                                "fact": "缺少书面劳动合同",
                                "reason": "影响劳动关系认定",
                                "suggestion": "补充工资流水和工牌",
                            }
                        ],
                        "catalog": {
                            "case_points": ["未签书面劳动合同"],
                            "legal_concepts": ["可能涉及二倍工资"],
                            "follow_up_questions": ["入职时间是什么时候？"],
                        },
                        "next_action": {
                            "action": "answer_now",
                            "reasons": ["核心事实已经清楚"],
                            "questions_to_ask": [],
                            "should_correct_previous_answer": False,
                        },
                    },
                    ensure_ascii=False,
                )
            ]
        )
        analyzer = LegalCaseAnalyzer(llm)

        analysis = analyzer.analyze(
            state=LegalCaseState(summary="未签劳动合同争议"),
            rag=self.build_empty_rag(),
        )

        self.assertEqual(1, llm.calls)
        self.assertEqual(["missing_evidence"], [risk.type for risk in analysis.risks])
        self.assertEqual("high", analysis.risks[0].severity)
        self.assertEqual(["未签书面劳动合同"], analysis.catalog.case_points)
        self.assertEqual(["入职时间是什么时候？"], analysis.catalog.follow_up_questions)
        self.assertEqual("answer_now", analysis.next_action.action)
        self.assertFalse(analysis.next_action.should_correct_previous_answer)

    def test_invalid_next_action_downgrades_to_ask_followup(self) -> None:
        """
        非法 action 应降级为追问，缺失的 catalog 字段应降级为空列表。
        """

        llm = FakeLLM(
            [
                json.dumps(
                    {
                        "risks": [],
                        "catalog": {},
                        "next_action": {"action": "do_everything"},
                    },
                    ensure_ascii=False,
                )
            ]
        )
        analyzer = LegalCaseAnalyzer(llm)

        analysis = analyzer.analyze(state=LegalCaseState(), rag=self.build_empty_rag())

        self.assertEqual([], analysis.risks)
        self.assertEqual([], analysis.catalog.case_points)
        self.assertEqual("ask_followup", analysis.next_action.action)


class LegalConsultationFactoryTests(unittest.TestCase):
    """
    测试默认工厂的关键装配边界。
    """

    def test_default_factory_uses_no_tools_for_final_answer_runner(self) -> None:
        """
        最终回答阶段不应再开放检索工具，避免模型把工具结果长篇塞回聊天气泡。
        """

        from agent_system.agent.tools import LocalTool
        import agent_system.legal_consultation.factory as factory

        def dummy_handler(_arguments):
            return {"ok": True, "results": []}

        legal_tool = LocalTool(
            name="search_legal_articles",
            description="测试法条工具",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=dummy_handler,
        )
        web_tool = LocalTool(
            name="web_search",
            description="测试公网搜索工具",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=dummy_handler,
        )

        with (
            patch.object(factory, "build_llm_client", return_value=FakeLLM([])),
            patch.object(factory, "build_legal_retriever", return_value=FakeRetriever()),
            patch.object(factory, "build_legal_tools", return_value=[legal_tool]),
            patch.object(factory, "build_web_search_tools", return_value=[web_tool]),
        ):
            session = factory.create_legal_consultation_session()

        self.assertEqual([], session.answer_runner.tools.to_openai_tools())
        self.assertEqual(["search_legal_articles", "web_search"], [tool["name"] for tool in session.web_search_subtask.tool_runner.to_openai_tools()])


def build_valid_state_update_response() -> str:
    """
    构造状态更新子任务的合法 JSON 响应。
    """

    return json.dumps(
        {
            "state": {
                "summary": "用户存在劳动争议。",
                "parties": ["劳动者", "公司"],
                "timeline": ["工作两年"],
                "confirmed_facts": ["公司未签书面劳动合同"],
                "disputed_facts": [],
                "adverse_facts": [],
                "contradictions": [],
                "evidence_gaps": ["工资流水"],
                "user_goals": ["要求补偿"],
                "legal_concepts": ["可能涉及二倍工资"],
                "follow_up_questions": ["是否有工资流水？"],
            },
            "newly_added_facts": ["公司未签书面劳动合同"],
            "changed_facts": [],
            "warnings": [],
        },
        ensure_ascii=False,
    )


def build_valid_case_analysis_response() -> str:
    """
    构造综合分析子任务的合法 JSON 响应。
    """

    return json.dumps(
        {
            "risks": [
                {
                    "type": "missing_evidence",
                    "severity": "medium",
                    "fact": "缺少工资流水",
                    "reason": "会影响劳动关系和工资标准证明",
                    "suggestion": "补充银行流水或工资条",
                }
            ],
            "catalog": {
                "case_points": ["未签书面劳动合同"],
                "legal_concepts": ["可能涉及二倍工资"],
                "follow_up_questions": ["入职时间是什么？"],
            },
            "next_action": {
                "action": "ask_followup",
                "reasons": ["还缺少工资和入职时间"],
                "questions_to_ask": ["你什么时候入职？"],
                "should_correct_previous_answer": False,
            },
        },
        ensure_ascii=False,
    )


class SubtaskSelfHealTests(unittest.TestCase):
    """
    测试结构化子任务内部的自修复：JSON 解析失败的修复重试和传输失败的自动重试。
    """

    def build_empty_rag(self) -> LegalCaseRagResult:
        """
        构造空 RAG 结果，供综合分析器测试使用。
        """

        return LegalCaseRagResult(
            query_plan=LegalQueryPlan(global_queries=[], issues=[], warnings=[]),
            issue_results=[],
            evidences=[],
        )

    def test_state_updater_repairs_invalid_json(self) -> None:
        """
        首次输出不是合法 JSON 时，应带着原始输出和错误信息发起修复重试并成功。
        """

        llm = FakeLLM(["这不是JSON输出", build_valid_state_update_response()])
        updater = LegalCaseStateUpdater(llm)

        update = updater.update(previous_state=LegalCaseState(), public_messages=[], user_input="公司没签合同")

        self.assertEqual(1, update.state.version)
        self.assertEqual("用户存在劳动争议。", update.state.summary)
        self.assertEqual(2, llm.calls)
        self.assertTrue(any("修复" in warning for warning in update.warnings))
        # 修复 prompt 必须携带上一次的原始输出，模型才知道要改什么。
        repair_prompt = llm.seen_messages[1][-1]["content"]
        self.assertIn("这不是JSON输出", repair_prompt)

    def test_state_updater_retries_transport_error(self) -> None:
        """
        LLM 调用本身抛异常（网络/接口失败）时应自动重试一次并成功。
        """

        llm = FakeLLM([RuntimeError("网络中断"), build_valid_state_update_response()])
        updater = LegalCaseStateUpdater(llm)

        update = updater.update(previous_state=LegalCaseState(), public_messages=[], user_input="公司没签合同")

        self.assertEqual(1, update.state.version)
        self.assertEqual(2, llm.calls)
        self.assertTrue(any("重试" in warning for warning in update.warnings))

    def test_state_updater_raises_after_repair_exhausted(self) -> None:
        """
        修复重试仍失败时向上抛出，由会话层决定降级策略。
        """

        llm = FakeLLM(["坏输出一", "坏输出二"])
        updater = LegalCaseStateUpdater(llm)

        with self.assertRaises(ValueError):
            updater.update(previous_state=LegalCaseState(), public_messages=[], user_input="公司没签合同")
        self.assertEqual(2, llm.calls)

    def test_state_updater_raises_after_transport_retry_exhausted(self) -> None:
        """
        传输失败重试后再次失败时向上抛出原始异常。
        """

        llm = FakeLLM([RuntimeError("挂了一次"), RuntimeError("又挂了")])
        updater = LegalCaseStateUpdater(llm)

        with self.assertRaises(RuntimeError):
            updater.update(previous_state=LegalCaseState(), public_messages=[], user_input="公司没签合同")
        self.assertEqual(2, llm.calls)

    def test_case_analyzer_repairs_invalid_json_and_records_warning(self) -> None:
        """
        综合分析器同样具备修复重试能力，修复过程记录在 analysis.warnings 里。
        """

        llm = FakeLLM(["不是JSON", build_valid_case_analysis_response()])
        analyzer = LegalCaseAnalyzer(llm)

        analysis = analyzer.analyze(state=LegalCaseState(), rag=self.build_empty_rag())

        self.assertEqual("ask_followup", analysis.next_action.action)
        self.assertEqual(1, len(analysis.risks))
        self.assertEqual(2, llm.calls)
        self.assertTrue(any("修复" in warning for warning in analysis.warnings))


class LegalConsultationSessionTests(unittest.TestCase):
    """
    测试法律咨询业务会话的执行链路、公开历史和失败回滚。
    """

    def build_session(
        self,
        *,
        runner_error: Exception | None = None,
        web_search_error: Exception | None = None,
        runner_answer_text: str | None = None,
        llm_responses: list | None = None,
        planner_error: Exception | None = None,
        runner_fail_times: int = 0,
        llm_usage_per_call: dict | None = None,
    ) -> tuple[LegalConsultationSession, FakeRunner, FakeWebSearchSubtask]:
        """
        创建一套完整 fake session。

        Args:
            llm_responses: 可选自定义 LLM 响应脚本；为空时使用“状态更新 + 综合分析”各一次
                成功响应的默认脚本。条目可以是 Exception，用于模拟传输失败。
            planner_error: 可选 planner 异常，用于验证规划失败时的链路降级。
            runner_fail_times: 最终回答前 N 次调用失败，用于验证自动重试。
            llm_usage_per_call: FakeLLM 每次成功调用累计的 usage，用于轮级 metrics 测试。
        """

        # 一轮成功链路只有两次内部 LLM 调用：案件状态更新 + 综合分析。
        # FakeLLM 的预设响应数就是调用次数上限；如果链路退化成多次串行调用，这里会直接失败。
        llm = FakeLLM(
            llm_responses
            if llm_responses is not None
            else [build_valid_state_update_response(), build_valid_case_analysis_response()],
            usage_per_call=llm_usage_per_call,
        )
        plan = LegalQueryPlan(
            global_queries=["劳动争议"],
            issues=[
                LegalIssueQuery(
                    issue="未签书面劳动合同的责任",
                    facts=["公司未签书面劳动合同"],
                    preferred_legal_names=["劳动合同法"],
                    queries=["未签劳动合同 二倍工资", "书面劳动合同 用人单位 责任", "劳动合同 未订立"],
                    positive_terms=["未签", "劳动合同"],
                    negative_terms=[],
                )
            ],
            warnings=[],
        )
        runner = FakeRunner(error=runner_error, answer_text=runner_answer_text, fail_times=runner_fail_times)
        web_search = FakeWebSearchSubtask(error=web_search_error)
        session = LegalConsultationSession(
            state_updater=LegalCaseStateUpdater(llm),
            rag_subtask=LegalCaseRagSubtask(planner=FakePlanner(plan, error=planner_error), retriever=FakeRetriever()),
            case_analyzer=LegalCaseAnalyzer(llm),
            web_search_subtask=web_search,
            answer_runner=runner,
            system_prompt="你是法律助手。",
        )
        return session, runner, web_search

    def test_state_update_failure_degrades_and_chain_completes(self) -> None:
        """
        状态更新彻底失败时应降级为沿用既有案件状态，链路其余部分照常走完并正常提交。
        """

        session, runner, _ = self.build_session(
            llm_responses=["坏输出一", "坏输出二", build_valid_case_analysis_response()],
        )

        answer, events = session.ask_with_events("公司没签合同怎么办？")

        self.assertIn("阶段性答复", answer)
        event_types = [event.type for event in events]
        self.assertIn("legal_selfheal", event_types)
        selfheal = next(event for event in events if event.type == "legal_selfheal")
        self.assertEqual("degraded", selfheal.data["action"])
        self.assertEqual("案件状态更新", selfheal.data["stage"])
        # 链路走完：RAG、综合分析和最终回答事件齐全。
        self.assertIn("legal_case_rag_done", event_types)
        self.assertIn("legal_risk_analyzed", event_types)
        self.assertIn("message_done", event_types)
        self.assertEqual(1, len(runner.seen_calls))
        self.assertEqual(["system", "user", "assistant"], [m["role"] for m in session.history()])

    def test_rag_failure_degrades_and_chain_completes(self) -> None:
        """
        query 规划失败时应降级为空法条证据继续分析，不中断整轮咨询。
        """

        session, runner, _ = self.build_session(planner_error=RuntimeError("planner boom"))

        answer, events = session.ask_with_events("公司没签合同怎么办？")

        self.assertIn("阶段性答复", answer)
        event_types = [event.type for event in events]
        selfheal = next(event for event in events if event.type == "legal_selfheal")
        self.assertEqual("degraded", selfheal.data["action"])
        self.assertIn("RAG", selfheal.data["stage"])
        rag_done = next(event for event in events if event.type == "legal_case_rag_done")
        self.assertEqual(0, rag_done.data["evidence_count"])
        self.assertTrue(rag_done.data["warnings"])
        self.assertIn("message_done", event_types)
        self.assertEqual(["system", "user", "assistant"], [m["role"] for m in session.history()])

    def test_analyzer_failure_degrades_and_chain_completes(self) -> None:
        """
        综合分析彻底失败时应降级为空风险 + 默认追问动作，最终回答仍然生成。
        """

        session, runner, _ = self.build_session(
            llm_responses=[build_valid_state_update_response(), "坏输出一", "坏输出二"],
        )

        answer, events = session.ask_with_events("公司没签合同怎么办？")

        self.assertIn("阶段性答复", answer)
        selfheal = next(event for event in events if event.type == "legal_selfheal")
        self.assertEqual("degraded", selfheal.data["action"])
        self.assertEqual("案情综合分析", selfheal.data["stage"])
        risk_event = next(event for event in events if event.type == "legal_risk_analyzed")
        self.assertEqual(0, risk_event.data["risk_count"])
        action_event = next(event for event in events if event.type == "legal_next_action_decided")
        self.assertEqual("ask_followup", action_event.data["action"])
        self.assertEqual(["system", "user", "assistant"], [m["role"] for m in session.history()])

    def test_final_answer_retries_once_without_streaming(self) -> None:
        """
        最终回答首次失败时应自动重试一次；重试改为非流式，避免与已推送的部分增量叠加。
        """

        session, runner, _ = self.build_session(runner_fail_times=1)
        received: list[AgentEvent] = []

        answer, events = session.ask_with_events("公司没签合同怎么办？", on_event=received.append)

        self.assertIn("阶段性答复", answer)
        selfheal = next(event for event in events if event.type == "legal_selfheal")
        self.assertEqual("retried", selfheal.data["action"])
        self.assertEqual("最终回答生成", selfheal.data["stage"])
        self.assertEqual(2, len(runner.seen_calls))
        self.assertTrue(runner.seen_calls[0]["streaming"])
        self.assertFalse(runner.seen_calls[1]["streaming"])
        self.assertEqual(["system", "user", "assistant"], [m["role"] for m in session.history()])

    def test_turn_emits_metrics_event_with_stages_and_usage(self) -> None:
        """
        成功一轮应在末尾发出 legal_turn_metrics：分阶段耗时、状态和 LLM usage 增量。
        """

        session, _, _ = self.build_session(
            llm_usage_per_call={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )

        _, events = session.ask_with_events("公司没签合同怎么办？")

        metrics = next(event for event in events if event.type == "legal_turn_metrics")
        stages = {item["stage"]: item for item in metrics.data["stages"]}
        for stage_name in ("案件状态更新", "案情拆解 + 多 query RAG", "案情综合分析", "最终回答生成"):
            self.assertIn(stage_name, stages)
            self.assertEqual("ok", stages[stage_name]["status"])
        self.assertTrue(all(item["duration_ms"] >= 0 for item in metrics.data["stages"]))
        self.assertGreaterEqual(metrics.data["total_duration_ms"], 0)
        # 状态更新 + 综合分析走 FakeLLM，各累计一次调用；最终回答走 FakeRunner，不计入。
        usage = metrics.data["llm_usage"]
        self.assertEqual(2, usage["calls"])
        self.assertEqual(240, usage["total_tokens"])
        self.assertEqual(0, metrics.data["selfheal_count"])
        # metrics 是本轮收尾事件，晚于最终回答完成。
        event_types = [event.type for event in events]
        self.assertGreater(event_types.index("legal_turn_metrics"), event_types.index("message_done"))

    def test_metrics_marks_degraded_and_retried_stages(self) -> None:
        """
        降级和重试要反映在对应阶段的 status 上，并累计 selfheal_count。
        """

        session, _, _ = self.build_session(
            planner_error=RuntimeError("planner boom"),
            runner_fail_times=1,
        )

        _, events = session.ask_with_events("公司没签合同怎么办？")

        metrics = next(event for event in events if event.type == "legal_turn_metrics")
        stages = {item["stage"]: item for item in metrics.data["stages"]}
        self.assertEqual("degraded", stages["案情拆解 + 多 query RAG"]["status"])
        self.assertEqual("retried", stages["最终回答生成"]["status"])
        self.assertEqual(2, metrics.data["selfheal_count"])

    def test_pause_turn_emits_metrics_with_state_stage_only(self) -> None:
        """
        暂停补充轮只执行状态更新，metrics 应只包含该阶段。
        """

        pause_response = json.dumps(
            {
                "state": {"summary": "缺少关键信息", "follow_up_questions": ["何时入职？"]},
                "should_pause_for_supplement": True,
                "pause_reason": "缺少入职时间",
                "supplement_questions": ["何时入职？"],
            },
            ensure_ascii=False,
        )
        session, _, _ = self.build_session(llm_responses=[pause_response])

        _, events = session.ask_with_events("公司没签合同")

        metrics = next(event for event in events if event.type == "legal_turn_metrics")
        self.assertEqual(["案件状态更新"], [item["stage"] for item in metrics.data["stages"]])
        self.assertEqual(1, metrics.data["llm_usage"]["calls"])

    def test_recalled_memories_enter_prompts_and_emit_event(self) -> None:
        """
        传入历史记忆时应发 legal_memory_recalled 事件，并注入状态更新 prompt 和最终回答运行时输入；
        公开 history 仍只保存用户原始输入，不携带记忆文本。
        """

        session, runner, _ = self.build_session()
        memories = [
            {
                "title": "此前的劳动合同咨询",
                "summary": "用户此前咨询过未签书面劳动合同问题。",
                "key_facts": ["公司未签书面劳动合同"],
                "user_goals": ["要求二倍工资"],
                "legal_concepts": ["可能涉及二倍工资"],
                "updated_at": "2026-07-01T00:00:00+00:00",
            }
        ]

        _, events = session.ask_with_events("公司现在又拖欠工资怎么办？", recalled_memories=memories)

        event_types = [event.type for event in events]
        self.assertIn("legal_memory_recalled", event_types)
        # 记忆唤起发生在链路最前面：先告诉用户“结合了哪些历史咨询”，再开始状态更新。
        self.assertLess(event_types.index("legal_memory_recalled"), event_types.index("case_state_updated"))
        memory_event = next(event for event in events if event.type == "legal_memory_recalled")
        self.assertEqual(1, memory_event.data["count"])
        self.assertEqual("此前的劳动合同咨询", memory_event.data["memories"][0]["title"])
        # 事件面向前端展示，只应携带标题、摘要和更新时间三个白名单字段。
        self.assertEqual({"title", "summary", "updated_at"}, set(memory_event.data["memories"][0].keys()))

        updater_prompt = session.state_updater.llm.seen_messages[0][-1]["content"]
        self.assertIn("【历史咨询记忆", updater_prompt)
        self.assertIn("此前的劳动合同咨询", updater_prompt)
        self.assertIn("不得把历史记忆内容写入", updater_prompt)

        runtime_input = runner.seen_calls[0]["messages"][-1]["content"]
        self.assertIn("【历史咨询记忆】", runtime_input)
        self.assertIn("此前的劳动合同咨询", runtime_input)

        history = session.history()
        self.assertEqual("公司现在又拖欠工资怎么办？", history[-2]["content"])

    def test_without_recalled_memories_prompts_stay_clean(self) -> None:
        """
        不传历史记忆时不发记忆事件，各 prompt 也不出现记忆区块，保持旧链路完全不变。
        """

        session, runner, _ = self.build_session()

        _, events = session.ask_with_events("公司没签合同怎么办？")

        self.assertNotIn("legal_memory_recalled", [event.type for event in events])
        updater_prompt = session.state_updater.llm.seen_messages[0][-1]["content"]
        self.assertNotIn("历史咨询记忆", updater_prompt)
        runtime_input = runner.seen_calls[0]["messages"][-1]["content"]
        self.assertNotIn("历史咨询记忆", runtime_input)

    def test_blank_recalled_memories_are_ignored(self) -> None:
        """
        空白或非法记忆条目应被静默过滤：记忆是辅助信号，脏数据不能打断咨询链路。
        """

        session, _, _ = self.build_session()

        _, events = session.ask_with_events(
            "公司没签合同怎么办？",
            recalled_memories=[{"title": " ", "summary": ""}, "junk", 42],
        )

        self.assertNotIn("legal_memory_recalled", [event.type for event in events])

    def test_full_turn_keeps_internal_subtasks_out_of_public_history(self) -> None:
        """
        成功执行一轮后，公开 history 只应保存 system/user/assistant，不应包含内部 JSON 子任务提示词。
        """

        session, runner, web_search = self.build_session()

        answer, events = session.ask_with_events("公司没签合同怎么办？")

        self.assertIn("阶段性答复", answer)
        event_types = [event.type for event in events]
        self.assertIn("case_state_updated", event_types)
        self.assertIn("legal_missing_details_suggested", event_types)
        self.assertIn("legal_case_rag_done", event_types)
        self.assertIn("legal_next_action_decided", event_types)
        self.assertIn("legal_web_search_started", event_types)
        self.assertIn("legal_web_search_done", event_types)
        self.assertIn("legal_reference_materials", event_types)
        self.assertLess(event_types.index("case_state_updated"), event_types.index("legal_missing_details_suggested"))
        self.assertLess(event_types.index("legal_missing_details_suggested"), event_types.index("legal_case_rag_done"))
        # 公网检索应在本地 RAG 开始前就启动，让第三方搜索等待时间和整个本地分析段重叠。
        self.assertLess(event_types.index("legal_web_search_started"), event_types.index("legal_case_rag_done"))
        self.assertLess(event_types.index("legal_web_search_started"), event_types.index("legal_risk_analyzed"))
        self.assertLess(event_types.index("legal_reference_materials"), event_types.index("message_done"))
        missing_event = next(event for event in events if event.type == "legal_missing_details_suggested")
        self.assertEqual(["是否有工资流水？"], missing_event.data["questions"])
        self.assertEqual(["工资流水"], missing_event.data["evidence_gaps"])
        materials_event = next(event for event in events if event.type == "legal_reference_materials")
        self.assertGreaterEqual(len(materials_event.data["laws"]), 1)
        self.assertGreaterEqual(len(materials_event.data["web"]), 1)
        self.assertEqual("《劳动合同法》第八十二条", materials_event.data["laws"][0]["title"])
        self.assertIn("用人单位", materials_event.data["laws"][0]["detail"])
        self.assertEqual("本地法条库", materials_event.data["laws"][0]["source"])
        self.assertEqual("未签劳动合同二倍工资案例", materials_event.data["web"][0]["title"])
        self.assertEqual("https://example.test/case", materials_event.data["web"][0]["url"])
        self.assertIn("公开案例摘要", materials_event.data["web"][0]["detail"])
        history = session.history()
        self.assertEqual(["system", "user", "assistant"], [message["role"] for message in history])
        self.assertEqual("公司没签合同怎么办？", history[1]["content"])
        serialized_history = json.dumps(history, ensure_ascii=False)
        self.assertNotIn("案件状态更新器", serialized_history)
        self.assertNotIn("只输出合法 JSON", serialized_history)
        self.assertEqual(1, len(web_search.seen_calls))
        # 公网检索在 RAG 和综合分析之前启动，此时这些结果尚未产出。
        self.assertIsNone(web_search.seen_calls[0]["rag"])
        self.assertIsNone(web_search.seen_calls[0]["next_action"])
        runtime_input = runner.seen_calls[0]["messages"][-1]["content"]
        self.assertIn("最新案件状态", runtime_input)
        self.assertIn("已检索法条证据", runtime_input)
        self.assertIn("公网案例与司法实践检索", runtime_input)
        self.assertIn("未签劳动合同二倍工资案例", runtime_input)
        self.assertIn("https://example.test/case", runtime_input)
        self.assertIn("右侧参考资料栏", runtime_input)
        self.assertIn("最终回答只保留最重要要点", runtime_input)
        self.assertIn("不要逐条罗列全部法条", runtime_input)
        self.assertIn("不要长篇列相似案例", runtime_input)
        self.assertNotIn("公司未签劳动合同 判决 典型案例", serialized_history)
        self.assertNotIn("https://example.test/case", serialized_history)
        self.assertNotIn("公网案例与司法实践检索", serialized_history)

    def test_final_answer_strips_reference_material_sections_from_chat_history(self) -> None:
        """
        最终聊天气泡不应继续展示长篇法条、案例和来源链接，这些内容应只留在资料栏事件中。
        """

        verbose_answer = """
## 先说结论
你可以先主张未签劳动合同的责任。

## 法律依据
《劳动合同法》第八十二条规定：用人单位未签书面劳动合同，应当支付二倍工资。

## 公网案例和实务参考
1. 搜狐转载案例，来源：https://example.test/case
2. 某咨询平台文章，来源：https://example.test/article

## 你现在该做什么
1. 先准备工资流水。
2. 再确认入职时间。

以下内容仅作一般信息参考，不构成正式法律意见。
""".strip()
        session, _, _ = self.build_session(runner_answer_text=verbose_answer)

        answer, events = session.ask_with_events("公司没签合同怎么办？")

        self.assertIn("先说结论", answer)
        self.assertIn("你现在该做什么", answer)
        self.assertNotIn("法律依据", answer)
        self.assertNotIn("公网案例和实务参考", answer)
        self.assertNotIn("https://example.test/case", answer)
        message_done = next(event for event in events if event.type == "message_done")
        self.assertEqual(answer, message_done.data["text"])
        self.assertEqual(answer, session.history()[-1]["content"])

    def test_streaming_answer_deltas_are_sanitized_before_push(self) -> None:
        """
        流式增量必须先过行级清洗：资料章节和 URL 行不能在打字机阶段闪现给用户。
        """

        verbose_answer = """
## 先说结论
你可以先主张未签劳动合同的责任。

## 法律依据
《劳动合同法》第八十二条规定：用人单位未签书面劳动合同，应当支付二倍工资。
来源：https://example.test/law

## 你现在该做什么
1. 先准备工资流水。

以下内容仅作一般信息参考，不构成正式法律意见。
""".strip()
        session, runner, _ = self.build_session(runner_answer_text=verbose_answer)
        received: list[AgentEvent] = []

        answer, events = session.ask_with_events("公司没签合同怎么办？", on_event=received.append)

        # 传入 on_event 时最终回答应走流式路径。
        self.assertTrue(runner.seen_calls[0]["streaming"])
        delta_events = [event for event in received if event.type == "answer_delta"]
        self.assertGreaterEqual(len(delta_events), 1)
        streamed_text = "".join(str(event.data["delta"]) for event in delta_events)
        self.assertIn("先说结论", streamed_text)
        self.assertIn("你现在该做什么", streamed_text)
        self.assertNotIn("法律依据", streamed_text)
        self.assertNotIn("https://example.test/law", streamed_text)
        # answer_delta 只推实时回调，不写入返回的 events 列表，避免事件回放被增量撑爆。
        self.assertNotIn("answer_delta", [event.type for event in events])
        self.assertNotIn("法律依据", answer)

    def test_no_streaming_without_on_event_callback(self) -> None:
        """
        没有实时回调（如 CLI 静默调用）时不应走流式路径，避免无谓的增量清洗开销。
        """

        session, runner, _ = self.build_session()

        session.ask_with_events("公司没签合同怎么办？")

        self.assertFalse(runner.seen_calls[0]["streaming"])

    def test_final_answer_strips_plain_reference_only_sections(self) -> None:
        """
        纯资料型回答也不能回退成原文，普通标题和无空格 Markdown 标题同样要识别。
        """

        reference_only_answer = """
法律依据：
《消费者权益保护法》第四十四条规定平台责任。

##公网案例和实务参考
1. 搜狐转载案例，来源：https://example.test/case
2. 咨询平台文章，来源：https://example.test/article
""".strip()
        session, _, _ = self.build_session(runner_answer_text=reference_only_answer)

        answer, events = session.ask_with_events("平台卖假货怎么办？")

        self.assertIn("右侧参考资料栏", answer)
        self.assertIn("不构成正式法律意见", answer)
        self.assertNotIn("法律依据", answer)
        self.assertNotIn("公网案例", answer)
        self.assertNotIn("https://example.test/case", answer)
        message_done = next(event for event in events if event.type == "message_done")
        self.assertEqual(answer, message_done.data["text"])

    def test_lawyer_style_risk_lines_survive_sanitizing(self) -> None:
        """
        律师风格答复的正文里会出现“司法实践”“案例”等词（如“民事：司法实践中……”）。
        这些是风险分析正文而不是资料章节标题，清洗器不能因为关键词命中就整段误删。
        """

        lawyer_answer = """
## 结论
车辆大概率会被认定为公司财产，由破产管理人接管处置。

## 法律风险
刑事：拒不移交或擅自转移车辆，可能涉嫌拒不执行判决、裁定罪或职务侵占罪，处三年以下有期徒刑或者拘役。
民事：司法实践中管理人可要求返还车辆并赔偿使用贬值损失。

## 现在该做什么
1. 主动联系管理人办理车辆和证照移交。

以下内容仅作一般信息参考，不构成正式法律意见。
""".strip()
        session, _, _ = self.build_session(runner_answer_text=lawyer_answer)

        answer, _ = session.ask_with_events("公司破产了，公司的车还在我手里怎么办？")

        self.assertIn("职务侵占罪", answer)
        self.assertIn("三年以下有期徒刑", answer)
        self.assertIn("司法实践中管理人可要求返还车辆", answer)
        self.assertIn("现在该做什么", answer)

    def test_new_core_headings_recover_after_reference_section(self) -> None:
        """
        “结论”“法律风险”等新核心标题必须能结束资料章节的跳过状态，
        否则模型偶发输出“法律依据”章节后，后面的风险分析会被整段吞掉。
        """

        verbose_answer = """
## 结论
车辆大概率会作为破产财产由管理人处置。

## 法律依据
《企业破产法》第二十五条规定管理人接管债务人财产。

## 法律风险
刑事：拒不移交可能涉嫌拒不执行判决、裁定罪。

以下内容仅作一般信息参考，不构成正式法律意见。
""".strip()
        session, _, _ = self.build_session(runner_answer_text=verbose_answer)

        answer, _ = session.ask_with_events("公司破产了，公司的车还在我手里怎么办？")

        self.assertIn("结论", answer)
        self.assertIn("法律风险", answer)
        self.assertIn("拒不移交可能涉嫌", answer)
        self.assertNotIn("第二十五条规定管理人接管", answer)

    def test_reference_materials_skip_invalid_law_evidences_before_cap(self) -> None:
        """
        法条资料条数上限应作用在有效资料上，不能被前面的空标题证据占满。
        """

        invalid_evidences = [
            LegalArticleEvidence(
                citation="",
                legal_name="",
                article_no="",
                text="空证据",
                issue="",
                source_query="",
                retrieval_type="semantic",
            )
            for _ in range(8)
        ]
        valid_evidence = LegalArticleEvidence(
            citation="《消费者权益保护法》第四十四条",
            legal_name="消费者权益保护法",
            article_no="第四十四条",
            text="网络交易平台提供者责任。",
            issue="平台责任",
            source_query="平台 假货",
            retrieval_type="semantic",
        )
        rag = LegalCaseRagResult(
            query_plan=LegalQueryPlan(global_queries=[], issues=[], warnings=[]),
            issue_results=[],
            evidences=[*invalid_evidences, valid_evidence],
        )

        materials = build_reference_materials(rag=rag, web_research=None, max_laws=8)

        self.assertEqual(["《消费者权益保护法》第四十四条"], [item.title for item in materials.laws])

    def test_pauses_for_required_supplement_before_rag_and_runner(self) -> None:
        """
        阻塞性信息缺失时，应提交状态并提前返回补充提示，不继续 RAG 和最终回答。
        """

        llm = FakeLLM(
            [
                json.dumps(
                    {
                        "state": {
                            "summary": "用户咨询交通事故赔偿。",
                            "parties": ["伤者", "对方司机"],
                            "timeline": [],
                            "confirmed_facts": ["发生交通事故"],
                            "disputed_facts": [],
                            "adverse_facts": [],
                            "contradictions": [],
                            "evidence_gaps": ["事故认定书", "医疗票据"],
                            "user_goals": ["了解赔偿范围"],
                            "legal_concepts": ["可能涉及侵权责任"],
                            "follow_up_questions": ["事故责任认定如何？"],
                        },
                        "newly_added_facts": ["发生交通事故"],
                        "changed_facts": [],
                        "warnings": [],
                        "should_pause_for_supplement": True,
                        "pause_reason": "缺少责任比例和损失信息，继续分析会不可靠。",
                        "supplement_questions": ["事故责任认定如何？"],
                        "supplement_evidence_gaps": ["事故认定书", "医疗票据"],
                    },
                    ensure_ascii=False,
                )
            ]
        )
        plan = LegalQueryPlan(global_queries=["交通事故赔偿"], issues=[], warnings=[])
        planner = FakePlanner(plan)
        retriever = FakeRetriever()
        runner = FakeRunner()
        web_search = FakeWebSearchSubtask()
        session = LegalConsultationSession(
            state_updater=LegalCaseStateUpdater(llm),
            rag_subtask=LegalCaseRagSubtask(planner=planner, retriever=retriever),
            case_analyzer=LegalCaseAnalyzer(llm),
            web_search_subtask=web_search,
            answer_runner=runner,
            system_prompt="你是法律助手。",
        )

        answer, events = session.ask_with_events("交通事故怎么赔？")

        event_types = [event.type for event in events]
        self.assertIn("请先补充以下信息", answer)
        self.assertIn("legal_supplement_required", event_types)
        self.assertNotIn("legal_case_rag_done", event_types)
        self.assertNotIn("legal_risk_analyzed", event_types)
        self.assertNotIn("legal_next_action_decided", event_types)
        self.assertEqual([], planner.seen_case_texts)
        self.assertEqual([], retriever.semantic_calls)
        self.assertEqual([], runner.seen_calls)
        self.assertEqual([], web_search.seen_calls)
        self.assertNotIn("legal_web_search_started", event_types)
        self.assertNotIn("legal_web_search_done", event_types)
        self.assertEqual(1, session.case_state.version)
        history = session.history()
        self.assertEqual(["system", "user", "assistant"], [message["role"] for message in history])
        self.assertIn("事故责任认定如何", history[2]["content"])

    def test_allow_pause_false_ignores_pause_and_completes_full_chain(self) -> None:
        """
        用户明确表示无法补充时（allow_pause=False），暂停判定应被忽略：
        发 legal_supplement_skipped 事件后继续走完整链路，不能把流程卡死。
        """

        pause_state = json.loads(build_valid_state_update_response())
        pause_state["should_pause_for_supplement"] = True
        pause_state["pause_reason"] = "缺少入职时间和工资信息。"
        pause_state["supplement_questions"] = ["入职时间是什么时候？"]
        pause_state["supplement_evidence_gaps"] = ["工资流水"]
        session, runner, web_search = self.build_session(
            llm_responses=[json.dumps(pause_state, ensure_ascii=False), build_valid_case_analysis_response()],
        )

        answer, events = session.ask_with_events("我暂时无法补充更多信息，请继续分析。", allow_pause=False)

        event_types = [event.type for event in events]
        self.assertIn("阶段性答复", answer)
        self.assertIn("legal_supplement_skipped", event_types)
        self.assertNotIn("legal_supplement_required", event_types)
        # 完整链路照常执行：本地 RAG、综合分析、公网检索和最终回答全部完成。
        self.assertIn("legal_case_rag_done", event_types)
        self.assertIn("legal_risk_analyzed", event_types)
        self.assertIn("message_done", event_types)
        self.assertEqual(1, len(runner.seen_calls))
        self.assertEqual(1, len(web_search.seen_calls))
        self.assertEqual(["system", "user", "assistant"], [m["role"] for m in session.history()])

    def test_web_search_failure_becomes_warning_and_final_answer_still_runs(self) -> None:
        """
        确定性公网检索失败时，应记录 warning 并继续最终回答。
        """

        session, runner, web_search = self.build_session(web_search_error=RuntimeError("web boom"))

        answer, events = session.ask_with_events("公司没签合同怎么办？")

        self.assertIn("阶段性答复", answer)
        self.assertEqual(1, len(web_search.seen_calls))
        self.assertEqual(1, len(runner.seen_calls))
        self.assertEqual(1, session.case_state.version)
        done_event = next(event for event in events if event.type == "legal_web_search_done")
        self.assertGreaterEqual(done_event.data["warning_count"], 1)
        materials_event = next(event for event in events if event.type == "legal_reference_materials")
        self.assertGreaterEqual(len(materials_event.data["laws"]), 1)
        self.assertEqual([], materials_event.data["web"])
        self.assertTrue(any("部分资料检索未成功" in warning for warning in materials_event.data["warnings"]))
        runtime_input = runner.seen_calls[0]["messages"][-1]["content"]
        self.assertIn("公网案例与司法实践检索", runtime_input)
        self.assertIn("web boom", runtime_input)

    def test_system_prompt_keeps_reference_materials_out_of_final_answer(self) -> None:
        """
        法律咨询系统提示应约束最终回答不要自行展开资料或编造缺失资料。
        """

        prompt = DEFAULT_LEGAL_CONSULTATION_SYSTEM_PROMPT
        for keyword in ["相似案例", "司法实践", "赔偿", "执行", "最终回答阶段不要再尝试自行检索", "不要编造"]:
            self.assertIn(keyword, prompt)
        self.assertNotIn("应调用 web_search", prompt)

    def test_rolls_back_state_and_history_when_final_runner_fails(self) -> None:
        """
        最终回答失败时，不能提交半成品案件状态，也不能把当前 user message 写入公开历史。
        """

        session, _, _ = self.build_session(runner_error=RuntimeError("runner boom"))

        with self.assertRaises(RuntimeError):
            session.ask_with_events("公司没签合同怎么办？")

        self.assertEqual(0, session.case_state.version)
        self.assertEqual(["system"], [message["role"] for message in session.history()])


if __name__ == "__main__":
    unittest.main()
