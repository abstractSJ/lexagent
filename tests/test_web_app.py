"""
Web UI 接口的本地单元测试。

这些测试只验证 FastAPI 包装层、NDJSON 事件格式和错误处理；通过 fake session 注入避免调用
真实 LLM、本地 BGE-M3、Chroma 或外部网络。
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from agent_system.agent.events import AgentEvent
from agent_system.storage import SessionStore
from web_app.server import create_app, sanitize_event_for_web


class FakeLegalSession:
    """
    Web 测试使用的假法律咨询会话。

    Args:
        error: 可选异常；传入后 ask_with_events 会抛出该异常，用于验证后端错误流。
    """

    def __init__(
        self,
        *,
        error: Exception | None = None,
        preload_error: Exception | None = None,
        emit_error_event_before_raise: bool = False,
        block_until_released: bool = False,
        block_preload_until_released: bool = False,
        pause: bool = False,
    ) -> None:
        self.error = error
        self.preload_error = preload_error
        self.emit_error_event_before_raise = emit_error_event_before_raise
        self.block_until_released = block_until_released
        self.block_preload_until_released = block_preload_until_released
        self.pause = pause
        self.preload_calls = 0
        self.seen_inputs: list[str] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.preload_started = threading.Event()
        self.release_preload = threading.Event()

    def preload_resources(self) -> None:
        """
        记录预热调用，不加载任何真实资源。
        """

        self.preload_calls += 1
        self.preload_started.set()
        if self.block_preload_until_released:
            self.release_preload.wait(timeout=5)
        if self.preload_error is not None:
            raise self.preload_error

    def ask_with_events(self, text, *, on_event=None):
        """
        模拟一轮法律咨询，并同步推送两个过程事件。
        """

        self.seen_inputs.append(text)
        self.started.set()
        if self.block_until_released:
            self.release.wait(timeout=5)
        if self.error is not None:
            if self.emit_error_event_before_raise and on_event is not None:
                on_event(AgentEvent(type="error", data={"error": str(self.error)}))
            raise self.error
        if self.pause:
            events = [
                AgentEvent(type="case_state_updated", data={"version": 1, "summary": "缺少关键信息"}),
                AgentEvent(
                    type="legal_supplement_required",
                    data={
                        "reason": "缺少入职时间和工资信息。",
                        "message": "请先补充入职时间和工资信息。",
                        "questions": ["入职时间是什么时候？", "月工资是多少？"],
                        "evidence_gaps": ["工资流水"],
                        "query": "内部暂停 query 不应展示",
                        "retrieval_type": "semantic",
                        "state_version": 1,
                    },
                ),
            ]
            if on_event is not None:
                for event in events:
                    on_event(event)
            return "请先补充入职时间和工资信息。", events
        if on_event is not None:
            on_event(AgentEvent(type="legal_step", data={"name": "案件状态更新", "status": "start"}))
            on_event(
                AgentEvent(
                    type="legal_missing_details_suggested",
                    data={
                        "questions": ["入职时间是什么时候？", "月工资是多少？"],
                        "evidence_gaps": ["工资流水"],
                        "message": "可以先准备这些关键信息；后台会继续检索相关法条并生成阶段性答复。",
                        "query": "内部检索 query 不应展示",
                        "retrieval_type": "semantic",
                    },
                )
            )
            on_event(
                AgentEvent(
                    type="legal_rag_query_started",
                    data={
                        "retrieval_type": "semantic",
                        "issue": "未签书面劳动合同的责任",
                        "query": text,
                    },
                )
            )
            on_event(
                AgentEvent(
                    type="legal_rag_query_started",
                    data={
                        "retrieval_type": "keyword",
                        "issue": "未签书面劳动合同的责任",
                        "query": "未签 劳动合同 二倍工资",
                    },
                )
            )
            on_event(
                AgentEvent(
                    type="legal_risk_analyzed",
                    data={
                        "risk_count": 1,
                        "risks": [
                            {
                                "fact": "内部风险事实不应展示",
                                "reason": "内部风险原因不应展示",
                                "suggestion": "内部风险建议不应展示",
                            }
                        ],
                    },
                )
            )
            on_event(
                AgentEvent(
                    type="legal_web_search_started",
                    data={
                        "status": "searching",
                        "query": "内部 web search query 不应展示",
                        "arguments": {"query": "内部 web search query 不应展示", "count": 5},
                    },
                )
            )
            on_event(
                AgentEvent(
                    type="legal_web_search_done",
                    data={
                        "status": "done",
                        "query_count": 2,
                        "result_count": 4,
                        "warning_count": 0,
                        "results": [
                            {
                                "title": "内部案例标题不应展示",
                                "url": "https://example.test/internal",
                                "snippet": "内部摘要不应展示",
                                "summary": "内部正文不应展示",
                            }
                        ],
                    },
                )
            )
            on_event(
                AgentEvent(
                    type="legal_reference_materials",
                    data={
                        "laws": [
                            {
                                "id": "law-0",
                                "material_type": "law",
                                "title": "《劳动合同法》第八十二条",
                                "subtitle": "未签书面劳动合同责任",
                                "detail": "用人单位未签书面劳动合同的二倍工资责任。",
                                "source": "本地法条库",
                                "issue": "未签书面劳动合同的责任",
                                "raw_prompt": "内部 prompt 不应展示",
                            }
                        ],
                        "web": [
                            {
                                "id": "web-0",
                                "material_type": "web",
                                "title": "未签劳动合同二倍工资案例",
                                "subtitle": "示例案例库",
                                "detail": "法院关注入职时间、工资标准和仲裁时效。",
                                "url": "https://example.test/case",
                                "source": "示例案例库",
                                "issue": "behavior_cases",
                                "arguments": {"query": "内部 query 不应展示"},
                            }
                        ],
                        "warnings": ["资料 warning"],
                    },
                )
            )
            on_event(AgentEvent(type="message_done", data={"text": "内部最终回答事件不展示到右侧"}))
        return "这是模拟法律咨询答复。", []


class WebAppTests(unittest.TestCase):
    """
    测试 Web 层 HTTP 接口。
    """

    def setUp(self) -> None:
        """
        测试期间关闭启动自动预热，避免 TestClient 生命周期触发额外 fake 调用。
        """

        self.old_preload = os.environ.get("LEGAL_RAG_PRELOAD")
        os.environ["LEGAL_RAG_PRELOAD"] = "0"

    def tearDown(self) -> None:
        """
        恢复环境变量，避免影响其他测试或本地手动运行。
        """

        if self.old_preload is None:
            os.environ.pop("LEGAL_RAG_PRELOAD", None)
        else:
            os.environ["LEGAL_RAG_PRELOAD"] = self.old_preload

    def build_client(self, session: FakeLegalSession | None = None) -> TestClient:
        """
        创建注入 fake session 的测试客户端。
        """

        return TestClient(create_app(session=session or FakeLegalSession()))

    def parse_ndjson(self, text: str) -> list[dict[str, object]]:
        """
        将 NDJSON 响应文本解析为对象列表。
        """

        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def wait_until(self, condition, *, timeout: float = 2.0, interval: float = 0.01) -> bool:
        """
        在测试中等待后台线程完成某个可观察状态。

        Args:
            condition: 返回布尔值的状态检查函数。
            timeout: 最长等待秒数。
            interval: 两次检查之间的等待秒数。

        Returns:
            bool: 条件在超时前满足则返回 True，否则返回 False。
        """

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return True
            time.sleep(interval)
        return condition()

    def test_health_returns_status(self) -> None:
        """
        健康检查应返回服务可用状态。
        """

        client = self.build_client()

        response = client.get("/api/health")

        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertEqual("legal-agent-web", data["service"])
        self.assertFalse(data["startup_preload_enabled"])

    def test_preload_calls_session(self) -> None:
        """
        手动预热接口应调用 session.preload_resources()。
        """

        session = FakeLegalSession()
        client = self.build_client(session)

        response = client.post("/api/preload")

        self.assertEqual(200, response.status_code)
        self.assertTrue(response.json()["ok"])
        self.assertEqual(1, session.preload_calls)

    def test_chat_returns_ndjson_events_final_and_done(self) -> None:
        """
        聊天接口应返回过程事件、最终回答和结束事件。
        """

        session = FakeLegalSession()
        client = self.build_client(session)

        response = client.post("/api/chat", json={"message": "公司没签劳动合同怎么办？"})

        self.assertEqual(200, response.status_code)
        self.assertIn("application/x-ndjson", response.headers["content-type"])
        items = self.parse_ndjson(response.text)
        self.assertEqual(["公司没签劳动合同怎么办？"], session.seen_inputs)
        self.assertIn("event", [item["type"] for item in items])
        self.assertIn("final", [item["type"] for item in items])
        self.assertEqual("done", items[-1]["type"])
        event_types = [item.get("event_type") for item in items if item.get("type") == "event"]
        self.assertIn("legal_step", event_types)
        self.assertIn("legal_missing_details_suggested", event_types)
        self.assertEqual(1, event_types.count("legal_rag_query_started"))
        self.assertNotIn("message_done", event_types)
        missing_items = [item for item in items if item.get("event_type") == "legal_missing_details_suggested"]
        self.assertEqual(["入职时间是什么时候？", "月工资是多少？"], missing_items[0]["data"]["questions"])
        self.assertEqual(["工资流水"], missing_items[0]["data"]["evidence_gaps"])
        serialized_items = json.dumps(items, ensure_ascii=False)
        self.assertIn("legal_risk_analyzed", event_types)
        self.assertIn("legal_web_search_started", event_types)
        self.assertIn("legal_web_search_done", event_types)
        self.assertIn("legal_reference_materials", event_types)
        materials_item = next(item for item in items if item.get("event_type") == "legal_reference_materials")
        self.assertEqual("《劳动合同法》第八十二条", materials_item["data"]["laws"][0]["title"])
        self.assertEqual("未签劳动合同二倍工资案例", materials_item["data"]["web"][0]["title"])
        self.assertEqual("https://example.test/case", materials_item["data"]["web"][0]["url"])
        self.assertNotIn("未签 劳动合同 二倍工资", serialized_items)
        self.assertNotIn("内部检索 query 不应展示", serialized_items)
        self.assertNotIn("内部 web search query 不应展示", serialized_items)
        self.assertNotIn("内部风险事实不应展示", serialized_items)
        self.assertNotIn("内部风险原因不应展示", serialized_items)
        self.assertNotIn("内部风险建议不应展示", serialized_items)
        self.assertNotIn("内部案例标题不应展示", serialized_items)
        self.assertNotIn("https://example.test/internal", serialized_items)
        self.assertNotIn("内部摘要不应展示", serialized_items)
        self.assertNotIn("内部正文不应展示", serialized_items)
        self.assertNotIn("内部 prompt 不应展示", serialized_items)
        self.assertNotIn("内部 query 不应展示", serialized_items)
        self.assertNotIn("raw_prompt", serialized_items)
        self.assertNotIn("retrieval_type", serialized_items)
        self.assertNotIn("arguments", serialized_items)
        self.assertNotIn("results", serialized_items)
        risk_done = next(item for item in items if item.get("event_type") == "legal_risk_analyzed")
        self.assertEqual({"status": "done", "risk_count": 1}, risk_done["data"])
        web_done = next(item for item in items if item.get("event_type") == "legal_web_search_done")
        self.assertEqual({"status": "done", "result_count": 4, "warning_count": 0}, web_done["data"])
        final_items = [item for item in items if item.get("type") == "final"]
        self.assertEqual("这是模拟法律咨询答复。", final_items[0]["answer"])

    def test_chat_pause_returns_pause_and_done_without_final(self) -> None:
        """
        需要用户补充时，Web 流应返回 pause 和 done，不直接返回 final。
        """

        session = FakeLegalSession(pause=True)
        client = self.build_client(session)

        response = client.post("/api/chat", json={"message": "公司没签劳动合同怎么办？"})

        self.assertEqual(200, response.status_code)
        items = self.parse_ndjson(response.text)
        item_types = [item.get("type") for item in items]
        self.assertIn("pause", item_types)
        self.assertNotIn("final", item_types)
        self.assertEqual("done", items[-1]["type"])
        pause_item = next(item for item in items if item.get("type") == "pause")
        self.assertEqual("缺少入职时间和工资信息。", pause_item["reason"])
        self.assertEqual(["入职时间是什么时候？", "月工资是多少？"], pause_item["questions"])
        self.assertEqual(["工资流水"], pause_item["evidence_gaps"])
        serialized_items = json.dumps(items, ensure_ascii=False)
        self.assertNotIn("内部暂停 query 不应展示", serialized_items)
        self.assertNotIn("retrieval_type", serialized_items)

    def test_chat_accepts_supplement_payload(self) -> None:
        """
        补充表单请求应合成为一段用户输入传给法律咨询会话。
        """

        session = FakeLegalSession()
        client = self.build_client(session)

        response = client.post(
            "/api/chat",
            json={
                "message": "我补充以下关键信息：",
                "supplement_answers": {"入职时间是什么时候？": "2023年5月入职"},
                "selected_questions": ["入职时间是什么时候？"],
                "selected_evidence_gaps": ["工资流水"],
                "free_text": "公司上周口头辞退，没有书面通知。",
            },
        )

        self.assertEqual(200, response.status_code)
        seen_input = session.seen_inputs[0]
        self.assertIn("用户补充的逐项回答", seen_input)
        self.assertIn("入职时间是什么时候？：2023年5月入职", seen_input)
        self.assertIn("工资流水", seen_input)
        self.assertIn("公司上周口头辞退", seen_input)

    def test_chat_rejects_empty_message(self) -> None:
        """
        空消息应返回 400，避免进入法律咨询链路。
        """

        client = self.build_client()

        response = client.post("/api/chat", json={"message": "   "})

        self.assertEqual(400, response.status_code)
        self.assertIn("message 或补充内容不能为空", response.json()["detail"])

    def test_chat_exception_returns_error_and_done_events(self) -> None:
        """
        session 抛异常时，NDJSON 流中应包含 error 和 done，不应把异常泄漏成断开的响应。
        """

        client = self.build_client(FakeLegalSession(error=RuntimeError("runner boom")))

        response = client.post("/api/chat", json={"message": "公司没签劳动合同怎么办？"})

        self.assertEqual(200, response.status_code)
        items = self.parse_ndjson(response.text)
        self.assertEqual("error", items[0]["type"])
        self.assertIn("runner boom", items[0]["message"])
        self.assertEqual("done", items[-1]["type"])

    def test_reference_materials_event_is_sanitized_for_web(self) -> None:
        """
        资料事件只应透出安全展示字段，并截断过长详情。
        """

        safe_event = sanitize_event_for_web(
            "legal_reference_materials",
            {
                "laws": [
                    {
                        "id": "law-0",
                        "material_type": "law",
                        "title": "《劳动合同法》第八十二条",
                        "subtitle": "未签书面劳动合同责任",
                        "detail": "法条正文" * 600,
                        "url": "",
                        "source": "本地法条库",
                        "issue": "未签书面劳动合同的责任",
                        "raw_prompt": "内部 prompt 不应展示",
                    },
                    {"id": "empty-title", "title": "", "detail": "空标题应丢弃"},
                ],
                "web": [
                    {
                        "id": "web-0",
                        "material_type": "web",
                        "title": "未签劳动合同二倍工资案例",
                        "detail": "案例摘要",
                        "url": "https://example.test/case",
                        "source": "示例案例库",
                        "arguments": {"query": "内部 query 不应展示"},
                    }
                ],
                "warnings": ["资料 warning"],
            },
        )

        self.assertIsNotNone(safe_event)
        event_type, data = safe_event
        self.assertEqual("legal_reference_materials", event_type)
        self.assertEqual(1, len(data["laws"]))
        self.assertEqual(1, len(data["web"]))
        self.assertEqual("《劳动合同法》第八十二条", data["laws"][0]["title"])
        self.assertLessEqual(len(data["laws"][0]["detail"]), 1001)
        self.assertNotIn("raw_prompt", data["laws"][0])
        self.assertNotIn("arguments", data["web"][0])
        self.assertEqual(["资料 warning"], data["warnings"])

    def test_chat_deduplicates_internal_error_event(self) -> None:
        """
        业务层先推送 error 再抛异常时，Web 层不应重复输出两个 error。
        """

        client = self.build_client(
            FakeLegalSession(error=RuntimeError("runner boom"), emit_error_event_before_raise=True)
        )

        response = client.post("/api/chat", json={"message": "公司没签劳动合同怎么办？"})

        items = self.parse_ndjson(response.text)
        error_items = [item for item in items if item.get("type") == "error"]
        self.assertEqual(1, len(error_items))
        self.assertIn("runner boom", error_items[0]["message"])
        self.assertEqual("done", items[-1]["type"])

    def test_startup_preload_success_records_state(self) -> None:
        """
        启动自动预热开启时，后台预热完成后应记录成功状态。
        """

        os.environ["LEGAL_RAG_PRELOAD"] = "1"
        session = FakeLegalSession()

        with TestClient(create_app(session=session)) as client:
            self.assertTrue(self.wait_until(lambda: session.preload_calls == 1))
            response = client.get("/api/health")

        self.assertEqual(1, session.preload_calls)
        self.assertTrue(response.json()["preloaded"])

    def test_startup_preload_failure_records_error(self) -> None:
        """
        启动自动预热失败不应阻止服务启动，应通过 health 暴露错误。
        """

        os.environ["LEGAL_RAG_PRELOAD"] = "1"
        session = FakeLegalSession(preload_error=RuntimeError("preload boom"))

        with TestClient(create_app(session=session)) as client:
            self.assertTrue(self.wait_until(lambda: client.get("/api/health").json()["preload_error"] is not None))
            response = client.get("/api/health")

        data = response.json()
        self.assertFalse(data["preloaded"])
        self.assertIn("preload boom", data["preload_error"])

    def test_startup_preload_runs_after_service_is_available(self) -> None:
        """
        自动预热耗时时，Web 服务应先可访问，不能卡在 RAG 加载完成之后才启动。
        """

        os.environ["LEGAL_RAG_PRELOAD"] = "1"
        session = FakeLegalSession(block_preload_until_released=True)
        health_ready = threading.Event()
        health_data: dict[str, object] = {}
        thread_error: list[BaseException] = []

        def run_client() -> None:
            """
            在独立线程中进入 TestClient 生命周期并请求健康检查。
            """

            try:
                with TestClient(create_app(session=session)) as client:
                    health_data.update(client.get("/api/health").json())
                    health_ready.set()
                    session.release_preload.wait(timeout=5)
            except BaseException as error:
                thread_error.append(error)
                health_ready.set()

        thread = threading.Thread(target=run_client)
        thread.start()
        self.assertTrue(session.preload_started.wait(timeout=2))
        try:
            self.assertTrue(health_ready.wait(timeout=0.5))
            self.assertFalse(thread_error)
            self.assertTrue(health_data["ok"])
            self.assertTrue(health_data["startup_preload_enabled"])
            self.assertFalse(health_data["preloaded"])
        finally:
            session.release_preload.set()
            thread.join(timeout=5)

        self.assertFalse(thread.is_alive())

    def test_concurrent_chat_returns_busy_stream(self) -> None:
        """
        全局 session 正在处理一轮咨询时，第二轮请求应返回 busy 错误流。
        """

        session = FakeLegalSession(block_until_released=True)
        client = self.build_client(session)
        responses: list[object] = []

        def run_first_request() -> None:
            responses.append(client.post("/api/chat", json={"message": "第一轮咨询"}))

        thread = threading.Thread(target=run_first_request)
        thread.start()
        self.assertTrue(session.started.wait(timeout=2))

        busy_response = client.post("/api/chat", json={"message": "第二轮咨询"})
        busy_items = self.parse_ndjson(busy_response.text)

        session.release.set()
        thread.join(timeout=5)

        self.assertEqual("error", busy_items[0]["type"])
        self.assertIn("当前已有咨询正在处理", busy_items[0]["message"])
        self.assertEqual("done", busy_items[-1]["type"])
        self.assertEqual(1, len(responses))


class PersistableFakeSession(FakeLegalSession):
    """
    支持快照导出/恢复的 fake 会话，用于多会话持久化测试。

    在 FakeLegalSession 的行为之上维护公开消息列表，并把资料事件补进返回的 events，
    以模拟真实 LegalConsultationSession 的 record_event（事件同时进入返回列表和回调）。
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.public_messages: list[dict[str, str]] = [{"role": "system", "content": "系统提示"}]
        self.restored_snapshots: list[dict[str, object]] = []

    def ask_with_events(self, text, *, on_event=None):
        answer, events = super().ask_with_events(text, on_event=on_event)
        if not self.pause:
            events = list(events)
            events.append(
                AgentEvent(
                    type="legal_reference_materials",
                    data={
                        "laws": [
                            {
                                "id": "law-0",
                                "material_type": "law",
                                "title": "《劳动合同法》第八十二条",
                                "subtitle": "未签书面劳动合同责任",
                                "detail": "二倍工资责任说明。",
                                "source": "本地法条库",
                                "issue": "未签书面劳动合同的责任",
                            }
                        ],
                        "web": [],
                        "warnings": [],
                    },
                )
            )
        self.public_messages.append({"role": "user", "content": text})
        self.public_messages.append({"role": "assistant", "content": answer})
        return answer, events

    def export_snapshot(self) -> dict[str, object]:
        """
        导出与真实会话结构一致的最小快照。
        """

        return {
            "messages": [dict(message) for message in self.public_messages],
            "case_state": {"summary": "fake 案件状态", "version": len(self.public_messages)},
        }

    def restore_snapshot(self, snapshot: dict[str, object]) -> None:
        """
        记录并应用恢复的快照，供测试断言恢复行为。
        """

        self.restored_snapshots.append(snapshot)
        messages = snapshot.get("messages")
        if isinstance(messages, list):
            self.public_messages = [dict(message) for message in messages]


class MultiSessionWebAppTests(unittest.TestCase):
    """
    多会话模式下的历史会话 API 和持久化行为测试。
    """

    def setUp(self) -> None:
        self.old_preload = os.environ.get("LEGAL_RAG_PRELOAD")
        os.environ["LEGAL_RAG_PRELOAD"] = "0"
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SessionStore(Path(self._tmp.name) / "sessions")

    def tearDown(self) -> None:
        if self.old_preload is None:
            os.environ.pop("LEGAL_RAG_PRELOAD", None)
        else:
            os.environ["LEGAL_RAG_PRELOAD"] = self.old_preload
        self._tmp.cleanup()

    def make_factory(self, **fake_kwargs):
        """
        构造记录创建实例的会话工厂。

        Returns:
            tuple: (工厂函数, 已创建实例列表)。
        """

        created: list[PersistableFakeSession] = []

        def factory() -> PersistableFakeSession:
            fake = PersistableFakeSession(**fake_kwargs)
            created.append(fake)
            return fake

        return factory, created

    def build_client(self, factory=None) -> TestClient:
        """
        创建多会话模式的测试客户端：注入 fake 工厂和临时目录存储。
        """

        if factory is None:
            factory, _ = self.make_factory()
        return TestClient(create_app(session_factory=factory, store=self.store))

    def parse_ndjson(self, text: str) -> list[dict[str, object]]:
        """
        将 NDJSON 响应文本解析为对象列表。
        """

        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def chat(self, client: TestClient, payload: dict[str, object]) -> list[dict[str, object]]:
        """
        发送一轮聊天并返回解析后的流事件。
        """

        response = client.post("/api/chat", json=payload)
        self.assertEqual(200, response.status_code)
        return self.parse_ndjson(response.text)

    def find_stream_item(self, items: list[dict[str, object]], item_type: str) -> dict[str, object] | None:
        """
        在流事件里查找指定类型的第一个条目。
        """

        for item in items:
            if item.get("type") == item_type:
                return item
        return None

    def test_chat_without_session_id_creates_and_persists_session(self) -> None:
        """
        不带 session_id 的请求应新建会话：流首返回 session 事件，轮末快照落盘并进入列表。
        """

        client = self.build_client()
        items = self.chat(client, {"message": "公司一直不签劳动合同，想主张二倍工资，该怎么办？"})

        session_item = items[0]
        self.assertEqual("session", session_item["type"])
        session_id = session_item["session_id"]
        self.assertTrue(session_id)
        self.assertIsNotNone(self.find_stream_item(items, "final"))
        self.assertEqual("done", items[-1]["type"])

        snapshot = self.store.load_snapshot(session_id)
        self.assertIsNotNone(snapshot)
        roles = [item["role"] for item in snapshot["messages"]]
        self.assertEqual(["system", "user", "assistant"], roles)
        self.assertEqual(1, len(snapshot["materials"]["laws"]))
        self.assertIsNone(snapshot["pending_supplement"])

        listing = client.get("/api/sessions").json()
        self.assertEqual(1, len(listing["sessions"]))
        entry = listing["sessions"][0]
        self.assertEqual(session_id, entry["session_id"])
        self.assertEqual(1, entry["turn_count"])
        self.assertIn("公司一直不签劳动合同", entry["title"])

    def test_chat_with_session_id_reuses_cached_session(self) -> None:
        """
        带 session_id 的后续轮次应复用内存中的同一会话实例，不重复创建。
        """

        factory, created = self.make_factory()
        client = self.build_client(factory)

        first_items = self.chat(client, {"message": "第一轮案情"})
        session_id = first_items[0]["session_id"]
        self.chat(client, {"session_id": session_id, "message": "第二轮追问"})

        self.assertEqual(1, len(created))
        self.assertEqual(["第一轮案情", "第二轮追问"], created[0].seen_inputs)
        meta = self.store.load_meta(session_id)
        self.assertEqual(2, meta["turn_count"])

    def test_chat_restores_session_from_disk_after_restart(self) -> None:
        """
        进程重启（新 app 实例）后带 session_id 请求应从磁盘快照恢复会话。
        """

        first_factory, _ = self.make_factory()
        first_client = self.build_client(first_factory)
        items = self.chat(first_client, {"message": "重启前的案情"})
        session_id = items[0]["session_id"]

        second_factory, second_created = self.make_factory()
        second_client = self.build_client(second_factory)
        self.chat(second_client, {"session_id": session_id, "message": "重启后的追问"})

        self.assertEqual(1, len(second_created))
        restored = second_created[0].restored_snapshots
        self.assertEqual(1, len(restored))
        contents = [item["content"] for item in restored[0]["messages"]]
        self.assertIn("重启前的案情", contents)

    def test_chat_with_unknown_session_id_returns_404(self) -> None:
        """
        目标会话不存在时应返回 404，而不是静默新建。
        """

        client = self.build_client()
        response = client.post(
            "/api/chat",
            json={"session_id": "sess_20990101_000000_dead", "message": "案情"},
        )
        self.assertEqual(404, response.status_code)

    def test_session_detail_returns_messages_and_materials(self) -> None:
        """
        会话详情应返回不含 system 的公开消息和白名单资料。
        """

        client = self.build_client()
        items = self.chat(client, {"message": "公司不签劳动合同"})
        session_id = items[0]["session_id"]

        detail = client.get(f"/api/sessions/{session_id}")
        self.assertEqual(200, detail.status_code)
        data = detail.json()
        self.assertEqual(session_id, data["session_id"])
        self.assertEqual(["user", "assistant"], [item["role"] for item in data["messages"]])
        self.assertEqual("《劳动合同法》第八十二条", data["materials"]["laws"][0]["title"])
        self.assertIsNone(data["pending_supplement"])

    def test_pause_turn_persists_pending_supplement(self) -> None:
        """
        暂停补充轮应把补充请求写入快照，详情接口能恢复补充面板数据。
        """

        factory, _ = self.make_factory(pause=True)
        client = self.build_client(factory)
        items = self.chat(client, {"message": "公司不签劳动合同"})
        session_id = items[0]["session_id"]
        self.assertIsNotNone(self.find_stream_item(items, "pause"))

        data = client.get(f"/api/sessions/{session_id}").json()
        supplement = data["pending_supplement"]
        self.assertIsNotNone(supplement)
        self.assertIn("入职时间是什么时候？", supplement["questions"])
        self.assertIn("工资流水", supplement["evidence_gaps"])

    def test_delete_session_removes_from_list_and_detail(self) -> None:
        """
        删除会话后列表和详情都不应再返回该会话。
        """

        client = self.build_client()
        items = self.chat(client, {"message": "待删除的案情"})
        session_id = items[0]["session_id"]

        delete_response = client.delete(f"/api/sessions/{session_id}")
        self.assertEqual(200, delete_response.status_code)
        self.assertEqual([], client.get("/api/sessions").json()["sessions"])
        self.assertEqual(404, client.get(f"/api/sessions/{session_id}").status_code)
        self.assertEqual(404, client.delete(f"/api/sessions/{session_id}").status_code)

    def test_failed_turn_not_persisted_and_not_listed(self) -> None:
        """
        失败轮不应写入快照；没有成功轮次的空会话不进入历史列表。
        """

        factory, created = self.make_factory(error=RuntimeError("模型超时"))
        client = self.build_client(factory)
        items = self.chat(client, {"message": "会失败的案情"})

        session_id = items[0]["session_id"]
        self.assertIsNotNone(self.find_stream_item(items, "error"))
        self.assertIsNone(self.store.load_snapshot(session_id))
        self.assertEqual([], client.get("/api/sessions").json()["sessions"])
        self.assertEqual(1, len(created))

    def test_single_session_mode_keeps_legacy_behavior(self) -> None:
        """
        注入单 session 的兼容模式不应产生 session 流事件，历史列表保持为空。
        """

        client = TestClient(create_app(session=FakeLegalSession()))
        response = client.post("/api/chat", json={"message": "旧模式案情"})
        items = self.parse_ndjson(response.text)

        self.assertIsNone(self.find_stream_item(items, "session"))
        self.assertEqual([], client.get("/api/sessions").json()["sessions"])


if __name__ == "__main__":
    unittest.main()
