"""
法律咨询 Agent Web 服务入口。

本模块提供一个本地单用户 Web 控制台：浏览器负责展示聊天与执行进度，后端直接复用
LegalConsultationSession 完成案件状态更新、法条 RAG、风险识别和最终回答生成。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
import queue
import threading
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_system.agent.events import AgentEvent
from agent_system.legal_consultation.factory import create_legal_consultation_session_factory
from agent_system.memory import (
    MemoryStore,
    MemoryStoreError,
    build_case_memory_from_snapshot,
    recalled_memories_to_prompt_payload,
)
from agent_system.storage import SessionStore, SessionStoreError


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
NDJSON_MEDIA_TYPE = "application/x-ndjson"
# 会话持久化根目录。相对路径基于项目根（uvicorn 从项目根启动），和 data/chroma 同级。
DEFAULT_SESSIONS_DIR = Path("data") / "sessions"
# 跨会话案件记忆根目录，与会话存档同级、互相独立：删除单个会话时只精确删除对应记忆。
DEFAULT_MEMORY_DIR = Path("data") / "memory"


class SupportsLegalWebSession(Protocol):
    """
    Web 层依赖的法律咨询会话最小协议。

    这样单元测试可以注入 fake session，避免真实调用 LLM、BGE-M3 或 Chroma。
    """

    def preload_resources(self) -> None:
        """
        预热本地法律检索资源。
        """

    def ask_with_events(
        self,
        text: str,
        *,
        on_event: Callable[[AgentEvent], None] | None = None,
        recalled_memories: list[dict[str, Any]] | None = None,
        allow_pause: bool = True,
    ) -> tuple[str, list[AgentEvent]]:
        """
        执行一轮法律咨询，并通过回调实时输出过程事件。

        Args:
            text: 用户输入的案情或追问。
            on_event: 可选过程事件回调。
            recalled_memories: 可选跨会话历史记忆负载。后端只在确有召回时传入该参数，
                因此不支持记忆的旧会话实现（含既有测试 fake）无需修改也能继续工作。
            allow_pause: 是否允许本轮暂停等待补充。后端只在用户明确跳过补充时传 False，
                与 recalled_memories 一样按需传入，旧实现无需修改。

        Returns:
            tuple[str, list[AgentEvent]]: 最终回答和过程事件。
        """


class ChatRequest(BaseModel):
    """
    前端发送的一轮聊天请求。

    Attributes:
        session_id: 目标会话 ID。为空表示新建会话（多会话模式）或使用注入会话（单会话模式）。
        message: 用户输入的案情或追问文本。
        supplement_answers: 用户对暂停追问逐项填写的回答。
        selected_questions: 用户确认本轮要处理的追问项。
        selected_evidence_gaps: 用户确认可补充或正在准备的证据材料。
        free_text: 用户额外补充的自由文本。
        skip_supplement: 用户明确表示无法补充，要求基于现有信息继续分析。
            为 True 时后端合成兜底输入并禁用本轮暂停判定，避免流程被阻塞性补充卡死。
    """

    session_id: str | None = None
    message: str = ""
    supplement_answers: dict[str, str] | None = None
    selected_questions: list[str] | None = None
    selected_evidence_gaps: list[str] | None = None
    free_text: str | None = None
    skip_supplement: bool = False


EVENT_TITLES: dict[str, str] = {
    "legal_step": "执行步骤",
    "legal_selfheal": "链路自修复",
    "legal_memory_recalled": "历史咨询记忆已唤起",
    "legal_turn_metrics": "本轮执行指标",
    "legal_rag_query_started": "法条检索中",
    "case_state_updated": "案件状态已更新",
    "legal_missing_details_suggested": "可先补充的关键信息",
    "legal_supplement_required": "等待补充关键信息",
    "legal_supplement_skipped": "已按现有信息继续分析",
    "legal_case_rag_done": "案情拆解与检索完成",
    "legal_web_search_started": "公网案例与司法实践检索中",
    "legal_web_search_done": "公网案例与司法实践检索完成",
    "legal_reference_materials": "参考资料已整理",
    "legal_risk_analyzed": "风险识别完成",
    "legal_analysis_catalog_built": "案情目录已生成",
    "legal_next_action_decided": "下一步动作已判断",
    "tool_call": "工具调用",
    "tool_result": "工具结果",
    "message_done": "最终回答生成完成",
    "error": "执行出错",
}


def create_app(
    session: SupportsLegalWebSession | None = None,
    *,
    session_factory: Callable[[], SupportsLegalWebSession] | None = None,
    store: SessionStore | None = None,
    memory_store: MemoryStore | None = None,
) -> FastAPI:
    """
    创建 FastAPI 应用。

    Args:
        session: 可选法律咨询会话。传入时进入单会话兼容模式：所有请求共用该会话，
            不做历史会话管理、磁盘持久化和跨会话记忆。主要供旧测试和特殊装配使用。
        session_factory: 可选会话工厂（多会话模式）。为空时使用共享重资源的默认工厂。
        store: 可选会话存储。为空且处于多会话模式时使用 `data/sessions` 默认目录。
        memory_store: 可选跨会话记忆存储。为空且处于多会话模式时使用 `data/memory` 默认目录。

    Returns:
        FastAPI: 已挂载静态文件和接口路由的 Web 应用。
    """

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        """
        应用生命周期钩子。

        使用 lifespan 而不是旧的 on_event，是为了避免 FastAPI 新版本的弃用告警。启动预热放到
        后台线程里执行，原因是 BGE-M3 和 Chroma 首次加载可能较慢；先让 Web 服务完成启动，
        浏览器才能尽快打开页面并显示当前状态。
        """

        if should_preload_on_startup():
            start_background_preload()
        yield

    app = FastAPI(title="Legal Agent Web UI", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # 单会话兼容模式：注入 session 时不启用持久化，行为与历史版本一致。
    # 多会话模式：session 为空时按 session_id 管理多个会话，并把每轮结果落盘。
    single_session_mode = session is not None
    app.state.legal_session = session
    app.state.session_factory = None if single_session_mode else (session_factory or create_legal_consultation_session_factory())
    app.state.store = None if single_session_mode else (store or SessionStore(DEFAULT_SESSIONS_DIR))
    # 跨会话记忆与会话持久化同开同关：单会话兼容模式没有 session_id，无法沉淀和排除自身记忆。
    app.state.memory_store = None if single_session_mode else (memory_store or MemoryStore(DEFAULT_MEMORY_DIR))
    # 内存中的活跃会话缓存：session_id -> 会话实例。磁盘快照在每轮成功后写入，
    # 缓存里的对象和磁盘内容保持一致；进程重启后按需从磁盘恢复。
    app.state.sessions = {}
    app.state.registry_lock = threading.Lock()
    app.state.preload_session = None
    app.state.session_lock = threading.Lock()
    app.state.chat_lock = threading.Lock()
    app.state.preload_lock = threading.Lock()
    app.state.preloaded = False
    app.state.preload_error = None
    app.state.preload_thread = None

    def get_preload_session() -> SupportsLegalWebSession:
        """
        获取用于预热的会话。

        单会话模式直接用注入会话；多会话模式惰性创建一个仅供预热的会话实例。
        预热的真实目标是工厂内部共享的 BGE-M3/Chroma 资源，任何一个会话预热后，
        其余会话（包括历史恢复的会话）都能直接复用已加载的模型。
        """

        if app.state.legal_session is not None:
            return app.state.legal_session

        with app.state.session_lock:
            if app.state.preload_session is None:
                app.state.preload_session = app.state.session_factory()
        return app.state.preload_session

    def get_or_load_session(session_id: str) -> SupportsLegalWebSession:
        """
        获取内存缓存中的会话，未命中时从磁盘快照恢复。

        Args:
            session_id: 目标会话 ID。

        Raises:
            HTTPException: 会话不存在时返回 404；快照损坏时返回 500。
        """

        with app.state.registry_lock:
            cached = app.state.sessions.get(session_id)
        if cached is not None:
            return cached

        store_instance: SessionStore = app.state.store
        if not store_instance.session_exists(session_id):
            raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在或已删除。")
        try:
            snapshot = store_instance.load_snapshot(session_id)
        except SessionStoreError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

        new_session = app.state.session_factory()
        restore = getattr(new_session, "restore_snapshot", None)
        if snapshot is not None and callable(restore):
            restore(snapshot)
        with app.state.registry_lock:
            # setdefault 兜底并发加载同一会话：第一个完成的实例胜出，避免两个副本各自演化。
            return app.state.sessions.setdefault(session_id, new_session)

    def persist_committed_turn(
        session_id: str | None,
        target_session: SupportsLegalWebSession,
        events: list[AgentEvent],
        answer: str,
    ) -> None:
        """
        把一轮成功（含暂停补充）的会话状态写入磁盘。

        Args:
            session_id: 会话 ID；单会话模式为 None，直接跳过。
            target_session: 本轮使用的会话实例。
            events: 本轮业务事件，用于提取资料侧栏和暂停补充数据。
            answer: 最终回答或补充提示文本。

        Raises:
            SessionStoreError: 底层写盘失败时抛出，由调用方决定如何提示。
        """

        store_instance: SessionStore | None = app.state.store
        if store_instance is None or session_id is None:
            return
        export = getattr(target_session, "export_snapshot", None)
        if not callable(export):
            return

        snapshot = export()
        snapshot["materials"] = extract_reference_materials(events)
        snapshot["pending_supplement"] = extract_pending_supplement(events, fallback_message=answer)
        messages = snapshot.get("messages") if isinstance(snapshot.get("messages"), list) else []
        turn_count = sum(1 for item in messages if isinstance(item, dict) and item.get("role") == "user")
        store_instance.save_snapshot(
            session_id,
            snapshot,
            title=derive_session_title(messages),
            turn_count=turn_count,
        )
        # 轮级 metrics 随 turn_committed 一起写入 events.jsonl：这是跨进程重启后仍可审计的
        # 持久观测数据，/api/metrics 聚合就以它为唯一数据源。写盘前先过一遍 Web 白名单，
        # 保证磁盘上的观测字段和推给浏览器的完全一致，不会多存内部诊断内容。
        turn_event_data: dict[str, Any] = {"turn_count": turn_count}
        metrics = extract_turn_metrics(events)
        if metrics is not None:
            turn_event_data["metrics"] = metrics
        store_instance.append_event(session_id, "turn_committed", turn_event_data, turn_id=turn_count)

    def recall_memories_for_input(user_input: str, session_id: str | None) -> list[dict[str, Any]]:
        """
        检索与本轮输入相关的跨会话历史记忆，返回可注入会话的白名单负载。

        Args:
            user_input: 本轮合并后的用户输入。
            session_id: 当前会话 ID，用于排除该会话自身的记忆（其知识已在 case_state 里）。

        Returns:
            list[dict[str, Any]]: 白名单记忆负载；未启用记忆、无命中或检索失败时返回空列表。
            记忆是辅助信号，任何异常都降级为“无记忆”，绝不阻断咨询链路。
        """

        memory_store_instance: MemoryStore | None = app.state.memory_store
        if memory_store_instance is None:
            return []
        try:
            recalls = memory_store_instance.search(user_input, exclude_session_id=session_id)
            return recalled_memories_to_prompt_payload(recalls)
        except Exception:
            return []

    def persist_case_memory(session_id: str | None, target_session: SupportsLegalWebSession) -> None:
        """
        把本轮提交后的案件知识沉淀为跨会话记忆（upsert，一会话一条）。

        Args:
            session_id: 会话 ID；单会话模式为 None，直接跳过。
            target_session: 本轮使用的会话实例。

        Why:
            case_state 是状态更新 LLM 已经蒸馏过的结构化知识，这里做确定性提取即可完成
            “知识沉淀”，不增加任何 LLM 调用。暂停补充轮同样提交了 case_state，因此同样沉淀。

        Raises:
            MemoryStoreError: 记忆写盘失败时抛出，由调用方降级为非致命提示。
        """

        memory_store_instance: MemoryStore | None = app.state.memory_store
        if memory_store_instance is None or session_id is None:
            return
        export = getattr(target_session, "export_snapshot", None)
        if not callable(export):
            return

        snapshot = export()
        messages = snapshot.get("messages") if isinstance(snapshot.get("messages"), list) else []
        turn_count = sum(1 for item in messages if isinstance(item, dict) and item.get("role") == "user")
        memory = build_case_memory_from_snapshot(
            session_id,
            snapshot,
            title=derive_session_title(messages),
            turn_count=turn_count,
        )
        if memory is None:
            # 案件状态还没有可沉淀内容（如首轮即失败恢复的空会话），跳过而不是写空记忆。
            return
        memory_store_instance.save(memory)

    def preload_current_session() -> None:
        """
        预热当前全局会话的本地资源。

        使用独立锁的原因是预热可能加载 BGE-M3 和 Chroma，耗时较长；锁能避免页面手动重试时
        并发触发多次模型加载。
        """

        with app.state.preload_lock:
            get_preload_session().preload_resources()
            app.state.preloaded = True
            app.state.preload_error = None

    def start_background_preload() -> None:
        """
        在后台线程中执行启动预热。

        预热仍复用 chat_lock。原因是预热和咨询都会触发底层 RAG 初始化，串行执行能避免同一模型
        或 Chroma collection 被并发加载；但后台线程不会阻塞 FastAPI 完成启动。
        """

        def run_preload() -> None:
            """
            捕获后台预热异常，并写入健康检查状态。
            """

            try:
                with app.state.chat_lock:
                    preload_current_session()
            except Exception as error:
                app.state.preloaded = False
                app.state.preload_error = str(error)

        thread = threading.Thread(target=run_preload, name="legal-rag-preload", daemon=True)
        app.state.preload_thread = thread
        thread.start()

    @app.get("/")
    def index() -> FileResponse:
        """
        返回单页前端入口。

        Returns:
            FileResponse: `web_app/static/index.html` 文件响应。
        """

        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """
        返回 Web 服务健康状态。

        Returns:
            dict[str, Any]: 当前服务、预热状态和最近一次预热错误。
        """

        return {
            "ok": True,
            "service": "legal-agent-web",
            "status": "ready",
            "preloaded": bool(app.state.preloaded),
            "preload_error": app.state.preload_error,
            "startup_preload_enabled": should_preload_on_startup(),
        }

    @app.post("/api/preload")
    def preload() -> JSONResponse:
        """
        手动预热本地法律检索资源。

        Returns:
            JSONResponse: 预热成功或失败的 JSON 结果。
        """

        if not app.state.chat_lock.acquire(blocking=False):
            return JSONResponse(
                status_code=409,
                content={"ok": False, "message": "当前已有咨询正在处理，请稍后再预热。"},
            )
        try:
            preload_current_session()
        except Exception as error:
            app.state.preloaded = False
            app.state.preload_error = str(error)
            return JSONResponse(
                status_code=500,
                content={"ok": False, "message": "本地法条 RAG 预热失败", "error": str(error)},
            )
        finally:
            app.state.chat_lock.release()
        return JSONResponse(content={"ok": True, "message": "本地法条 RAG 预热完成"})

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> StreamingResponse:
        """
        接收用户消息，并以 NDJSON 流式返回法律咨询执行事件。

        Args:
            request: 前端提交的聊天请求；多会话模式下 session_id 为空表示新建会话。

        Returns:
            StreamingResponse: 每行一个 JSON 对象的 NDJSON 流。首个事件为 session
            （多会话模式），前端用它记录本轮实际使用的会话 ID。

        Raises:
            HTTPException: 用户输入为空返回 400；目标会话不存在返回 404。
        """

        user_input = build_chat_input(request)
        if not user_input:
            raise HTTPException(status_code=400, detail="message 或补充内容不能为空。")

        if not app.state.chat_lock.acquire(blocking=False):
            return StreamingResponse(iter_busy_stream(), media_type=NDJSON_MEDIA_TYPE)

        # 锁一旦获取，任何提前退出路径（会话工厂抛 RuntimeError、线程创建失败等）都必须释放；
        # 否则后续所有 /api/chat 会永远收到“正在处理”，服务假死。worker 成功启动后，
        # 释放责任移交给 run_agent 的 finally，本函数的 finally 不再重复释放。
        session_id: str | None = None
        worker_started = False
        try:
            # 会话解析放在 chat_lock 之内：一方面新会话目录只在确定本轮会执行时才创建，
            # 避免忙碌重试在磁盘上留下一堆空会话；另一方面与删除接口天然互斥。
            try:
                if app.state.legal_session is not None:
                    target_session: SupportsLegalWebSession = app.state.legal_session
                else:
                    requested_id = str(request.session_id or "").strip()
                    if requested_id:
                        target_session = get_or_load_session(requested_id)
                        session_id = requested_id
                    else:
                        session_id = app.state.store.create_session()
                        target_session = app.state.session_factory()
                        with app.state.registry_lock:
                            app.state.sessions[session_id] = target_session
            except SessionStoreError as error:
                raise HTTPException(status_code=500, detail=str(error)) from error

            # 记忆检索在启动后台线程前完成：它只读本地小文件，毫秒级返回，
            # 放在这里可以让整轮链路（包括状态更新 prompt）从一开始就带上历史背景。
            recalled_memories = recall_memories_for_input(user_input, session_id)

            event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
            error_event_forwarded = False
            search_started_forwarded = False

            def push(item: dict[str, Any]) -> None:
                """
                将待发送的事件写入线程安全队列。

                Args:
                    item: 已规范化的前端事件。
                """

                event_queue.put(item)

            if session_id is not None:
                # 首个流事件回传会话 ID。新会话由后端分配 ID，前端必须先拿到它，
                # 后续轮次和刷新后的历史恢复才能路由到同一个会话。
                push({"type": "session", "session_id": session_id})

            def on_event(event: AgentEvent) -> None:
                """
                接收业务层事件并转成 Web 前端易消费的格式。
                """

                nonlocal error_event_forwarded, search_started_forwarded
                event_type = str(event.type)
                if event_type == "error":
                    error_event_forwarded = True
                    error_data = event.data if isinstance(event.data, dict) else {"error": event.data}
                    push({"type": "error", "message": str(error_data.get("error") or error_data)})
                    return
                if event_type == "answer_delta":
                    # 最终回答增量走顶层 answer_delta 流事件，不进入右侧执行进度区。
                    # 原因是每个增量都包装成 event 卡片会把进度区刷爆，前端只需要把它拼进聊天气泡。
                    delta_data = event.data if isinstance(event.data, dict) else {}
                    delta_text = str(delta_data.get("delta") or "")
                    if delta_text:
                        push({"type": "answer_delta", "delta": delta_text})
                    return
                if event_type == "legal_rag_query_started":
                    if search_started_forwarded:
                        return
                    search_started_forwarded = True
                normalized_event = normalize_agent_event(event)
                if normalized_event is not None:
                    push(normalized_event)

            def run_agent() -> None:
                """
                在后台线程中执行同步法律咨询链路。

                这样做的原因是 LLM 调用和本地 RAG 都是阻塞型工作；放到后台线程后，HTTP 流可以边取
                队列边向浏览器输出事件，不会等整轮任务结束才一次性返回。
                """

                try:
                    # 可选参数只在需要时传入：不带记忆、不跳过补充的常规轮保持旧调用形态，
                    # 兼容尚未支持这些参数的旧会话实现和测试 fake。
                    ask_kwargs: dict[str, Any] = {}
                    if recalled_memories:
                        ask_kwargs["recalled_memories"] = recalled_memories
                    if request.skip_supplement:
                        # 用户明确表示无法补充：本轮禁用暂停判定，强制基于现有信息走完整链路。
                        ask_kwargs["allow_pause"] = False
                    answer, events = target_session.ask_with_events(user_input, on_event=on_event, **ask_kwargs)
                    try:
                        # 持久化在 final/pause 事件之前执行：前端收到 final 后可能立即刷新会话列表，
                        # 此时快照必须已经落盘，否则列表里拿到的还是上一轮的标题和轮次。
                        persist_committed_turn(session_id, target_session, events, answer)
                    except SessionStoreError as persist_error:
                        # 回答本身已成功，持久化失败只影响历史保存；用普通事件卡片提示，
                        # 不走顶层 error 流事件，避免前端把整轮标记为失败。
                        push(
                            {
                                "type": "event",
                                "event_type": "error",
                                "title": "会话保存失败",
                                "data": {"error": str(persist_error)},
                            }
                        )
                    try:
                        persist_case_memory(session_id, target_session)
                    except Exception as memory_error:
                        # 记忆沉淀失败同样不影响本轮回答，也不影响会话快照；单独软提示。
                        push(
                            {
                                "type": "event",
                                "event_type": "error",
                                "title": "记忆沉淀失败",
                                "data": {"error": str(memory_error)},
                            }
                        )
                    pause_event = find_event(events, "legal_supplement_required")
                    if pause_event is not None:
                        push(build_pause_stream_item(pause_event, fallback_message=answer))
                    else:
                        push({"type": "final", "answer": answer})
                except Exception as error:
                    if app.state.store is not None and session_id is not None:
                        try:
                            app.state.store.append_event(session_id, "turn_failed", {"error": str(error)})
                        except SessionStoreError:
                            # 失败轮的事件记录属于尽力而为，写不进去也不能掩盖原始业务错误。
                            pass
                    if not error_event_forwarded:
                        push({"type": "error", "message": str(error)})
                finally:
                    push({"type": "done"})
                    event_queue.put(None)
                    app.state.chat_lock.release()

            worker = threading.Thread(target=run_agent, name="legal-web-chat", daemon=True)
            worker.start()
            worker_started = True
            return StreamingResponse(iter_queue_as_ndjson(event_queue, worker), media_type=NDJSON_MEDIA_TYPE)
        finally:
            if not worker_started:
                app.state.chat_lock.release()

    @app.get("/api/sessions")
    def list_sessions() -> dict[str, Any]:
        """
        返回历史会话列表，按最近更新时间倒序。

        Returns:
            dict[str, Any]: `{"sessions": [...]}`；单会话兼容模式返回空列表。
            尚未完成任何一轮的空会话不进入列表，它们没有可恢复的内容。
        """

        store_instance: SessionStore | None = app.state.store
        if store_instance is None:
            return {"sessions": []}
        sessions = [
            {
                "session_id": meta.get("session_id"),
                "title": str(meta.get("title") or ""),
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
                "turn_count": safe_int(meta.get("turn_count")),
            }
            for meta in store_instance.list_sessions()
            if safe_int(meta.get("turn_count")) > 0
        ]
        return {"sessions": sessions}

    @app.get("/api/sessions/{session_id}")
    def session_detail(session_id: str) -> dict[str, Any]:
        """
        返回单个历史会话的可恢复内容。

        Args:
            session_id: 会话 ID。

        Returns:
            dict[str, Any]: 公开聊天消息、最后一轮参考资料和未完成的补充请求。
            消息只包含 user/assistant；system prompt 属于内部装配，不透出给浏览器。

        Raises:
            HTTPException: 持久化未启用、会话不存在或尚无快照时返回 404。
        """

        store_instance: SessionStore | None = app.state.store
        if store_instance is None:
            raise HTTPException(status_code=404, detail="当前模式未启用会话持久化。")
        try:
            snapshot = store_instance.load_snapshot(session_id)
            meta = store_instance.load_meta(session_id)
        except SessionStoreError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if snapshot is None:
            raise HTTPException(status_code=404, detail="该会话还没有可恢复的内容。")

        raw_messages = snapshot.get("messages") if isinstance(snapshot.get("messages"), list) else []
        messages = [
            {"role": str(item.get("role")), "text": str(item.get("content") or "")}
            for item in raw_messages
            if isinstance(item, dict) and item.get("role") in {"user", "assistant"}
        ]
        raw_materials = snapshot.get("materials") if isinstance(snapshot.get("materials"), dict) else {}
        # 快照文件人工可编辑，返回前再过一遍白名单，保持与实时流相同的安全边界。
        materials = {
            "laws": normalize_reference_materials(raw_materials.get("laws"), limit=8),
            "web": normalize_reference_materials(raw_materials.get("web"), limit=8),
            "warnings": normalize_safe_text_list(raw_materials.get("warnings"), limit=5),
        }
        return {
            "session_id": session_id,
            "title": str(meta.get("title") or ""),
            "updated_at": meta.get("updated_at"),
            "turn_count": safe_int(meta.get("turn_count")),
            "messages": messages,
            "materials": materials,
            "pending_supplement": normalize_pending_supplement(snapshot.get("pending_supplement")),
        }

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, Any]:
        """
        删除历史会话及其磁盘目录。

        Raises:
            HTTPException: 持久化未启用或会话不存在返回 404；有咨询正在处理返回 409。
        """

        store_instance: SessionStore | None = app.state.store
        if store_instance is None:
            raise HTTPException(status_code=404, detail="当前模式未启用会话持久化。")
        if not store_instance.session_exists(session_id):
            raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在或已删除。")
        # 复用 chat_lock 与咨询互斥：正在执行的一轮结束后才允许删除，避免轮末快照把目录写回来。
        if not app.state.chat_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="当前已有咨询正在处理，请稍后再删除。")
        try:
            # 先删记忆再删会话目录：记忆删除失败时整个操作以 500 失败、会话保留，用户重试
            # 即可；反过来先删会话，残留的记忆会让“已删除会话”的知识继续出现在召回里。
            if app.state.memory_store is not None:
                app.state.memory_store.delete(session_id)
            store_instance.delete_session(session_id)
            with app.state.registry_lock:
                app.state.sessions.pop(session_id, None)
        except (SessionStoreError, MemoryStoreError) as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        finally:
            app.state.chat_lock.release()
        return {"ok": True, "session_id": session_id}

    @app.get("/api/metrics")
    def metrics_summary() -> dict[str, Any]:
        """
        聚合所有会话的轮级运行指标。

        Returns:
            dict[str, Any]: 成功轮数、失败轮数、成功率、总耗时和 LLM usage 汇总。
            单会话兼容模式没有持久化事件，返回全零结果。

        Why:
            直接扫描各会话的 events.jsonl 而不是维护内存计数器：turn_committed/turn_failed
            本来就是每轮的持久审计流水，以它为唯一数据源可以让指标跨进程重启保持一致，
            也不用引入任何新的存储结构。本地单用户的事件量很小，全量扫描足够快。
        """

        store_instance: SessionStore | None = app.state.store
        turns = 0
        failed_turns = 0
        total_duration_ms = 0
        llm_calls = 0
        total_tokens = 0
        if store_instance is not None:
            for meta in store_instance.list_sessions():
                try:
                    session_events = store_instance.read_events(str(meta.get("session_id") or ""))
                except SessionStoreError:
                    # 单个会话事件文件损坏只影响该会话的统计；聚合指标是辅助观测，
                    # 跳过坏会话继续汇总，不让一个坏文件把整个接口打成 500。
                    continue
                for item in session_events:
                    item_type = item.get("type")
                    if item_type == "turn_failed":
                        failed_turns += 1
                        continue
                    if item_type != "turn_committed":
                        continue
                    turns += 1
                    data = item.get("data") if isinstance(item.get("data"), dict) else {}
                    turn_metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
                    total_duration_ms += safe_int(turn_metrics.get("total_duration_ms"))
                    llm_usage = turn_metrics.get("llm_usage") if isinstance(turn_metrics.get("llm_usage"), dict) else {}
                    llm_calls += safe_int(llm_usage.get("calls"))
                    total_tokens += safe_int(llm_usage.get("total_tokens"))
        completed = turns + failed_turns
        return {
            "ok": True,
            "turns": turns,
            "failed_turns": failed_turns,
            "success_rate": round(turns / completed, 4) if completed else 0.0,
            "total_duration_ms": total_duration_ms,
            "llm_calls": llm_calls,
            "total_tokens": total_tokens,
        }

    return app


def build_chat_input(request: ChatRequest) -> str:
    """
    把普通消息和暂停补充表单合并成一段用户输入。

    Args:
        request: 前端请求体。

    Returns:
        str: 传给法律咨询会话的用户输入；为空表示没有有效内容。
    """

    parts: list[str] = []
    message = request.message.strip()
    if message:
        parts.append(message)

    supplement_answers = request.supplement_answers or {}
    answer_lines: list[str] = []
    for question, answer in supplement_answers.items():
        question_text = str(question).strip()
        answer_text = str(answer).strip()
        if question_text and answer_text:
            answer_lines.append(f"- {question_text}：{answer_text}")
    if answer_lines:
        parts.append("【用户补充的逐项回答】")
        parts.extend(answer_lines)

    selected_questions = normalize_safe_text_list(request.selected_questions, limit=10)
    if selected_questions:
        parts.append("【用户确认需要处理的问题】")
        parts.extend(f"- {item}" for item in selected_questions)

    selected_evidence_gaps = normalize_safe_text_list(request.selected_evidence_gaps, limit=10)
    if selected_evidence_gaps:
        parts.append("【用户确认可补充或正在准备的证据材料】")
        parts.extend(f"- {item}" for item in selected_evidence_gaps)

    free_text = str(request.free_text or "").strip()
    if free_text:
        parts.append("【其他补充说明】")
        parts.append(free_text)

    if request.skip_supplement:
        # 跳过补充时必须保证输入非空：即使用户什么都没填，也要给状态更新器一句明确的
        # “无法补充”声明，配合 prompt 规则避免下一轮再次触发同样的暂停。
        parts.append("我暂时无法补充更多信息，请基于目前已提供的信息继续分析。")

    return "\n".join(parts).strip()


def find_event(events: list[AgentEvent], event_type: str) -> AgentEvent | None:
    """
    在事件列表中查找指定类型事件。

    Args:
        events: 本轮业务事件。
        event_type: 目标事件类型。

    Returns:
        AgentEvent | None: 找到则返回事件，否则返回 None。
    """

    for event in events:
        if str(event.type) == event_type:
            return event
    return None


def build_pause_stream_item(event: AgentEvent, *, fallback_message: str) -> dict[str, Any]:
    """
    把内部暂停事件转换为 Web 顶层 pause 流事件。

    Args:
        event: `legal_supplement_required` 内部事件。
        fallback_message: 事件缺少 message 时使用的兜底文本。

    Returns:
        dict[str, Any]: 前端可直接渲染的 pause 事件。
    """

    data = event.data if isinstance(event.data, dict) else {}
    return {
        "type": "pause",
        "reason": str(data.get("reason") or ""),
        "message": str(data.get("message") or fallback_message),
        "questions": normalize_safe_text_list(data.get("questions"), limit=5),
        "evidence_gaps": normalize_safe_text_list(data.get("evidence_gaps"), limit=5),
        "state_version": data.get("state_version"),
    }


def extract_reference_materials(events: list[AgentEvent]) -> dict[str, Any]:
    """
    从本轮事件中提取资料侧栏快照。

    Args:
        events: 本轮业务事件。

    Returns:
        dict[str, Any]: 已按白名单精简的 `{"laws", "web", "warnings"}` 结构。持久化保存
        脱敏后的展示结构而不是内部原始结果，这样历史会话恢复时可以直接喂给前端资料栏。
    """

    event = find_event(events, "legal_reference_materials")
    data = event.data if event is not None and isinstance(event.data, dict) else {}
    return {
        "laws": normalize_reference_materials(data.get("laws"), limit=8),
        "web": normalize_reference_materials(data.get("web"), limit=8),
        "warnings": normalize_safe_text_list(data.get("warnings"), limit=5),
    }


def extract_turn_metrics(events: list[AgentEvent]) -> dict[str, Any] | None:
    """
    从本轮事件中提取白名单化的轮级运行指标。

    Args:
        events: 本轮业务事件。

    Returns:
        dict[str, Any] | None: 阶段耗时、总耗时、LLM usage 和自修复次数；本轮没有
        metrics 事件时返回 None，调用方直接省略该字段而不是写入空对象。
    """

    event = find_event(events, "legal_turn_metrics")
    if event is None or not isinstance(event.data, dict):
        return None
    safe_event = sanitize_event_for_web("legal_turn_metrics", event.data)
    if safe_event is None:
        return None
    return safe_event[1]


def extract_pending_supplement(events: list[AgentEvent], *, fallback_message: str) -> dict[str, Any] | None:
    """
    从本轮事件中提取未完成的补充请求。

    Args:
        events: 本轮业务事件。
        fallback_message: 暂停提示缺失时的兜底文本。

    Returns:
        dict[str, Any] | None: 暂停轮返回补充请求数据；正常回答轮返回 None，
        同时覆盖掉快照里上一轮可能遗留的暂停状态。
    """

    pause_event = find_event(events, "legal_supplement_required")
    if pause_event is None:
        return None
    item = build_pause_stream_item(pause_event, fallback_message=fallback_message)
    item.pop("type", None)
    return item


def normalize_pending_supplement(value: Any) -> dict[str, Any] | None:
    """
    规范化快照中的补充请求数据。

    Args:
        value: 快照里的 pending_supplement 字段。

    Returns:
        dict[str, Any] | None: 白名单化后的补充请求；无有效内容时返回 None。
    """

    if not isinstance(value, dict):
        return None
    questions = normalize_safe_text_list(value.get("questions"), limit=5)
    evidence_gaps = normalize_safe_text_list(value.get("evidence_gaps"), limit=5)
    message = str(value.get("message") or "").strip()
    if not questions and not evidence_gaps and not message:
        return None
    return {
        "reason": str(value.get("reason") or ""),
        "message": message,
        "questions": questions,
        "evidence_gaps": evidence_gaps,
        "state_version": value.get("state_version"),
    }


def derive_session_title(messages: list[Any]) -> str:
    """
    从公开消息中派生会话标题。

    Args:
        messages: 快照中的公开消息列表。

    Returns:
        str: 首条用户消息截断为短标题；没有用户消息时返回空字符串。
    """

    for item in messages:
        if isinstance(item, dict) and item.get("role") == "user":
            first_line = str(item.get("content") or "").strip().splitlines()
            text = first_line[0].strip() if first_line else ""
            if text:
                return truncate_safe_text(text, 30)
    return ""


def should_preload_on_startup() -> bool:
    """
    判断启动阶段是否自动预热 RAG。

    Returns:
        bool: `LEGAL_RAG_PRELOAD` 不为 0/false/no/否 时返回 True。
    """

    return os.getenv("LEGAL_RAG_PRELOAD", "1").strip().lower() not in {"0", "false", "no", "否"}


def normalize_agent_event(event: AgentEvent | dict[str, Any]) -> dict[str, Any] | None:
    """
    将内部 AgentEvent 规范化为前端事件。

    Args:
        event: 业务层事件。正常情况下是 AgentEvent；测试或后续扩展也允许传入 dict。

    Returns:
        dict[str, Any] | None: 前端事件；返回 None 表示该内部事件不需要展示给用户。
    """

    if isinstance(event, dict):
        event_type = str(event.get("type", "unknown"))
        raw_data = event.get("data", {})
    else:
        event_type = str(event.type)
        raw_data = event.data

    data = raw_data if isinstance(raw_data, dict) else {"value": raw_data}
    safe_event = sanitize_event_for_web(event_type, data)
    if safe_event is None:
        return None
    safe_type, safe_data = safe_event
    return {
        "type": "event",
        "event_type": safe_type,
        "title": build_event_title(safe_type, safe_data),
        "data": safe_data,
    }


def sanitize_event_for_web(event_type: str, data: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """
    过滤不适合直接展示给用户的内部事件数据。

    Args:
        event_type: 内部事件类型。
        data: 内部事件数据。

    Returns:
        tuple[str, dict[str, Any]] | None: 安全事件类型和数据；None 表示跳过该事件。
    """

    if event_type == "message_done":
        return None
    if event_type == "legal_supplement_required":
        return None
    if event_type == "legal_supplement_skipped":
        # reason 是状态更新器的内部判定文案，可能引用具体案情细节；进度区只需要概括状态。
        return event_type, {"status": "continued"}
    if event_type == "legal_step":
        return event_type, {
            "name": str(data.get("name") or ""),
            "status": str(data.get("status") or ""),
        }
    if event_type == "legal_selfheal":
        # detail 是原始异常摘要，只留在内部事件里；浏览器只需要知道哪个环节发生了什么动作。
        return event_type, {
            "stage": truncate_safe_text(data.get("stage"), 40),
            "action": truncate_safe_text(data.get("action"), 20),
        }
    if event_type == "legal_memory_recalled":
        memories: list[dict[str, Any]] = []
        for item in data.get("memories") or []:
            if len(memories) >= 3:
                break
            if not isinstance(item, dict):
                continue
            title = truncate_safe_text(item.get("title"), 80)
            if not title:
                continue
            memories.append(
                {
                    "title": title,
                    "summary": truncate_safe_text(item.get("summary"), 160),
                    "updated_at": truncate_safe_text(item.get("updated_at"), 40),
                }
            )
        return event_type, {"count": safe_int(data.get("count")) or len(memories), "memories": memories}
    if event_type == "legal_turn_metrics":
        # 只保留数值型观测字段。stages 的 stage/status 是内部固定枚举文案，属于安全文本；
        # 除白名单之外的任何字段（含未来新增的内部诊断字段）一律剥掉，避免顺带透出敏感内容。
        stages: list[dict[str, Any]] = []
        for item in data.get("stages") or []:
            if len(stages) >= 12:
                break
            if not isinstance(item, dict):
                continue
            stage_name = truncate_safe_text(item.get("stage"), 40)
            if not stage_name:
                continue
            stages.append(
                {
                    "stage": stage_name,
                    "duration_ms": safe_int(item.get("duration_ms")),
                    "status": truncate_safe_text(item.get("status"), 20),
                }
            )
        raw_usage = data.get("llm_usage") if isinstance(data.get("llm_usage"), dict) else {}
        return event_type, {
            "stages": stages,
            "total_duration_ms": safe_int(data.get("total_duration_ms")),
            "llm_usage": {
                key: safe_int(raw_usage.get(key))
                for key in ("calls", "input_tokens", "output_tokens", "total_tokens")
            },
            "selfheal_count": safe_int(data.get("selfheal_count")),
        }
    if event_type == "case_state_updated":
        return event_type, {"status": "done", "version": safe_int(data.get("version"))}
    if event_type == "legal_rag_query_started":
        return event_type, {"status": "searching"}
    if event_type == "legal_missing_details_suggested":
        return event_type, {
            "questions": normalize_safe_text_list(data.get("questions"), limit=5),
            "evidence_gaps": normalize_safe_text_list(data.get("evidence_gaps"), limit=5),
            "message": str(data.get("message") or ""),
        }
    if event_type == "legal_case_rag_done":
        return event_type, {"status": "done"}
    if event_type == "legal_risk_analyzed":
        return event_type, {"status": "done", "risk_count": safe_int(data.get("risk_count"))}
    if event_type == "legal_analysis_catalog_built":
        return event_type, {
            "status": "done",
            "follow_up_question_count": len(normalize_safe_text_list(data.get("follow_up_questions"), limit=20)),
            "legal_concept_count": len(normalize_safe_text_list(data.get("legal_concepts"), limit=20)),
        }
    if event_type == "legal_next_action_decided":
        return event_type, {
            "status": "done",
            "action": str(data.get("action") or ""),
            "should_correct_previous_answer": data.get("should_correct_previous_answer") is True,
        }
    if event_type == "legal_web_search_started":
        return event_type, {"status": "searching"}
    if event_type == "legal_web_search_done":
        return event_type, {
            "status": "done",
            "result_count": safe_int(data.get("result_count")),
            "warning_count": safe_int(data.get("warning_count")),
        }
    if event_type == "legal_reference_materials":
        return event_type, {
            "laws": normalize_reference_materials(data.get("laws"), limit=8),
            "web": normalize_reference_materials(data.get("web"), limit=8),
            "warnings": normalize_safe_text_list(data.get("warnings"), limit=5),
        }
    if event_type == "tool_call":
        return event_type, {"status": "running"}
    if event_type == "tool_result":
        return event_type, {"status": "done"}
    # 默认丢弃未知事件。原因是业务事件经常携带内部 query、事实摘要或检索结果，
    # 未显式白名单脱敏前不应透传到浏览器网络流里。
    return None


def normalize_reference_materials(value: Any, *, limit: int) -> list[dict[str, Any]]:
    """
    把内部资料事件规范化为前端可展示的白名单字段。

    Args:
        value: 原始资料列表。
        limit: 最多保留多少条资料。

    Returns:
        list[dict[str, Any]]: 仅包含安全展示字段的资料条目。
    """

    if not isinstance(value, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in value:
        if len(normalized) >= limit:
            break
        if not isinstance(item, dict):
            continue
        title = truncate_safe_text(item.get("title"), 120)
        if not title:
            continue
        normalized.append(
            {
                "id": truncate_safe_text(item.get("id"), 80) or f"material-{len(normalized)}",
                "material_type": truncate_safe_text(item.get("material_type"), 20),
                "title": title,
                "subtitle": truncate_safe_text(item.get("subtitle"), 160),
                "detail": truncate_safe_text(item.get("detail"), 1000),
                "url": truncate_safe_text(item.get("url"), 500),
                "source": truncate_safe_text(item.get("source"), 120),
                "issue": truncate_safe_text(item.get("issue"), 120),
            }
        )
    return normalized


def truncate_safe_text(value: Any, limit: int) -> str:
    """
    把任意值转换为短文本。
    """

    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…"


def normalize_safe_text_list(value: Any, *, limit: int) -> list[str]:
    """
    把内部事件中的列表字段规范化为前端可展示的短文本列表。

    Args:
        value: 原始字段值。
        limit: 最多保留多少项。

    Returns:
        list[str]: 去掉空值后的字符串列表。
    """

    if isinstance(value, list):
        raw_items = value
    elif value:
        raw_items = [value]
    else:
        raw_items = []
    normalized: list[str] = []
    for item in raw_items:
        if len(normalized) >= limit:
            break
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def safe_int(value: Any) -> int:
    """
    安全转换前端计数字段。
    """

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def build_event_title(event_type: str, data: dict[str, Any]) -> str:
    """
    根据事件类型和关键字段生成前端展示标题。

    Args:
        event_type: 内部事件类型。
        data: 内部事件数据。

    Returns:
        str: 简短中文标题。
    """

    if event_type == "legal_step":
        name = data.get("name") or EVENT_TITLES[event_type]
        status = "开始" if data.get("status") == "start" else data.get("status")
        return f"步骤：{status} {name}" if status else f"步骤：{name}"
    if event_type == "legal_selfheal":
        stage = data.get("stage") or "内部环节"
        if data.get("action") == "retried":
            return f"自修复：{stage}已自动重试"
        return f"自修复：{stage}失败，已降级继续"
    if event_type == "legal_memory_recalled":
        return f"唤起历史咨询记忆：{data.get('count', 0)} 条"
    if event_type == "legal_turn_metrics":
        seconds = safe_int(data.get("total_duration_ms")) / 1000
        llm_usage = data.get("llm_usage") if isinstance(data.get("llm_usage"), dict) else {}
        return f"本轮执行指标：耗时 {seconds:.1f} 秒，LLM 调用 {safe_int(llm_usage.get('calls'))} 次"
    if event_type == "legal_rag_query_started":
        return "正在检索本地法条"
    if event_type == "legal_case_rag_done":
        return "本地法条检索完成"
    if event_type == "legal_web_search_started":
        return "正在检索公网案例与司法实践"
    if event_type == "legal_web_search_done":
        return f"公网案例与司法实践检索完成：{data.get('result_count', 0)} 条"
    if event_type == "legal_reference_materials":
        law_count = len(data.get("laws") or [])
        web_count = len(data.get("web") or [])
        return f"参考资料已整理：法条 {law_count} 条，案例/实务 {web_count} 条"
    if event_type == "legal_missing_details_suggested":
        return "可先补充的关键信息"
    if event_type == "legal_supplement_skipped":
        return "无法补充，已按现有信息继续分析"
    if event_type == "legal_risk_analyzed":
        return f"风险识别完成：{data.get('risk_count', 0)} 项"
    if event_type == "legal_next_action_decided":
        action = data.get("action") or ""
        return f"下一步动作：{action}" if action else EVENT_TITLES[event_type]
    if event_type == "tool_call":
        return "正在处理必要工具"
    if event_type == "tool_result":
        return "工具处理完成"
    return EVENT_TITLES.get(event_type, event_type)


def iter_queue_as_ndjson(
    event_queue: queue.Queue[dict[str, Any] | None],
    worker: threading.Thread,
) -> Iterator[str]:
    """
    将线程队列转换为 NDJSON 字符串迭代器。

    Args:
        event_queue: 后台线程写入的事件队列。
        worker: 后台执行法律咨询链路的线程。

    Yields:
        str: JSON 行，每行以换行符结尾。
    """

    while True:
        item = event_queue.get()
        if item is None:
            break
        yield json.dumps(item, ensure_ascii=False, default=str) + "\n"

    # 正常情况下线程已经结束；短暂 join 是为了测试环境更稳定，不在请求线程中长时间等待。
    worker.join(timeout=0.1)


def iter_busy_stream() -> Iterator[str]:
    """
    返回当前会话忙碌时的 NDJSON 错误流。

    Yields:
        str: error 和 done 两个事件。
    """

    yield json.dumps({"type": "error", "message": "当前已有咨询正在处理，请稍后再试。"}, ensure_ascii=False) + "\n"
    yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app.server:app", host="127.0.0.1", port=8000, reload=True)
