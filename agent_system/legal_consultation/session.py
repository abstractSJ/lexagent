"""
法律咨询业务会话。

LegalConsultationSession 是法律咨询专用的轻量编排层：它负责每轮执行案件状态更新、
案情拆解 + 多 query RAG、综合分析（风险 + 目录 + 下一步动作）和最终回答生成。

它和通用 AgentSession 的边界不同：本类保存法律业务状态和公开对话历史，但不会把内部
LLM 子任务的原始 prompt/response 写入公开 messages。最终回答仍复用现有 AgentRunner，
这样法律检索工具调用能力可以继续沿用当前 Agent 框架。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
import json
import re
import time
from typing import Any, Callable, Protocol

from agent_system.agent.events import AgentEvent
from agent_system.agent.runner import AgentRunOptions
from agent_system.config import LLMCallOptions
from agent_system.llm.messages import Message, assistant_message, system_message, user_message
from agent_system.legal_consultation.models import (
    LegalAnalysisCatalog,
    LegalCaseAnalysis,
    LegalCaseRagResult,
    LegalCaseState,
    LegalConsultationTurnResult,
    LegalNextAction,
    LegalReferenceMaterial,
    LegalReferenceMaterials,
    LegalStateUpdate,
    LegalWebSearchResearchResult,
)
from agent_system.planning.legal_query_planner import LegalQueryPlan
from agent_system.legal_consultation.subtasks import (
    WEB_SEARCH_PURPOSE_LABELS,
    LegalCaseAnalyzer,
    LegalCaseRagSubtask,
    LegalCaseStateUpdater,
    compact_rag_for_prompt,
    compact_web_research_for_prompt,
    merge_state_with_analysis,
    query_plan_to_prompt_dict,
    truncate_text,
)


# 风险矩阵（刑事/民事/行政三层）比旧版“结论 + 理由”结构长，1400 容易在行政层被截断；
# 提示词本身要求高信息密度，放宽到 2000 只兜住复杂案情，不会鼓励啰嗦。
FINAL_ANSWER_OPTIONS = AgentRunOptions(
    initial_options=LLMCallOptions(temperature=0.2, reasoning_effort="medium", max_tokens=2000),
    tool_result_options=LLMCallOptions(temperature=0.1, reasoning_effort="medium", max_tokens=2000),
)

DEFAULT_LEGAL_CONSULTATION_SYSTEM_PROMPT = """
你是一个中文法律咨询 Agent，以执业律师向当事人出口头意见的方式，基于本地法条库和结构化案件状态提供一般信息参考。

表达纪律：
1. 像律师，不像普法宣传：直接定性、点明利害，每句话都要有具体所指；不说教、不车轱辘话，不写“风险较高，请谨慎”“建议咨询专业律师”这类没有信息量的空话。
2. 回答围绕三件事组织：当前局面的法律定性、分刑事/民事/行政（或程序）层面的具体法律风险、接下来最关键的动作。
3. 刑事风险要落到可能罪名、对应法定刑档和触发条件（行为方式、数额门槛、情节）；民事落到返还、赔偿、行为被撤销/无效等具体后果；行政或程序落到罚款、资格限制、限制高消费、配合义务等。
4. 罪名和法定刑档可基于通用法律知识给出，但不做具体个案刑期、罚金或赔偿数额预测，只给法定区间和决定档位的关键情节。

事实与资料纪律：
5. 以最新案件状态为准；如果用户后续补充事实推翻前文，应主动修正之前的阶段性判断。
6. 法条、相似案例和司法实践资料用于内部判断，详细内容放在资料侧栏，不要在最终回答里长篇铺开。
7. 法条依据和公网资料由前置业务链路提供；最终回答阶段不要再尝试自行检索或要求调用工具。
8. 公网资料必须和正式法条依据区分；不得把搜索摘要当作正式法条原文。
9. 不得编造条号、条文原文、案例标题或来源链接；条号和条文原文只有运行时上下文提供时才可引用。对“法院一般怎么判”“赔多少钱”“执行怎么操作”这类实务问题，上下文资料不足时明说资料不足，不要编造。
10. 如果关键事实不足，先给目前能判断的部分，再点出会改变定性或风险等级的关键追问。
11. 结尾必须提示：以下内容仅作一般信息参考，不构成正式法律意见。
""".strip()


class SupportsWebSearchSubtask(Protocol):
    """
    支持确定性公网检索子任务的最小协议。
    """

    def run(
        self,
        *,
        user_input: str,
        state: LegalCaseState,
        rag: Any,
        risks: list[Any] | None,
        catalog: LegalAnalysisCatalog | None,
        next_action: LegalNextAction | None,
    ) -> LegalWebSearchResearchResult:
        """
        执行公网案例与司法实践检索。
        """


class SupportsRun(Protocol):
    """
    支持 AgentRunner.run() 的最小协议。

    这样测试可以传入 fake runner，而不需要构造真实 Responses API 客户端和工具注册表。
    """

    def run(
        self,
        messages: list[Message],
        *,
        options: AgentRunOptions | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> tuple[str, list[AgentEvent]]:
        """
        执行一轮最终回答生成。

        Args:
            messages: 本轮运行时消息。
            options: 可选分阶段 LLM 参数。
            on_delta: 可选文本增量回调，支持流式输出时逐段调用。
        """


class LegalConsultationSession:
    """
    法律咨询多轮业务会话。

    Args:
        state_updater: 案件状态更新子任务。
        rag_subtask: 案情拆解 + 多 query RAG 子任务。
        case_analyzer: 风险识别、案情目录和下一步动作的合并分析子任务。
        answer_runner: 复用现有 AgentRunner 的最终回答生成器。
        system_prompt: 公开主会话的系统提示词。
        answer_options: 最终回答阶段的 AgentRunOptions。
    """

    def __init__(
        self,
        *,
        state_updater: LegalCaseStateUpdater,
        rag_subtask: LegalCaseRagSubtask,
        case_analyzer: LegalCaseAnalyzer,
        answer_runner: SupportsRun,
        web_search_subtask: SupportsWebSearchSubtask | None = None,
        system_prompt: str | None = None,
        answer_options: AgentRunOptions | None = None,
        usage_source: Any = None,
    ) -> None:
        self.state_updater = state_updater
        self.rag_subtask = rag_subtask
        self.case_analyzer = case_analyzer
        self.answer_runner = answer_runner
        self.web_search_subtask = web_search_subtask
        self.answer_options = answer_options or FINAL_ANSWER_OPTIONS
        # usage_source 是轮级 metrics 读取 LLM usage 累计的对象。默认取状态更新器的 llm：
        # 工厂装配时全链路共享同一个客户端实例，从任一子任务拿到的都是同一份累计值。
        self.usage_source = usage_source if usage_source is not None else getattr(state_updater, "llm", None)
        self.case_state = LegalCaseState()
        self.public_messages: list[Message] = []

        prompt = system_prompt or DEFAULT_LEGAL_CONSULTATION_SYSTEM_PROMPT
        if prompt:
            self.public_messages.append(system_message(prompt))

    def preload_resources(self) -> None:
        """
        预加载法律咨询链路中的本地资源。

        当前主要预热本地 RAG：BGE-M3、Chroma collection 校验和关键词索引。
        这样做的原因是把首次加载成本放到 CLI 启动阶段，避免用户输入后终端长时间无反馈。
        """

        self.rag_subtask.preload_resources()

    def ask(self, text: str) -> str:
        """
        执行一轮法律咨询并只返回最终答复。

        Args:
            text: 用户原始输入。

        Returns:
            str: 最终法律咨询答复。
        """

        answer, _ = self.ask_with_events(text)
        return answer

    def ask_with_events(
        self,
        text: str,
        *,
        on_event: Callable[[AgentEvent], None] | None = None,
        recalled_memories: list[dict[str, Any]] | None = None,
        allow_pause: bool = True,
    ) -> tuple[str, list[AgentEvent]]:
        """
        执行一轮完整法律咨询链路。

        Args:
            text: 用户原始输入。
            on_event: 可选实时事件回调。CLI 可用它在长耗时步骤开始时立即打印进度。
            recalled_memories: 可选跨会话历史记忆（白名单字段字典列表，通常由 Web 层
                从 MemoryStore 检索后传入）。记忆只注入内部 prompt 和事件，不进入公开 history。
            allow_pause: 是否允许本轮因阻塞性信息缺失而暂停等待补充。用户明确表示无法补充时，
                Web 层会传 False 强制走完整链路：状态更新器的暂停判定被忽略，缺失信息只作为
                非阻塞追问保留，最终回答基于现有信息给出阶段性意见。

        Returns:
            tuple[str, list[AgentEvent]]: 最终答复和过程事件。

        Raises:
            ValueError: 用户输入为空时抛出。
            Exception: 任一子任务或最终回答失败时向上抛出，并回滚状态和公开历史。
        """

        user_input = text.strip()
        if not user_input:
            raise ValueError("用户输入不能为空。")

        old_state = self.case_state
        old_messages = [dict(message) for message in self.public_messages]
        events: list[AgentEvent] = []
        # 轮级可观测性：记录整轮和各阶段耗时，以及本轮 LLM usage 增量（轮前后快照做差）。
        turn_started_at = time.perf_counter()
        stage_metrics: list[dict[str, Any]] = []
        usage_before = snapshot_llm_usage(self.usage_source)
        # 历史记忆先做一遍清洗：记忆是辅助信号，脏数据（空条目、非字典）直接丢弃，
        # 绝不能让它打断正常咨询链路。
        memories = sanitize_recalled_memories(recalled_memories)

        try:
            if memories:
                # 记忆唤起事件放在链路最前面：让用户第一时间看到“结合了哪些历史咨询”，
                # 也保证后续状态更新的输入口径与事件展示一致。
                record_event(events, on_event, build_memory_recalled_event(memories))
            record_event(events, on_event, build_step_event("案件状态更新", "start"))
            stage_started_at = time.perf_counter()
            stage_status = "ok"
            try:
                state_update = self.state_updater.update(
                    previous_state=old_state,
                    public_messages=self.public_messages,
                    user_input=user_input,
                    recalled_memories=memories or None,
                )
            except Exception as error:
                # 状态更新失败（含子任务内部重试耗尽）不让整轮咨询报废：降级为沿用既有
                # 案件状态。后续 RAG 与最终回答仍然拿得到用户原始输入，可给出可用的阶段性
                # 意见；降级轮自然不触发暂停补充逻辑。
                stage_status = "degraded"
                state_update = LegalStateUpdate(
                    state=old_state,
                    warnings=[f"案件状态更新失败，本轮沿用既有案件状态继续分析：{truncate_text(str(error), 160)}"],
                )
                record_event(
                    events,
                    on_event,
                    build_selfheal_event(stage="案件状态更新", action="degraded", detail=str(error)),
                )
            stage_metrics.append(build_stage_metric("案件状态更新", stage_started_at, stage_status))
            record_event(events, on_event, build_state_event(state_update))
            missing_details_event = build_missing_details_event(state_update)
            if missing_details_event is not None:
                record_event(events, on_event, missing_details_event)

            if state_update.should_pause_for_supplement:
                if allow_pause:
                    supplement_message = build_supplement_prompt_message(state_update)
                    record_event(events, on_event, build_supplement_required_event(state_update, supplement_message))
                    record_event(
                        events,
                        on_event,
                        build_turn_metrics_event(
                            stages=stage_metrics,
                            turn_started_at=turn_started_at,
                            usage_before=usage_before,
                            usage_source=self.usage_source,
                            events=events,
                        ),
                    )
                    # 暂停补充是一次成功轮次。原因是用户已经给出新事实，状态更新也已完成；
                    # 只是后续 RAG/风险/最终回答需要等用户补齐阻塞性信息后再执行。
                    self.case_state = state_update.state
                    self.public_messages.append(user_message(user_input))
                    self.public_messages.append(assistant_message(supplement_message))
                    return supplement_message, events
                # 用户明确表示无法补充时不能把流程卡死：忽略暂停判定继续走完整链路。
                # 缺失问题仍保留在 state 的 follow_up_questions/evidence_gaps 里，
                # 最终回答的“还缺哪些关键信息”章节会自然覆盖它们。
                record_event(events, on_event, build_supplement_skipped_event(state_update))

            record_event(events, on_event, AgentEvent(type="legal_web_search_started", data={"status": "searching"}))
            # 公网检索的 query 素材主要来自用户输入和结构化案件状态，不强依赖 RAG 命中结果。
            # 因此案件状态更新一完成就后台启动，让第三方搜索的等待时间和本地 RAG、综合分析
            # 全程重叠；这是本链路耗时最长的可并行段。
            web_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="legal-web-subtask")
            web_timing_sink: dict[str, int] = {}
            web_future = web_executor.submit(
                self._run_web_search_subtask,
                user_input=user_input,
                state=state_update.state,
                rag=None,
                risks=None,
                catalog=None,
                next_action=None,
                timing_sink=web_timing_sink,
            )
            try:
                record_event(events, on_event, build_step_event("案情拆解 + 多 query RAG", "start"))
                stage_started_at = time.perf_counter()
                stage_status = "ok"
                try:
                    rag = self.rag_subtask.run(
                        case_text=user_input,
                        state=state_update.state,
                        on_event=lambda event: record_event(events, on_event, event),
                    )
                except Exception as error:
                    # 规划或检索失败降级为空法条证据：综合分析与最终回答改用通用法律知识
                    # 和公网资料继续；prompt 纪律会约束模型对资料不足处明说“需核实”。
                    stage_status = "degraded"
                    rag = build_degraded_rag_result(error)
                    record_event(
                        events,
                        on_event,
                        build_selfheal_event(stage="案情拆解 + 多 query RAG", action="degraded", detail=str(error)),
                    )
                stage_metrics.append(build_stage_metric("案情拆解 + 多 query RAG", stage_started_at, stage_status))
                record_event(
                    events,
                    on_event,
                    AgentEvent(
                        type="legal_case_rag_done",
                        data={
                            "issue_count": len(rag.issue_results),
                            "evidence_count": len(rag.evidences),
                            "issues": [item.issue for item in rag.issue_results],
                            "warnings": rag.warnings[:8],
                        },
                    ),
                )

                record_event(events, on_event, build_step_event("案情综合分析", "start"))
                # 风险识别、案情目录和下一步动作共享同一份输入，由综合分析器一次 LLM 调用产出。
                # 事件仍按三个独立步骤发出，保持 CLI 和 Web 前端的展示协议不变。
                stage_started_at = time.perf_counter()
                stage_status = "ok"
                try:
                    analysis = self.case_analyzer.analyze(state=state_update.state, rag=rag)
                except Exception as error:
                    # 综合分析失败降级为空风险 + 默认追问动作：追问比缺分析支撑的结论更安全，
                    # 最终回答仍能基于案件状态和已检索资料给出阶段性意见。
                    stage_status = "degraded"
                    analysis = LegalCaseAnalysis(
                        warnings=[f"案情综合分析失败，已降级为空风险与默认追问动作：{truncate_text(str(error), 160)}"],
                    )
                    record_event(
                        events,
                        on_event,
                        build_selfheal_event(stage="案情综合分析", action="degraded", detail=str(error)),
                    )
                stage_metrics.append(build_stage_metric("案情综合分析", stage_started_at, stage_status))
                risks = analysis.risks
                catalog = analysis.catalog
                next_action = analysis.next_action
                record_event(
                    events,
                    on_event,
                    AgentEvent(
                        type="legal_risk_analyzed",
                        data={
                            "risk_count": len(risks),
                            "risks": [asdict(item) for item in risks[:8]],
                        },
                    ),
                )
                record_event(
                    events,
                    on_event,
                    AgentEvent(
                        type="legal_analysis_catalog_built",
                        data=asdict(catalog),
                    ),
                )
                record_event(
                    events,
                    on_event,
                    AgentEvent(
                        type="legal_next_action_decided",
                        data=asdict(next_action),
                    ),
                )

                committed_state = merge_state_with_analysis(
                    state=state_update.state,
                    risks=risks,
                    catalog=catalog,
                )
                web_research = web_future.result()
            except Exception:
                web_future.cancel()
                web_executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                web_executor.shutdown(wait=True)

            record_event(
                events,
                on_event,
                AgentEvent(
                    type="legal_web_search_done",
                    data={
                        "status": "done",
                        "result_count": count_web_research_results(web_research),
                        "warning_count": len(web_research.warnings),
                    },
                ),
            )
            # 公网检索与本地链路并行，耗时由子任务线程自行测量后经 sink 带回；
            # 失败已在子任务内降级为 warning，阶段状态始终按 ok 记录。
            stage_metrics.append(
                {
                    "stage": "公网案例与司法实践检索",
                    "duration_ms": max(0, int(web_timing_sink.get("duration_ms") or 0)),
                    "status": "ok",
                }
            )
            reference_materials = build_reference_materials(rag=rag, web_research=web_research)
            record_event(events, on_event, build_reference_materials_event(reference_materials))

            runtime_messages = [dict(message) for message in self.public_messages]
            runtime_messages.append(
                user_message(
                    build_runtime_agent_input(
                        user_input=user_input,
                        state=committed_state,
                        rag=rag,
                        risks=risks,
                        catalog=catalog,
                        next_action=next_action,
                        web_research=web_research,
                        recalled_memories=memories,
                    )
                )
            )
            record_event(events, on_event, build_step_event("最终回答生成", "start"))
            # 流式增量只推给实时回调，不写入本轮 events 列表。原因是 events 面向回放和测试，
            # 成百上千个 delta 会把事件列表撑爆；完整最终文本仍由 message_done 事件承载。
            on_delta: Callable[[str], None] | None = None
            sanitizer: StreamingAnswerSanitizer | None = None
            if on_event is not None:
                sanitizer = StreamingAnswerSanitizer()

                def push_answer_delta(delta: str) -> None:
                    """
                    把模型原始增量过滤成安全增量后推送给实时回调。
                    """

                    safe_delta = sanitizer.push(delta)
                    if safe_delta:
                        on_event(AgentEvent(type="answer_delta", data={"delta": safe_delta}))

                on_delta = push_answer_delta

            streamed_successfully = False
            stage_started_at = time.perf_counter()
            final_answer_status = "ok"
            try:
                answer, agent_events = self.answer_runner.run(
                    runtime_messages,
                    options=self.answer_options,
                    on_delta=on_delta,
                )
                streamed_successfully = on_delta is not None
            except Exception as error:
                # 最终回答是唯一没有降级产物的环节：失败后原样自动重试一次。重试改为非流式，
                # 原因是首次尝试可能已推送过部分增量，再次流式会在聊天气泡里叠加两份文本；
                # 非流式重试的完整答案由 final 事件一次性替换气泡。重试再失败则向上抛出，
                # 走既有的整轮回滚。
                final_answer_status = "retried"
                record_event(
                    events,
                    on_event,
                    build_selfheal_event(stage="最终回答生成", action="retried", detail=str(error)),
                )
                answer, agent_events = self.answer_runner.run(
                    runtime_messages,
                    options=self.answer_options,
                    on_delta=None,
                )
            stage_metrics.append(build_stage_metric("最终回答生成", stage_started_at, final_answer_status))
            if streamed_successfully and sanitizer is not None and on_event is not None:
                tail_delta = sanitizer.flush()
                if tail_delta:
                    on_event(AgentEvent(type="answer_delta", data={"delta": tail_delta}))
            answer = strip_reference_sections_from_answer(answer)
            for agent_event in agent_events:
                # 最终回答事件也要同步使用瘦身后的文本。原因是 CLI 或调试入口可能直接读取
                # events；如果只改返回值，事件里仍会残留长篇法条/案例内容。
                if agent_event.type == "message_done" and isinstance(agent_event.data, dict):
                    agent_event = AgentEvent(type=agent_event.type, data={**agent_event.data, "text": answer})
                record_event(events, on_event, agent_event)

            record_event(
                events,
                on_event,
                build_turn_metrics_event(
                    stages=stage_metrics,
                    turn_started_at=turn_started_at,
                    usage_before=usage_before,
                    usage_source=self.usage_source,
                    events=events,
                ),
            )

            self.case_state = committed_state
            self.public_messages.append(user_message(user_input))
            self.public_messages.append(assistant_message(answer))
            return answer, events
        except Exception as error:
            # 任何中间步骤失败都回滚。
            # 原因是没有最终 assistant 回复的一轮不应污染公开主会话，也不应提交半成品案件状态。
            record_event(
                events,
                on_event,
                AgentEvent(
                    type="error",
                    data={"error": str(error)},
                ),
            )
            self.case_state = old_state
            self.public_messages = old_messages
            raise

    def _run_web_search_subtask(
        self,
        *,
        user_input: str,
        state: LegalCaseState,
        rag: Any,
        risks: list[Any] | None,
        catalog: LegalAnalysisCatalog | None,
        next_action: LegalNextAction | None,
        timing_sink: dict[str, int] | None = None,
    ) -> LegalWebSearchResearchResult:
        """
        执行确定性公网检索，并把异常转为 warnings。

        这样做的原因是公网搜索属于补充材料，不应因为搜索配额、网络或第三方接口问题中断
        本地法条分析和最终阶段性答复。

        Args:
            timing_sink: 可选耗时回传字典。公网检索在独立线程运行，由本方法自测耗时写入
                sink，主线程等待结束后读取；这样 metrics 记录的是真实工作耗时而非等待耗时。
        """

        started_at = time.perf_counter()
        try:
            if self.web_search_subtask is None:
                return LegalWebSearchResearchResult(warnings=["未配置公网案例与司法实践检索子任务。"])
            try:
                return self.web_search_subtask.run(
                    user_input=user_input,
                    state=state,
                    rag=rag,
                    risks=risks,
                    catalog=catalog,
                    next_action=next_action,
                )
            except Exception as error:
                return LegalWebSearchResearchResult(warnings=[f"公网案例与司法实践检索失败：{error}"])
        finally:
            if timing_sink is not None:
                timing_sink["duration_ms"] = elapsed_ms(started_at)

    def ask_with_result(self, text: str) -> LegalConsultationTurnResult:
        """
        执行一轮法律咨询并返回完整业务结果。

        当前 CLI 只需要 ask_with_events；这个方法主要给后续调试或测试使用。
        """

        answer, events = self.ask_with_events(text)
        # 这里不重复保存内部子任务结果，原因是 ask_with_events 已经提交状态并返回事件。
        # 若后续需要完整对象，可在 ask_with_events 内部重构返回路径。
        return LegalConsultationTurnResult(
            answer=answer,
            state_update=LegalStateUpdate(state=self.case_state),
            rag=events_to_empty_rag_placeholder(),
            risks=[],
            catalog=LegalAnalysisCatalog(),
            next_action=LegalNextAction(),
            events=events,
        )

    def history(self) -> list[Message]:
        """
        返回公开主会话历史。

        Returns:
            list[Message]: 只包含 system/user/assistant，不包含内部子任务 prompt、JSON 输出或 RAG 原始过程。
        """

        return [dict(message) for message in self.public_messages]

    def export_snapshot(self) -> dict[str, Any]:
        """
        导出可 JSON 序列化的会话快照。

        Returns:
            dict[str, Any]: 包含公开消息和案件状态的最小可恢复状态。持久化层可以在此基础上
            附加资料侧栏等展示性字段；本方法只负责会话自身的真相状态。
        """

        return {
            "messages": self.history(),
            "case_state": asdict(self.case_state),
        }

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        """
        从快照恢复公开历史和案件状态。

        Args:
            snapshot: `export_snapshot()` 产出的（或磁盘加载的）快照字典。

        Why:
            恢复时逐条过滤 role 和未知字段，而不是直接反序列化整个结构。这样旧版本快照多出
            或缺少字段时仍能恢复出合法状态，不会因为 schema 漂移让历史会话整体不可用。
        """

        raw_messages = snapshot.get("messages")
        if isinstance(raw_messages, list):
            restored: list[Message] = []
            for item in raw_messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "")
                content = item.get("content")
                if role not in {"system", "user", "assistant"} or content is None:
                    continue
                restored.append({"role": role, "content": str(content)})
            if restored:
                # 快照缺 system 时保留当前会话的 system prompt，避免恢复后模型失去角色约束。
                if restored[0].get("role") != "system" and self.public_messages and self.public_messages[0].get("role") == "system":
                    restored.insert(0, dict(self.public_messages[0]))
                self.public_messages = restored

        raw_state = snapshot.get("case_state")
        if isinstance(raw_state, dict):
            valid_fields = {item.name for item in fields(LegalCaseState)}
            kwargs = {key: value for key, value in raw_state.items() if key in valid_fields}
            try:
                self.case_state = LegalCaseState(**kwargs)
            except TypeError:
                # 字段类型完全对不上时保持当前状态，宁可丢结构化案件状态也不让恢复失败。
                pass

    def clear(self, keep_system: bool = True) -> None:
        """
        清空公开历史和案件状态。

        Args:
            keep_system: 是否保留 system prompt。
        """

        self.case_state = LegalCaseState()
        if not keep_system:
            self.public_messages.clear()
            return
        if self.public_messages and self.public_messages[0].get("role") == "system":
            self.public_messages[:] = [self.public_messages[0]]
        else:
            self.public_messages.clear()


# 资料侧栏的来源权威度展示映射。normal 不加标识，避免给一般站点贴“可信”标签。
WEB_AUTHORITY_BADGES = {
    "high": "权威来源",
    "medium": "专业来源",
    "low": "低置信来源",
}
# 资料侧栏展示顺序：权威来源在前，低置信度垫底；normal 和未知层级居中。
WEB_AUTHORITY_DISPLAY_ORDER = {
    "high": 0,
    "medium": 1,
    "normal": 2,
    "low": 3,
}


def build_reference_materials(
    *,
    rag: Any,
    web_research: LegalWebSearchResearchResult | None,
    max_laws: int = 8,
    max_web: int = 8,
    detail_limit: int = 500,
) -> LegalReferenceMaterials:
    """
    构造供 Web 资料侧栏展示的安全资料列表。

    Args:
        rag: 本轮本地法条 RAG 结果。
        web_research: 本轮公网案例与司法实践检索结果。
        max_laws: 最多保留多少条法条资料。
        max_web: 最多保留多少条公网资料。
        detail_limit: 单条详情最大长度。

    Returns:
        LegalReferenceMaterials: 已按前端展示需要精简后的资料集合。
    """

    laws: list[LegalReferenceMaterial] = []
    for evidence in getattr(rag, "evidences", []):
        if len(laws) >= max_laws:
            break
        title = evidence.citation or " ".join(item for item in [evidence.legal_name, evidence.article_no] if item)
        if not title:
            continue
        laws.append(
            LegalReferenceMaterial(
                id=f"law-{len(laws)}",
                material_type="law",
                title=title,
                subtitle=safe_material_text(evidence.issue or evidence.legal_name, 160),
                detail=safe_material_text(evidence.text, detail_limit),
                source="本地法条库",
                issue=safe_material_text(evidence.issue, 120),
            )
        )

    web: list[LegalReferenceMaterial] = []
    seen_web_keys: set[str] = set()
    if web_research is not None:
        # 先按 query 顺序摊平，再按权威度稳定排序。这样资料栏优先展示法院/政府和专业法律平台
        # 来源，低置信度站点即使被兜底保留也只会排在末尾。
        flattened: list[tuple[Any, Any]] = [
            (query_result, item)
            for query_result in web_research.query_results
            for item in query_result.results
        ]
        flattened.sort(key=lambda pair: WEB_AUTHORITY_DISPLAY_ORDER.get(pair[1].authority_level, 2))
        for query_result, item in flattened:
            if len(web) >= max_web:
                break
            key = item.url or item.title
            if not key or key in seen_web_keys:
                continue
            seen_web_keys.add(key)
            detail = item.summary or item.snippet
            source_label = item.site_name or item.display_url
            badge = WEB_AUTHORITY_BADGES.get(item.authority_level, "")
            subtitle = f"{badge} · {source_label}" if badge and source_label else (badge or source_label)
            purpose_label = WEB_SEARCH_PURPOSE_LABELS.get(query_result.purpose, query_result.purpose)
            web.append(
                LegalReferenceMaterial(
                    id=f"web-{len(web)}",
                    material_type="web",
                    title=safe_material_text(item.title or item.display_url or item.url, 120),
                    subtitle=safe_material_text(subtitle, 160),
                    detail=safe_material_text(detail, detail_limit),
                    url=safe_http_url(item.url),
                    source=safe_material_text(source_label, 120),
                    issue=safe_material_text(purpose_label, 120),
                )
            )

    warnings = [safe_material_warning(item) for item in getattr(rag, "warnings", [])[:3]]
    if web_research is not None:
        warnings.extend(safe_material_warning(item) for item in web_research.warnings[:5])
    return LegalReferenceMaterials(laws=laws, web=web, warnings=[item for item in warnings[:8] if item])


def safe_material_text(value: Any, limit: int) -> str:
    """
    生成资料侧栏展示文本，并做轻量隐私脱敏。

    侧栏是给用户点击查看的公开 UI，不应暴露手机号、身份证号等可直接识别个人的信息。
    """

    text = truncate_text(str(value or "").strip(), limit)
    text = re.sub(r"1[3-9]\d{9}", "[手机号已隐藏]", text)
    text = re.sub(r"\d{17}[0-9Xx]", "[身份证号已隐藏]", text)
    return text


def safe_material_warning(value: Any) -> str:
    """
    生成资料侧栏 warning，避免泄漏内部检索 query。
    """

    text = str(value or "").strip()
    if not text:
        return ""
    if "检索" in text or "query" in text.lower() or "关键词" in text:
        return "部分资料检索未成功，已展示当前可用资料。"
    return safe_material_text(text, 160)


def safe_http_url(value: Any) -> str:
    """
    只允许前端展示 http(s) 来源链接。
    """

    url = str(value or "").strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return ""
    return truncate_text(url, 500)


def build_reference_materials_event(materials: LegalReferenceMaterials) -> AgentEvent:
    """
    构造资料侧栏事件。

    Args:
        materials: 本轮已整理的安全资料集合。

    Returns:
        AgentEvent: Web 层会再次白名单过滤后发给前端。
    """

    return AgentEvent(
        type="legal_reference_materials",
        data={
            "laws": [asdict(item) for item in materials.laws],
            "web": [asdict(item) for item in materials.web],
            "warnings": materials.warnings[:8],
        },
    )


REFERENCE_SECTION_KEYWORDS = (
    "法律依据",
    "法条依据",
    "相关法条",
    "适用法条",
    "法条全文",
    "公网案例",
    "相似案例",
    "参考案例",
    "案例和实务",
    "案例 / 实务",
    "司法实践",
    "裁判规则",
    "实务参考",
    "参考资料",
    "来源链接",
)

# 聊天气泡应保留的核心回答章节标题关键词。与最终回答提示词中的固定结构对应：
# 结论 / 当前关键点 / 法律风险（含刑事、民事、行政子层）/ 现在该做什么 / 还缺哪些关键信息。
# 旧结构关键词（为什么、下一步等）保留，兼容历史会话回放和模型偶发的旧习惯输出。
CORE_ANSWER_SECTION_KEYWORDS = (
    "结论",
    "关键点",
    "法律风险",
    "刑事风险",
    "民事风险",
    "行政风险",
    "为什么",
    "怎么判断",
    "你现在",
    "该做什么",
    "下一步",
    "还缺",
    "关键信息",
)


def strip_reference_sections_from_answer(answer: str) -> str:
    """
    从最终聊天答复中移除长篇依据资料章节。

    Args:
        answer: 模型生成的原始最终答复。

    Returns:
        str: 适合放进聊天气泡的短答复。

    Why:
        资料侧栏已经承载法条、案例和来源链接。最终模型即使被提示约束，仍可能按旧习惯输出
        “法律依据”“公网案例”等章节；这里做一道轻量兜底，避免用户再次在对话框里看到长篇资料。
    """

    lines = str(answer or "").splitlines()
    kept: list[str] = []
    skipping_reference_section = False

    for line in lines:
        stripped = line.strip()
        starts_reference_section = is_reference_section_heading(stripped)
        starts_answer_section = is_answer_section_heading(stripped) or starts_reference_section

        if skipping_reference_section:
            # 引用资料章节内常见“1. 案例标题”这类编号列表；它不是新的回答章节，不能用来结束跳过。
            # 只有明确的新核心回答标题才恢复保留，避免案例条目漏回聊天气泡。
            if starts_answer_section and not starts_reference_section and is_core_answer_section_heading(stripped):
                skipping_reference_section = False
            else:
                continue

        if starts_reference_section:
            skipping_reference_section = True
            continue

        # 来源链接应进入资料栏，不放在聊天气泡里。这样做会牺牲极少数普通链接展示，但能防止
        # 搜索结果 URL 长串重新污染最终回答。
        if "http://" in stripped or "https://" in stripped:
            continue
        kept.append(line)

    cleaned = collapse_blank_lines("\n".join(kept)).strip()
    return cleaned or build_reference_only_answer_notice()


class StreamingAnswerSanitizer:
    """
    最终回答流式增量的逐行安全过滤器。

    Why:
        流式输出如果原样透传，模型偶尔按旧习惯生成的“法律依据”“公网案例”章节和来源链接
        会先闪现在聊天气泡里，等 final 事件替换后又消失，观感割裂。本过滤器按行复用
        strip_reference_sections_from_answer 的判定规则：一行只有在确认安全后才作为增量放行，
        引用资料章节和 URL 行在流式阶段就被拦下。代价是增量按“整行”粒度推送——必须看到
        行尾才能判断该行是否是资料标题；对以短行为主的中文咨询答复，这个粒度足够流畅。
    """

    def __init__(self) -> None:
        self._pending_line = ""
        self._skipping_reference_section = False
        # 初始视为“上一行是空行”，可以顺带吞掉答案开头的空行。
        self._last_emitted_blank = True

    def push(self, delta: str) -> str:
        """
        接收一段原始增量，返回当前可安全推送的增量文本。

        Args:
            delta: 模型原始文本片段。

        Returns:
            str: 已通过过滤的增量；本次没有完整安全行时返回空字符串。
        """

        self._pending_line += str(delta or "")
        if "\n" not in self._pending_line:
            return ""

        lines = self._pending_line.split("\n")
        self._pending_line = lines.pop()
        emitted: list[str] = []
        for line in lines:
            kept_line = self._filter_line(line)
            if kept_line is not None:
                emitted.append(kept_line + "\n")
        return "".join(emitted)

    def flush(self) -> str:
        """
        流结束后处理最后一段没有换行符的残余文本。

        Returns:
            str: 最后一行通过过滤后的增量；被过滤掉时返回空字符串。
        """

        if not self._pending_line:
            return ""
        line = self._pending_line
        self._pending_line = ""
        kept_line = self._filter_line(line)
        return kept_line if kept_line is not None else ""

    def _filter_line(self, line: str) -> str | None:
        """
        对单个完整行应用与最终清洗一致的保留/跳过规则。

        Args:
            line: 不含换行符的完整行。

        Returns:
            str | None: 应保留时返回原行；应丢弃时返回 None。
        """

        stripped = line.strip()
        starts_reference_section = is_reference_section_heading(stripped)

        if self._skipping_reference_section:
            if not starts_reference_section and is_core_answer_section_heading(stripped):
                self._skipping_reference_section = False
            else:
                return None

        if starts_reference_section:
            self._skipping_reference_section = True
            return None

        if "http://" in stripped or "https://" in stripped:
            return None

        is_blank = not stripped
        if is_blank and self._last_emitted_blank:
            # 与 collapse_blank_lines 对齐：连续空行只保留一个，避免删掉资料章节后出现大段空白。
            return None
        self._last_emitted_blank = is_blank
        return line


def is_answer_section_heading(text: str) -> bool:
    """
    判断一行是否像 Markdown/中文编号章节标题。
    """

    if not text:
        return False
    if re.match(r"^#{1,6}\s*\S", text):
        return True
    if re.match(r"^\*\*[^*]{1,80}\*\*$", text):
        return True
    if is_core_answer_section_heading(text) or is_reference_section_heading(text):
        return True
    return bool(re.match(r"^(?:[一二三四五六七八九十]+、|\d+[.、]|[（(][一二三四五六七八九十\d]+[）)])\s*\S.{0,80}$", text))


def normalize_section_heading_text(text: str) -> str:
    """
    提取一行文本的章节标题正文，供关键词匹配使用。

    去掉 Markdown 井号、加粗星号、中英文编号等标题装饰，以及结尾的冒号。
    """

    return re.sub(r"^[#\s*（(\d.、一二三四五六七八九十）)]+", "", text).strip("* ：:")


def looks_like_section_heading(normalized: str) -> bool:
    """
    判断规范化后的文本是否具备章节标题的形态：足够短且不含句内标点。

    Why:
        资料章节关键词（如“司法实践”“参考资料”）同样会出现在正文句子里。如果任何包含
        关键词的行都按资料章节开头处理，律师风格答复中“民事：司法实践中管理人可要求返还”
        这类风险正文会连同后续内容被整段误删。真实章节标题都很短、没有逗号句号冒号，
        先用形态过滤能把正文句子和标题区分开。
    """

    if not normalized or len(normalized) > 24:
        return False
    return not re.search(r"[，。；：！？,.;:!?]", normalized)


def is_reference_section_heading(text: str) -> bool:
    """
    判断章节标题是否属于应迁移到资料栏的依据资料类内容。
    """

    normalized = normalize_section_heading_text(text)
    return looks_like_section_heading(normalized) and any(
        keyword in normalized for keyword in REFERENCE_SECTION_KEYWORDS
    )


def is_core_answer_section_heading(text: str) -> bool:
    """
    判断章节标题是否是聊天气泡应保留的核心回答段落。
    """

    normalized = normalize_section_heading_text(text)
    return looks_like_section_heading(normalized) and any(
        keyword in normalized for keyword in CORE_ANSWER_SECTION_KEYWORDS
    )


def collapse_blank_lines(text: str) -> str:
    """
    折叠连续空行，避免删除资料章节后留下大块空白。
    """

    return re.sub(r"\n{3,}", "\n\n", text)


def build_reference_only_answer_notice() -> str:
    """
    生成资料型回答被整体移出聊天气泡后的兜底提示。

    Returns:
        str: 可直接保存到公开聊天历史的短提示。
    """

    return "详细法条和案例已整理到右侧参考资料栏，请点击标题展开查看。\n\n以下内容仅作一般信息参考，不构成正式法律意见。"


def build_runtime_agent_input(
    *,
    user_input: str,
    state: LegalCaseState,
    rag: Any,
    risks: list[Any],
    catalog: LegalAnalysisCatalog,
    next_action: LegalNextAction,
    web_research: LegalWebSearchResearchResult | None = None,
    recalled_memories: list[dict[str, Any]] | None = None,
) -> str:
    """
    构造给最终 AgentRunner 的本轮临时增强输入。

    该输入会参与最终回答生成，但成功后不会作为用户原文写入公开 history。
    """

    memory_block = ""
    if recalled_memories:
        # 历史记忆区块只在确有召回时插入，保持无记忆场景的 prompt 与旧版本完全一致。
        memory_block = f"""
【历史咨询记忆】
以下是该用户此前其他咨询会话沉淀的背景记忆，仅供理解上下文。历史记忆不是本案事实和依据，
不得据此编造本案未提供的事实；如与本案明显相关，可在回答中用一句话自然呼应既往咨询。
{json.dumps(recalled_memories, ensure_ascii=False, indent=2)}
"""

    return f"""
请基于以下运行时上下文回答用户。本段是内部结构化上下文，不要向用户暴露“内部子任务”字样。

【用户原始输入】
{user_input}

【最新案件状态】
{json.dumps(asdict(state), ensure_ascii=False, indent=2)}
{memory_block}
【案情拆解与检索计划】
{json.dumps(query_plan_to_prompt_dict(rag.query_plan), ensure_ascii=False, indent=2)}

【已检索法条证据】
{json.dumps(compact_rag_for_prompt(rag, max_items=8), ensure_ascii=False, indent=2)}

【公网案例与司法实践检索】
{json.dumps(compact_web_research_for_prompt(web_research), ensure_ascii=False, indent=2)}

【不利事实、矛盾和证据缺口】
{json.dumps([asdict(item) for item in risks], ensure_ascii=False, indent=2)}

【案情要点、法律概念和追问目录】
{json.dumps(asdict(catalog), ensure_ascii=False, indent=2)}

【下一步动作判断】
{json.dumps(asdict(next_action), ensure_ascii=False, indent=2)}

【回答要求】
1. 用执业律师给当事人出口头意见的口吻回答：直接、克制、每句话有具体所指；不说教、不重复，最终回答只保留最重要要点。
2. 回答结构和配额固定如下，标题名照用：
   - 「结论」：1~3 句，直接给当前局面的法律定性和最可能的走向，不加铺垫。
   - 「当前关键点」：最多 4 条，只列决定定性的事实、时间节点和争议点，每条一句话。
   - 「法律风险」：按“刑事／民事／行政（或程序）”分层。刑事写明可能罪名、对应法定刑档和触发条件（行为方式、数额门槛、情节）；民事写返还、赔偿、行为被撤销/无效等具体后果；行政或程序写罚款、资格限制、限制高消费、配合调查义务等。某一层没有现实风险就整层省略，不要硬凑。
   - 「现在该做什么」：最多 4 条，按先后顺序，只写具体且非显而易见的动作；“咨询律师”这类常识不单独成条。
   - 「还缺哪些关键信息」：最多 3 个，只保留会改变定性或风险等级的问题；没有就省略整节。
3. 罪名和法定刑档可基于通用法律知识给出（如“处三年以下有期徒刑或者拘役；数额巨大的，处三年以上七年以下有期徒刑”），并点出决定档位的关键情节；不得预测具体个案刑期、罚金或赔偿数额；条号和条文原文只有本上下文提供时才可引用。
4. 详细法条、相似案例和实务资料已经整理到右侧参考资料栏，回答中最多用一句话提示用户查看；不要输出“法律依据”“相关法条”“公网案例”“司法实践参考”等资料章节，不要逐条罗列全部法条，不要复制法条全文；如确需点依据，只写极短引用，例如“主要依据：《企业破产法》第二十五条”。
5. 不要长篇列相似案例、裁判规则、公网搜索结果或来源链接；公网资料只提炼对当前判断最关键的一句话，并与正式法条依据区分。公网资料的 authority_level 表示来源可信度：high 是法院/检察院/政府官方站点，medium 是专业法律数据库或权威媒体，low 是低置信度网站；概括司法实践口径时优先采信 high/medium 来源，low 来源只作背景线索，不得单独作为实务判断依据。
6. 以最新案件状态为准；如果 should_correct_previous_answer 为 true，开头先用一句话说明新事实改变了此前判断。
7. 如果 action 是 ask_followup，先给当前能确定的结论和风险，再列关键追问。
8. 不得编造未检索到的条文内容、案例或链接；上下文资料不足以支撑某个判断时明说“需核实”，不要硬答。
9. 结尾必须提示：以下内容仅作一般信息参考，不构成正式法律意见。
""".strip()


# 单轮注入的历史记忆上限。跨会话记忆是背景信号而不是本案证据，条数放开只会稀释
# 状态更新和最终回答对本案事实的注意力。
MAX_RECALLED_MEMORIES = 3


def sanitize_recalled_memories(value: Any) -> list[dict[str, Any]]:
    """
    清洗外部传入的历史记忆列表。

    Args:
        value: 调用方传入的原始记忆负载，期望是白名单字段字典列表。

    Returns:
        list[dict[str, Any]]: 只保留合法条目和白名单字段的记忆列表；无有效内容时返回空列表。

    Why:
        记忆来自磁盘文件且经由 Web 层透传，字段可能缺失或被手工改坏。会话层自己再做一次
        逐字段清洗，保证注入 prompt 的结构永远可控，也让直接调用会话 API 的测试和脚本
        不依赖上游是否规范。
    """

    if not isinstance(value, list):
        return []

    memories: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or "").strip()
        if not title and not summary:
            continue
        memories.append(
            {
                "title": truncate_text(title, 60),
                "summary": truncate_text(summary, 200),
                "key_facts": normalize_memory_text_list(item.get("key_facts"), max_items=5),
                "user_goals": normalize_memory_text_list(item.get("user_goals"), max_items=3),
                "legal_concepts": normalize_memory_text_list(item.get("legal_concepts"), max_items=6),
                "updated_at": str(item.get("updated_at") or "").strip(),
            }
        )
        if len(memories) >= MAX_RECALLED_MEMORIES:
            break
    return memories


def normalize_memory_text_list(value: Any, *, max_items: int) -> list[str]:
    """
    规范化记忆条目里的字符串列表字段。
    """

    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        normalized.append(truncate_text(text, 80))
        if len(normalized) >= max_items:
            break
    return normalized


def build_memory_recalled_event(memories: list[dict[str, Any]]) -> AgentEvent:
    """
    构造历史记忆唤起事件。

    事件面向前端进度区展示，只携带标题、短摘要和更新时间；完整记忆内容只进内部 prompt。
    """

    return AgentEvent(
        type="legal_memory_recalled",
        data={
            "count": len(memories),
            "memories": [
                {
                    "title": item["title"],
                    "summary": truncate_text(item["summary"], 160),
                    "updated_at": item["updated_at"],
                }
                for item in memories
            ],
        },
    )


# 轮级 metrics 中 llm_usage 的固定键，与 OpenAIChatClient.usage_totals 保持一致。
LLM_USAGE_KEYS = ("calls", "input_tokens", "output_tokens", "total_tokens")


def elapsed_ms(started_at: float) -> int:
    """
    计算从 started_at（perf_counter 时刻）到现在的毫秒数，负值截断为 0。
    """

    return max(0, int((time.perf_counter() - started_at) * 1000))


def build_stage_metric(stage: str, started_at: float, status: str) -> dict[str, Any]:
    """
    构造单个阶段的耗时指标。

    Args:
        stage: 阶段名，与 legal_step 步骤名保持一致。
        started_at: 阶段开始的 perf_counter 时刻。
        status: ok（正常）、degraded（降级继续）或 retried（自动重试后成功）。
    """

    return {"stage": stage, "duration_ms": elapsed_ms(started_at), "status": status}


def snapshot_llm_usage(source: Any) -> dict[str, int] | None:
    """
    从 usage 来源对象读取累计 usage 快照。

    Args:
        source: 通常是共享的 OpenAIChatClient。兼容两种形态：提供 snapshot_usage_totals()
            方法，或直接暴露 usage_totals 字典；两者都没有（如部分测试 fake）时返回 None，
            metrics 中的 usage 按零值处理。
    """

    if source is None:
        return None
    snapshot = getattr(source, "snapshot_usage_totals", None)
    data = snapshot() if callable(snapshot) else getattr(source, "usage_totals", None)
    if not isinstance(data, dict):
        return None
    result: dict[str, int] = {}
    for key in LLM_USAGE_KEYS:
        try:
            result[key] = max(0, int(data.get(key) or 0))
        except (TypeError, ValueError):
            result[key] = 0
    return result


def diff_llm_usage(before: dict[str, int] | None, after: dict[str, int] | None) -> dict[str, int]:
    """
    计算轮前后两次 usage 快照的增量；无法取到快照时返回零值。
    """

    if after is None:
        return {key: 0 for key in LLM_USAGE_KEYS}
    before = before or {}
    return {key: max(0, int(after.get(key) or 0) - int(before.get(key) or 0)) for key in LLM_USAGE_KEYS}


def build_turn_metrics_event(
    *,
    stages: list[dict[str, Any]],
    turn_started_at: float,
    usage_before: dict[str, int] | None,
    usage_source: Any,
    events: list[AgentEvent],
) -> AgentEvent:
    """
    构造轮级可观测性事件。

    Args:
        stages: 各阶段耗时与状态列表。
        turn_started_at: 整轮开始的 perf_counter 时刻。
        usage_before: 轮开始时的 usage 快照。
        usage_source: usage 来源对象，用于取轮结束快照做差。
        events: 本轮已记录事件，用于统计 selfheal 次数。

    Why:
        metrics 作为普通 AgentEvent 发出而不是单独返回值：CLI、Web 和测试都已经消费事件流，
        观测数据走同一条通道就能同时覆盖实时展示、NDJSON 推送和 events.jsonl 持久化。
    """

    return AgentEvent(
        type="legal_turn_metrics",
        data={
            "stages": [dict(item) for item in stages],
            "total_duration_ms": elapsed_ms(turn_started_at),
            "llm_usage": diff_llm_usage(usage_before, snapshot_llm_usage(usage_source)),
            "selfheal_count": sum(1 for event in events if event.type == "legal_selfheal"),
        },
    )


def build_selfheal_event(*, stage: str, action: str, detail: str) -> AgentEvent:
    """
    构造链路自修复事件。

    Args:
        stage: 发生自修复的环节名，与 legal_step 的步骤名保持一致。
        action: retried（该环节已自动重试）或 degraded（该环节失败，以降级产物继续）。
        detail: 原始错误摘要。只进入内部事件列表供 CLI 和测试排查；Web 白名单会剥掉它，
            避免把异常内文透传到浏览器。
    """

    return AgentEvent(
        type="legal_selfheal",
        data={
            "stage": stage,
            "action": action,
            "detail": truncate_text(str(detail or ""), 300),
        },
    )


def build_degraded_rag_result(error: Exception | None = None) -> LegalCaseRagResult:
    """
    构造检索失败时的空降级 RAG 结果。

    Args:
        error: 触发降级的异常；为 None 时返回纯空占位（供兼容路径复用）。

    Returns:
        LegalCaseRagResult: 空计划、空证据的合法结果对象，可直接进入后续分析与回答链路。
    """

    warnings: list[str] = []
    if error is not None:
        warnings.append(f"案情拆解与法条检索失败，本轮无本地法条证据：{truncate_text(str(error), 160)}")
    return LegalCaseRagResult(
        query_plan=LegalQueryPlan(global_queries=[], issues=[], warnings=[]),
        issue_results=[],
        evidences=[],
        warnings=warnings,
    )


def record_event(
    events: list[AgentEvent],
    on_event: Callable[[AgentEvent], None] | None,
    event: AgentEvent,
) -> None:
    """
    记录事件并立即推送给可选回调。

    Args:
        events: 本轮最终返回的事件列表。
        on_event: 实时事件回调；为空时只记录不推送。
        event: 当前事件。
    """

    events.append(event)
    if on_event is not None:
        on_event(event)


def build_step_event(name: str, status: str) -> AgentEvent:
    """
    构造 workflow 步骤进度事件。

    Args:
        name: 步骤名称。
        status: 步骤状态，当前主要使用 start。

    Returns:
        AgentEvent: 进度事件。
    """

    return AgentEvent(type="legal_step", data={"name": name, "status": status})


def build_state_event(state_update: LegalStateUpdate) -> AgentEvent:
    """
    构造案件状态更新事件。
    """

    return AgentEvent(
        type="case_state_updated",
        data={
            "version": state_update.state.version,
            "summary": state_update.state.summary,
            "newly_added_facts": state_update.newly_added_facts,
            "changed_facts": state_update.changed_facts,
            "warnings": state_update.warnings,
        },
    )


def build_missing_details_event(state_update: LegalStateUpdate) -> AgentEvent | None:
    """
    构造可提前展示给用户的关键信息补充事件。

    Args:
        state_update: 本轮案件状态更新结果。

    Returns:
        AgentEvent | None: 有追问或证据缺口时返回展示事件，否则返回 None。
    """

    questions = (state_update.supplement_questions or state_update.state.follow_up_questions)[:5]
    evidence_gaps = (state_update.supplement_evidence_gaps or state_update.state.evidence_gaps)[:5]
    if not questions and not evidence_gaps:
        return None
    if state_update.should_pause_for_supplement:
        message = "这些信息会显著影响判断，建议先补充后再继续分析。"
    else:
        message = "可以先准备这些关键信息；后台会继续检索相关法条并生成阶段性答复。"
    return AgentEvent(
        type="legal_missing_details_suggested",
        data={
            "questions": questions,
            "evidence_gaps": evidence_gaps,
            "message": message,
        },
    )


def build_supplement_prompt_message(state_update: LegalStateUpdate) -> str:
    """
    构造暂停补充时返回给用户的助手消息。

    Args:
        state_update: 本轮案件状态更新结果。

    Returns:
        str: 用户可直接阅读的补充提示。
    """

    questions = (state_update.supplement_questions or state_update.state.follow_up_questions)[:5]
    evidence_gaps = (state_update.supplement_evidence_gaps or state_update.state.evidence_gaps)[:5]
    lines = [
        "为了避免在关键信息不足时给出不可靠判断，请先补充以下信息。",
    ]
    if state_update.pause_reason:
        lines.extend(["", f"暂停原因：{state_update.pause_reason}"])
    if questions:
        lines.extend(["", "需要确认的问题："])
        lines.extend(f"{index}. {question}" for index, question in enumerate(questions, start=1))
    if evidence_gaps:
        lines.extend(["", "建议准备或说明的证据/材料："])
        lines.extend(f"{index}. {gap}" for index, gap in enumerate(evidence_gaps, start=1))
    lines.extend(["", "请在补充后继续发送，我会再检索法条、案例和实务资料并生成分析。"])
    return "\n".join(lines)


def build_supplement_required_event(state_update: LegalStateUpdate, message: str) -> AgentEvent:
    """
    构造暂停补充控制事件。

    Args:
        state_update: 本轮案件状态更新结果。
        message: 已构造好的用户提示文本。

    Returns:
        AgentEvent: Web 层会把该事件转换为 top-level pause 流事件。
    """

    return AgentEvent(
        type="legal_supplement_required",
        data={
            "reason": state_update.pause_reason,
            "questions": (state_update.supplement_questions or state_update.state.follow_up_questions)[:5],
            "evidence_gaps": (state_update.supplement_evidence_gaps or state_update.state.evidence_gaps)[:5],
            "message": message,
            "state_version": state_update.state.version,
        },
    )


def build_supplement_skipped_event(state_update: LegalStateUpdate) -> AgentEvent:
    """
    构造“跳过补充、按现有信息继续”事件。

    Args:
        state_update: 本轮案件状态更新结果（其暂停判定被 allow_pause=False 覆盖）。

    Returns:
        AgentEvent: 提示前端本轮已忽略暂停判定继续分析；未决问题仍通过
        legal_missing_details_suggested 事件展示，这里只携带概括状态。
    """

    return AgentEvent(
        type="legal_supplement_skipped",
        data={
            "status": "continued",
            "reason": state_update.pause_reason,
        },
    )


def count_web_research_results(web_research: LegalWebSearchResearchResult) -> int:
    """
    统计公网检索结果条目数。
    """

    return sum(len(item.results) for item in web_research.query_results)


def events_to_empty_rag_placeholder() -> Any:
    """
    为 ask_with_result 的临时兼容返回空占位。

    该方法避免把完整中间结果长期挂在 session 上。若后续确实需要完整结果，应调整主流程返回值，
    而不是让 session 保存上一轮所有内部对象。
    """

    return build_degraded_rag_result()


__all__ = [
    "DEFAULT_LEGAL_CONSULTATION_SYSTEM_PROMPT",
    "FINAL_ANSWER_OPTIONS",
    "LegalConsultationSession",
    "StreamingAnswerSanitizer",
    "build_runtime_agent_input",
]
