"""
法律咨询业务会话工厂。

本模块负责把 LLM、query planner、RAG 检索器、法律工具和业务子任务装配成
LegalConsultationSession。入口脚本只需要调用 create_legal_consultation_session()，不需要关心
内部链路如何组装。
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from agent_system.agent.legal_tools import build_legal_tools
from agent_system.agent.runner import AgentRunner, AgentRunOptions
from agent_system.agent.tools import ToolRegistry
from agent_system.agent.web_search_tools import build_web_search_tools
from agent_system.llm.factory import build_llm_client
from agent_system.planning.legal_query_planner import LegalQueryPlanner
from agent_system.retrieval.legal_retriever import LegalArticleRetriever, build_legal_retriever
from agent_system.legal_consultation.session import (
    DEFAULT_LEGAL_CONSULTATION_SYSTEM_PROMPT,
    LegalConsultationSession,
)
from agent_system.legal_consultation.subtasks import (
    LegalCaseAnalyzer,
    LegalCaseRagSubtask,
    LegalCaseStateUpdater,
    LegalDeterministicWebSearchSubtask,
)


def create_legal_consultation_session(
    *,
    system_prompt: str | None = None,
    answer_options: AgentRunOptions | None = None,
    llm: Any | None = None,
    retriever: LegalArticleRetriever | None = None,
) -> LegalConsultationSession:
    """
    创建默认法律咨询业务会话。

    Args:
        system_prompt: 可选最终回答系统提示词；为空时使用默认法律咨询提示词。
        answer_options: 可选最终回答阶段参数。
        llm: 可选共享 LLM 客户端；为空时新建。多会话场景传入共享实例可避免重复装配。
        retriever: 可选共享法条检索器；为空时新建。BGE-M3 属于重资源，多会话必须共享。

    Returns:
        LegalConsultationSession: 已装配状态更新、RAG、综合分析和最终回答链路的会话。
    """

    # 子任务和最终回答共用一个 LLM 客户端。
    # 原因是当前项目是小型学习项目，复用客户端能减少装配复杂度，也能沿用统一配置。
    llm = llm or build_llm_client()

    # retriever 同时给业务 RAG 子任务和最终 Agent 工具使用。
    # 这样首次加载 BGE-M3 后可以复用同一个检索器实例，避免重复加载模型。
    retriever = retriever or build_legal_retriever()
    planner = LegalQueryPlanner(llm=llm)
    # 公网搜索工具单独构造后再和法律工具合并。
    # 原因是确定性公网检索子任务和最终 AgentRunner 需要共享同一个 ToolRegistry，避免重复注册和配置分叉。
    web_search_tools = build_web_search_tools()
    search_tools = ToolRegistry(build_legal_tools(retriever=retriever) + web_search_tools)
    # 最终回答阶段不再开放检索工具。原因是本轮法条和公网资料已经由确定性业务链路整理到
    # 资料栏；如果继续让最终模型调用工具，工具结果只会进入聊天气泡，反而绕过资料栏协议。
    # 空工具注册表同时让 AgentRunner 可以走流式路径，前端能逐字渲染最终回答。
    answer_runner = AgentRunner(llm=llm, tools=ToolRegistry([]))
    web_search_subtask = LegalDeterministicWebSearchSubtask(tool_runner=search_tools)

    return LegalConsultationSession(
        state_updater=LegalCaseStateUpdater(llm),
        rag_subtask=LegalCaseRagSubtask(planner=planner, retriever=retriever),
        case_analyzer=LegalCaseAnalyzer(llm),
        answer_runner=answer_runner,
        web_search_subtask=web_search_subtask,
        system_prompt=system_prompt or DEFAULT_LEGAL_CONSULTATION_SYSTEM_PROMPT,
        answer_options=answer_options,
    )


def create_legal_consultation_session_factory(
    *,
    system_prompt: str | None = None,
    answer_options: AgentRunOptions | None = None,
) -> Callable[[], LegalConsultationSession]:
    """
    返回可重复创建法律咨询会话的工厂，重资源跨会话共享。

    Why:
        Web 历史会话场景下每个会话对应一个 LegalConsultationSession；LLM 客户端和
        BGE-M3 检索器必须在会话之间共享，否则每恢复一个历史会话就会重复加载一份
        embedding 模型。共享组件按需惰性构建，保证模块导入阶段不会触发任何配置读取
        或模型加载。

    Args:
        system_prompt: 传给每个新会话的系统提示词。
        answer_options: 传给每个新会话的最终回答参数。

    Returns:
        Callable[[], LegalConsultationSession]: 无参工厂，每次调用返回全新会话实例。
    """

    shared: dict[str, Any] = {}
    lock = threading.Lock()

    def factory() -> LegalConsultationSession:
        """
        创建一个共享底层重资源的新法律咨询会话。
        """

        with lock:
            if not shared:
                shared["llm"] = build_llm_client()
                shared["retriever"] = build_legal_retriever()
        return create_legal_consultation_session(
            system_prompt=system_prompt,
            answer_options=answer_options,
            llm=shared["llm"],
            retriever=shared["retriever"],
        )

    return factory


__all__ = ["create_legal_consultation_session", "create_legal_consultation_session_factory"]
