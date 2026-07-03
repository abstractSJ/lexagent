"""
Agent 运行事件定义。

这个模块只定义轻量事件对象，不绑定 OpenAI SDK，也不绑定具体 UI。
这样后续无论是命令行打印、Web UI 展示，还是写入日志文件，都可以复用同一套事件结构。
"""

from dataclasses import dataclass
from typing import Any, Literal


AgentEventType = Literal[
    "message_done",
    "answer_delta",
    "tool_call",
    "tool_result",
    "error",
    "legal_step",
    "legal_rag_query_started",
    "case_state_updated",
    "legal_missing_details_suggested",
    "legal_supplement_required",
    "legal_case_rag_done",
    "legal_risk_analyzed",
    "legal_analysis_catalog_built",
    "legal_next_action_decided",
    "legal_web_search_started",
    "legal_web_search_done",
]


@dataclass(frozen=True)
class AgentEvent:
    """
    Agent 执行过程中的一个事件。

    Args:
        type: 事件类型，例如 tool_call、tool_result、message_done。
        data: 事件数据。使用字典是为了在学习阶段保持扩展灵活，避免过早拆分太多事件类。
    """

    type: AgentEventType
    data: dict[str, Any]
