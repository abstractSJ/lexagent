"""
法律咨询业务链路对外入口。

该包位于通用 Agent 框架之上，负责多轮案件状态、案情拆解 + 多 query RAG、风险识别、
追问目录和最终答复编排。通用工具调用仍由 agent_system.agent 负责。
"""

from agent_system.legal_consultation.factory import create_legal_consultation_session
from agent_system.legal_consultation.models import (
    LegalAnalysisCatalog,
    LegalArticleEvidence,
    LegalCaseAnalysis,
    LegalCaseRagResult,
    LegalCaseState,
    LegalConsultationTurnResult,
    LegalIssueRagResult,
    LegalNextAction,
    LegalRiskFinding,
    LegalStateUpdate,
)
from agent_system.legal_consultation.session import LegalConsultationSession

__all__ = [
    "LegalAnalysisCatalog",
    "LegalArticleEvidence",
    "LegalCaseAnalysis",
    "LegalCaseRagResult",
    "LegalCaseState",
    "LegalConsultationSession",
    "LegalConsultationTurnResult",
    "LegalIssueRagResult",
    "LegalNextAction",
    "LegalRiskFinding",
    "LegalStateUpdate",
    "create_legal_consultation_session",
]
