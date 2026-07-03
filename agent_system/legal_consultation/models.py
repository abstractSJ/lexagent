"""
法律咨询业务模型。

本模块只保存多轮法律咨询链路需要的结构化状态和子任务结果，不保存任何内部 LLM
子调用的原始 prompt 或 response。这样做的原因是：公开主会话应该只承载用户原话和最终
助手回复，内部分析过程作为业务状态和事件存在，避免污染后续对话上下文。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_system.agent.events import AgentEvent
from agent_system.planning.legal_query_planner import LegalQueryPlan


NEXT_ACTION_ANSWER_NOW = "answer_now"
NEXT_ACTION_ASK_FOLLOWUP = "ask_followup"
NEXT_ACTION_SEARCH_MORE = "search_more"
NEXT_ACTION_CANNOT_ANSWER = "cannot_answer"
VALID_NEXT_ACTIONS = {
    NEXT_ACTION_ANSWER_NOW,
    NEXT_ACTION_ASK_FOLLOWUP,
    NEXT_ACTION_SEARCH_MORE,
    NEXT_ACTION_CANNOT_ANSWER,
}


@dataclass(frozen=True)
class LegalCaseState:
    """
    多轮法律咨询案件状态。

    Attributes:
        summary: 当前案件的中性摘要，只描述事实和诉求，不下最终法律结论。
        parties: 当事人或相关主体，例如劳动者、公司、出借人、借款人。
        timeline: 已知时间线事实，用于识别时效、期限、先后顺序等问题。
        confirmed_facts: 当前较明确的案件事实。
        disputed_facts: 存在争议、前后不一致或尚未确认的事实。
        adverse_facts: 可能对用户主张不利的事实。
        contradictions: 用户陈述中存在的前后矛盾或需要澄清的冲突。
        evidence_gaps: 可能影响判断的证据缺口。
        user_goals: 用户想解决的问题或目标。
        legal_concepts: 当前可能涉及的法律概念，使用“可能涉及”层级，不直接下结论。
        follow_up_questions: 后续最值得追问的问题。
        version: 状态版本号，每成功处理一轮用户输入后递增。
    """

    summary: str = ""
    parties: list[str] = field(default_factory=list)
    timeline: list[str] = field(default_factory=list)
    confirmed_facts: list[str] = field(default_factory=list)
    disputed_facts: list[str] = field(default_factory=list)
    adverse_facts: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)
    user_goals: list[str] = field(default_factory=list)
    legal_concepts: list[str] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    version: int = 0


@dataclass(frozen=True)
class LegalStateUpdate:
    """
    单轮用户输入后的案件状态更新结果。

    Attributes:
        state: 更新后的完整案件状态。
        newly_added_facts: 本轮新增的事实。
        changed_facts: 本轮修正、推翻或显著改变前文理解的事实。
        warnings: 状态更新中的非致命提示，例如事实冲突或字段缺失。
        should_pause_for_supplement: 是否需要先暂停后续链路，要求用户补充阻塞性关键信息。
        pause_reason: 暂停原因，用于向用户解释为什么不能直接继续分析。
        supplement_questions: 需要用户优先回答的问题。
        supplement_evidence_gaps: 需要用户优先补充或确认的证据材料。
    """

    state: LegalCaseState
    newly_added_facts: list[str] = field(default_factory=list)
    changed_facts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    should_pause_for_supplement: bool = False
    pause_reason: str = ""
    supplement_questions: list[str] = field(default_factory=list)
    supplement_evidence_gaps: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LegalArticleEvidence:
    """
    多 query RAG 后去重得到的法条证据。

    Attributes:
        citation: 可展示的引用格式，例如“《劳动合同法》第八十二条”。
        legal_name: 法律名称。
        article_no: 条号。
        text: 法条正文。
        issue: 该法条最初关联的法律事项。
        source_query: 命中该法条的 query 或关键词。
        retrieval_type: 召回类型，通常为 semantic 或 keyword。
        score: 检索分数；关键词命中或未知时可以为 None。
        hit_count: 多 query 聚合后该法条被命中的次数，用于简单重排。
    """

    citation: str
    legal_name: str
    article_no: str
    text: str
    issue: str
    source_query: str
    retrieval_type: str
    score: float | None = None
    hit_count: int = 1


@dataclass(frozen=True)
class LegalIssueRagResult:
    """
    单个法律事项的多 query 检索结果。

    Attributes:
        issue: 法律事项标题。
        facts: 与该事项相关的案件事实。
        used_queries: 实际执行过的语义检索 query。
        used_keywords: 实际执行过的关键词兜底词。
        evidences: 去重后的法条证据。
        warnings: 该事项检索过程中的非致命提示。
    """

    issue: str
    facts: list[str]
    used_queries: list[str]
    used_keywords: list[str]
    evidences: list[LegalArticleEvidence]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LegalCaseRagResult:
    """
    “案情拆解 + 多重 query RAG”子任务总结果。

    Attributes:
        query_plan: 复用现有 LegalQueryPlanner 得到的检索计划。
        issue_results: 每个 issue 的检索结果。
        evidences: 全案合并去重后的法条证据。
        warnings: 规划和检索过程中的非致命提示。
    """

    query_plan: LegalQueryPlan
    issue_results: list[LegalIssueRagResult]
    evidences: list[LegalArticleEvidence]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LegalWebSearchItem:
    """
    公网案例与司法实践检索结果条目。

    Attributes:
        title: 网页标题。
        url: 结果链接，用于最终回答给出来源。
        snippet: 搜索引擎返回的简短摘录。
        summary: 搜索工具生成的摘要；只能作为公网资料摘要使用，不能等同正式法条原文。
        site_name: 来源站点名称。
        display_url: 适合展示的短链接。
        date_published: 页面发布时间，可能为空。
        date_last_crawled: 搜索工具抓取时间，可能为空。
        authority_level: 来源域名权威度分级，取 high（法院/检察院/政府等官方站点）、
            medium（专业法律数据库或权威媒体）、normal（一般站点）、low（低置信度内容农场）。
            该分级只依据域名判断，不采信页面内容的自我声明，用于结果重排和最终回答的采信提示。
    """

    title: str
    url: str
    snippet: str = ""
    summary: str = ""
    site_name: str = ""
    display_url: str = ""
    date_published: str = ""
    date_last_crawled: str = ""
    authority_level: str = "normal"


@dataclass(frozen=True)
class LegalWebSearchQueryResult:
    """
    单条确定性公网检索 query 的结果。

    Attributes:
        purpose: 检索目的，例如 similar_cases、judicial_interpretation 或 judicial_practice。
        query: 实际执行的公网检索 query。该字段只进入内部 prompt，不直接暴露给 Web 进度区。
        ok: 当前 query 是否执行成功。
        results: 已按 URL 去重并精简字段的结果条目。
        error: 当前 query 的非致命错误信息。
    """

    purpose: str
    query: str
    ok: bool
    results: list[LegalWebSearchItem] = field(default_factory=list)
    error: str = ""


@dataclass(frozen=True)
class LegalWebSearchResearchResult:
    """
    每轮法律咨询的确定性公网案例与司法实践检索汇总。

    Attributes:
        query_results: 按固定顺序执行的公网检索结果。
        warnings: 工具失败、返回异常或结果为空时产生的非致命告警。
    """

    query_results: list[LegalWebSearchQueryResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LegalReferenceMaterial:
    """
    可展示在 Web 资料侧栏的安全资料条目。

    Attributes:
        id: 前端列表 key 和展开状态使用的稳定标识。
        material_type: 资料类型，law 表示本地法条，web 表示公网案例或实务资料。
        title: 侧栏默认展示标题。
        subtitle: 标题下方的简短来源或关联事项。
        detail: 用户点击展开后看到的正文片段或摘要。
        url: 公网资料来源链接；本地法条通常为空。
        source: 资料来源名称。
        issue: 该资料关联的法律事项。
    """

    id: str
    material_type: str
    title: str
    subtitle: str = ""
    detail: str = ""
    url: str = ""
    source: str = ""
    issue: str = ""


@dataclass(frozen=True)
class LegalReferenceMaterials:
    """
    本轮可展示资料汇总。

    Attributes:
        laws: 本地法条资料。
        web: 公网案例、司法实践或实务资料。
        warnings: 资料整理或公网检索产生的非致命提示。
    """

    laws: list[LegalReferenceMaterial] = field(default_factory=list)
    web: list[LegalReferenceMaterial] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LegalRiskFinding:
    """
    法律咨询中的风险项。

    Attributes:
        type: 风险类型，例如 adverse_fact、contradiction、missing_evidence、legal_uncertainty。
        severity: 风险强度，建议使用 high、medium、low。
        fact: 触发风险的事实、矛盾或证据缺口。
        reason: 为什么该点会影响案件判断。
        suggestion: 下一步如何澄清或补强。
    """

    type: str
    severity: str
    fact: str
    reason: str
    suggestion: str


@dataclass(frozen=True)
class LegalAnalysisCatalog:
    """
    案情要点、法律概念和追问目录。

    Attributes:
        case_points: 当前最重要的案件要点。
        legal_concepts: 可能涉及的法律概念。
        follow_up_questions: 后续追问目录，优先覆盖会改变案件性质的事实。
    """

    case_points: list[str] = field(default_factory=list)
    legal_concepts: list[str] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LegalNextAction:
    """
    当前轮次后的下一步动作判断。

    Attributes:
        action: 下一步动作，取值见 VALID_NEXT_ACTIONS。
        reasons: 选择该动作的原因。
        questions_to_ask: 如果需要追问，优先询问的问题。
        should_correct_previous_answer: 新事实是否足以要求修正前面阶段性判断。
    """

    action: str = NEXT_ACTION_ASK_FOLLOWUP
    reasons: list[str] = field(default_factory=list)
    questions_to_ask: list[str] = field(default_factory=list)
    should_correct_previous_answer: bool = False


@dataclass(frozen=True)
class LegalCaseAnalysis:
    """
    风险识别、案情目录和下一步动作的合并分析结果。

    这三部分共享同一份输入（案件状态 + 法条证据摘要），由一次结构化 LLM 调用同时产出，
    以减少每轮咨询的串行 LLM 往返；对外事件仍按三个独立步骤发出。

    Attributes:
        risks: 不利事实、矛盾和证据缺口等风险项。
        catalog: 案情要点、法律概念和追问目录。
        next_action: 下一步动作判断。
    """

    risks: list[LegalRiskFinding] = field(default_factory=list)
    catalog: LegalAnalysisCatalog = field(default_factory=LegalAnalysisCatalog)
    next_action: LegalNextAction = field(default_factory=LegalNextAction)


@dataclass(frozen=True)
class LegalConsultationTurnResult:
    """
    一轮法律咨询 workflow 的完整结果。

    Attributes:
        answer: 最终返回给用户的答复。
        state_update: 本轮状态更新结果。
        rag: 案情拆解和多 query RAG 结果。
        risks: 不利事实、矛盾和证据缺口等风险项。
        catalog: 案情要点、法律概念和追问目录。
        next_action: 下一步动作判断。
        web_research: 公网案例与司法实践检索结果；暂停补充或旧兼容路径可为空。
        events: 对外展示或调试用事件；不等于主会话 messages。
    """

    answer: str
    state_update: LegalStateUpdate
    rag: LegalCaseRagResult
    risks: list[LegalRiskFinding]
    catalog: LegalAnalysisCatalog
    next_action: LegalNextAction
    web_research: LegalWebSearchResearchResult | None = None
    events: list[AgentEvent] = field(default_factory=list)
