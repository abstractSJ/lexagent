"""
法律咨询业务子任务。

本模块把多轮案件状态更新、案情拆解 + 多重 query RAG、综合分析（风险识别 + 案情目录 +
下一步动作）拆成可测试的轻量组件。它们是普通 Python 子任务，不是独立 Agent；原因是这些
步骤属于法律咨询的确定性业务链路，应由代码控制顺序和状态提交，避免让主模型自由决定是否
遗漏关键步骤。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
import re
from typing import Any, Callable, Protocol

from agent_system.agent.events import AgentEvent
from agent_system.config import LLMCallOptions
from agent_system.llm.messages import Message, system_message, user_message
from agent_system.planning.legal_query_planner import (
    LegalQueryPlan,
    LegalQueryPlanner,
    parse_json_object,
    plan_to_dict,
)
from agent_system.retrieval.legal_retriever import LegalArticleRetriever
from agent_system.legal_consultation.models import (
    NEXT_ACTION_ASK_FOLLOWUP,
    VALID_NEXT_ACTIONS,
    LegalAnalysisCatalog,
    LegalArticleEvidence,
    LegalCaseAnalysis,
    LegalCaseRagResult,
    LegalCaseState,
    LegalIssueRagResult,
    LegalNextAction,
    LegalRiskFinding,
    LegalStateUpdate,
    LegalWebSearchItem,
    LegalWebSearchQueryResult,
    LegalWebSearchResearchResult,
)


STRUCTURED_SUBTASK_OPTIONS = LLMCallOptions(
    temperature=0.1,
    reasoning_effort="low",
    max_tokens=1600,
)

# 多 query RAG 的第一版规模控制。
# 原因是每轮咨询都会执行该子任务，如果不做上限，复杂案情会把本地检索和最终 prompt 都撑得过大。
DEFAULT_MAX_QUERIES_PER_ISSUE = 3
DEFAULT_QUERY_TOP_K = 8
DEFAULT_KEYWORD_TOP_K = 8
DEFAULT_ISSUE_EVIDENCE_LIMIT = 8
DEFAULT_TOTAL_EVIDENCE_LIMIT = 25
DEFAULT_MAX_LEGAL_NAME_FILTERS = 2
DEFAULT_RETRIEVAL_WORKERS = 4
# 每条公网 query 请求 10 条候选而只保留 5 条。原因是重排需要候选池：
# 搜索引擎前几名经常被 SEO 咨询站占据，多取一倍候选才有机会把官方/专业来源换上来。
DEFAULT_WEB_SEARCH_COUNT = 10
DEFAULT_WEB_SEARCH_MAX_QUERIES = 3
DEFAULT_WEB_SEARCH_RESULT_LIMIT = 5
DEFAULT_WEB_SEARCH_WORKERS = 3

# ---------- 法条证据融合重排参数 ----------
# 语义检索的弱相关命中直接剔除的分数线；每条 query 的前几名保底保留，防止全部低分时证据清零。
MIN_SEMANTIC_EVIDENCE_SCORE = 0.30
SEMANTIC_JOB_MIN_KEEP = 2
# 多 query 重复命中只作为加分项，不再绝对压制单次高分命中。
# 原因是 planner 生成的多条 query 语义相近，重复命中常常只说明“该条文表述宽泛”，
# 让 hit_count 绝对优先会把 0.9 分的精准条文排到两次 0.4 分的泛化条文之后。
EVIDENCE_EXTRA_HIT_BONUS = 0.06
EVIDENCE_EXTRA_HIT_BONUS_CAP = 3
# planner 输出的 positive_terms/negative_terms 用于轻量加权重排，权重必须小于分数主体，
# 只用来在相近分数之间调序，不能反转明显的相关度差异。
EVIDENCE_POSITIVE_TERM_BONUS = 0.02
EVIDENCE_POSITIVE_TERM_BONUS_CAP = 0.06
EVIDENCE_NEGATIVE_TERM_PENALTY = 0.05
EVIDENCE_NEGATIVE_TERM_PENALTY_CAP = 0.15

# ---------- 公网检索来源权威度分级 ----------
# 域名后缀采用整段匹配（等于该后缀或以 ".后缀" 结尾），避免 fakegov.cn 之类的伪装域名蹭权威分。
# high：法院/检察院/政府/人大等官方站点，以及最高法背景的法信平台，覆盖司法解释、指导案例和裁判文书原文。
HIGH_AUTHORITY_DOMAIN_SUFFIXES = (
    "gov.cn",          # 覆盖 court.gov.cn、spp.gov.cn、moj.gov.cn、flk.npc.gov.cn 等全部政务域名
    "chinacourt.org",  # 中国法院网、人民法院报
    "faxin.cn",        # 法信（人民法院出版社）
)
# medium：专业法律数据库和权威媒体，内容通常有编辑校验，可作为司法实践口径参考。
MEDIUM_AUTHORITY_DOMAIN_SUFFIXES = (
    "pkulaw.com",
    "pkulaw.cn",       # 北大法宝
    "wkinfo.com.cn",   # 威科先行
    "itslaw.com",      # 无讼
    "lexiscn.com",     # 律商联讯
    "law-lib.com",     # 法律图书馆
    "legaldaily.com.cn",  # 法治日报
    "xinhuanet.com",   # 新华网
    "people.com.cn",   # 人民网
    "gmw.cn",          # 光明网（法治频道）
    "chinanews.com.cn",  # 中国新闻网
    "chinanews.com",
)
# low：SEO 问答农场、自媒体聚合和文库转载站。这类页面正是“低置信度小文章”的主要来源：
# 内容多为模板化改写，经常过时甚至互相矛盾，只允许在权威结果不足时兜底展示。
LOW_AUTHORITY_DOMAIN_SUFFIXES = (
    "66law.cn",        # 华律网
    "findlaw.cn",      # 找法网
    "64365.com",       # 律图
    "lawtime.cn",      # 法律快车
    "maxlaw.cn",       # 大律师网
    "baijiahao.baidu.com",
    "zhidao.baidu.com",
    "wenku.baidu.com",
    "tieba.baidu.com",
    "jingyan.baidu.com",
    "zhihu.com",
    "sohu.com",
    "163.com",
    "toutiao.com",
    "jianshu.com",
    "csdn.net",
    "360doc.com",
    "docin.com",
    "doc88.com",
    "book118.com",
    "bilibili.com",
    "douyin.com",
    "kuaishou.com",
    "xiaohongshu.com",
)

# 域名权威度的基础分。低置信度设为大幅负分，保证只要有 normal 以上的结果，农场文章就排不进前列。
WEB_AUTHORITY_DOMAIN_SCORES = {
    "high": 6.0,
    "medium": 3.0,
    "normal": 0.0,
    "low": -6.0,
}
# 标题/摘要中出现权威内容特征词（指导案例、司法解释、裁判要旨等）时按词加分。
# 权重低于域名分：内容可以自称“最高法案例解读”，域名伪装不了，所以内容特征只作次级信号。
WEB_AUTHORITY_CONTENT_KEYWORDS = (
    "指导性案例",
    "指导案例",
    "公报案例",
    "典型案例",
    "参考案例",
    "司法解释",
    "裁判要旨",
    "裁判规则",
    "会议纪要",
    "批复",
    "审理指南",
    "理解与适用",
    "量刑指导意见",
    "最高人民法院",
    "最高人民检察院",
    "高级人民法院",
    "裁判文书",
    "判决书",
    "裁定书",
)
WEB_AUTHORITY_KEYWORD_BONUS = 1.0
WEB_AUTHORITY_KEYWORD_BONUS_CAP = 3.0
# 命中规范案号（如“（2023）京01民终1234号”）通常意味着真实裁判文书或案例评析，单独加分。
WEB_CASE_NUMBER_PATTERN = re.compile(r"[（(]\s*(?:19|20)\d{2}\s*[）)]\s*[^（）()，。；\s]{1,18}号")
WEB_CASE_NUMBER_BONUS = 2.0
# 非低置信度结果不足这个数量时，才允许回填低置信度结果，避免资料栏整栏空白。
WEB_LOW_AUTHORITY_BACKFILL_LIMIT = 2

# 检索目的对应的中文标签，用于资料栏展示，避免把内部英文 purpose 直接暴露给用户。
WEB_SEARCH_PURPOSE_LABELS = {
    "similar_cases": "相似案例与裁判文书",
    "judicial_interpretation": "司法解释与权威规定",
    "judicial_practice": "裁判规则与司法实践",
}

# 传给博查 exclude 参数的低质站点列表（服务端过滤，官方上限 100 个域名）。
# 在服务端排除比本地丢结果更划算：count 个候选位不会被内容农场文章占用。
WEB_SEARCH_EXCLUDE_SITES = "|".join(LOW_AUTHORITY_DOMAIN_SUFFIXES)
# 司法解释/权威规定 query 使用 include 限定站点：司法解释原文和“理解与适用”类文章
# 集中在官方站点和专业法律数据库，开放全网检索反而会被解读小文章稀释。
WEB_SEARCH_AUTHORITATIVE_INCLUDE_SITES = "gov.cn|chinacourt.org|faxin.cn|pkulaw.com|pkulaw.cn|wkinfo.com.cn"

# 案件状态里的法律概念常带“可能涉及”前缀；直接进搜索词会稀释关键词，需要剥掉。
LEGAL_CONCEPT_PREFIX_PATTERN = re.compile(r"^(可能涉及|可能构成|可能存在|或涉及|涉嫌|涉及)")
# 案情材料出现赔偿/量刑类诉求时，实务 query 追加对应标准词，把检索引向裁判口径而不是普法文章。
COMPENSATION_HINT_PATTERN = re.compile(r"赔偿|补偿|工资|欠款|利息|违约金|退款|退赔")
SENTENCING_HINT_PATTERN = re.compile(r"量刑|判刑|判几年|刑事责任|拘留|逮捕|取保候审|缓刑")

STATE_UPDATE_SYSTEM_PROMPT = """
你是法律咨询案件状态更新器。

你的任务是根据上一轮案件状态、公开对话摘要和用户本轮输入，更新案件的结构化事实状态。

规则：
1. 只记录事实、诉求、疑点、证据缺口和可能涉及的法律概念，不做最终法律结论。
2. 用户本轮明确纠正前文时，以最新陈述为准，同时把被修正内容写入 changed_facts 或 contradictions。
3. 对用户不利的事实要保留到 adverse_facts，不要因为对用户不利就忽略。
4. 不得编造用户没有提供的事实。
5. 必须判断是否需要先暂停后续检索和回答，要求用户补充阻塞性关键信息。
6. 只有缺失信息会显著改变法律关系、责任基础、时效期限、程序路径、刑事/民事性质或赔偿/量刑/补偿区间时，should_pause_for_supplement 才能为 true。
7. 如果只是有助于增强证据但不影响当前阶段性判断，不要暂停；把问题放入 state.follow_up_questions 或 state.evidence_gaps 即可。
8. pause_reason 应简短说明为什么继续会导致不可靠分析；supplement_questions 和 supplement_evidence_gaps 只列最关键的 3-5 项。
9. 只输出合法 JSON，不要输出 Markdown、解释文字或代码块。
""".strip()

CASE_ANALYSIS_SYSTEM_PROMPT = """
你是法律咨询综合分析器。

你的任务是基于案件状态和已检索到的法条证据，一次性完成三部分结构化分析：
1. risks：识别会影响咨询走向的不利事实、事实矛盾、证据缺口和适用风险。
2. catalog：生成案情要点、可能涉及的法律概念和后续追问目录。
3. next_action：判断当前轮次应该直接给阶段性答复、继续追问、提示需要补检索，还是说明暂时无法判断。

规则：
1. 只做风险提示，不得写“必然败诉”“必然构成犯罪”等确定结论。
2. 每条风险必须绑定具体事实、矛盾或缺失证据。
3. 法律概念使用“可能涉及……”层级，不直接下最终法律结论。
4. 追问目录优先覆盖会改变案件性质、责任大小或程序路径的事实。
5. next_action.action 只能是 answer_now、ask_followup、search_more、cannot_answer 四者之一。
6. 如果关键事实缺失且会改变案件性质，next_action 优先 ask_followup。
7. 如果用户新事实推翻前文判断，should_correct_previous_answer 必须为 true。
8. 不得编造用户没有提供的事实。
9. 只输出合法 JSON，不要输出 Markdown、解释文字或代码块。
""".strip()

# 综合分析一次输出风险、目录和下一步动作三段 JSON，比单个子任务长；
# 单独放宽 max_tokens，避免复杂案情下输出被截断导致 JSON 解析失败。
CASE_ANALYSIS_OPTIONS = LLMCallOptions(
    temperature=0.1,
    reasoning_effort="low",
    max_tokens=2400,
)


class SupportsToolRun(Protocol):
    """
    支持 ToolRegistry.run() 的最小协议。

    这样确定性公网检索子任务可以复用现有 web_search 工具，也方便测试传入 fake runner。
    """

    def run(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        执行指定工具并返回工具结果。
        """


class SupportsComplete(Protocol):
    """
    支持完整文本调用的最小 LLM 协议。

    使用 Protocol 的原因是测试可以传入 fake LLM，不需要初始化真实远程客户端。
    """

    def complete(
        self,
        messages: list[Message],
        *,
        options: LLMCallOptions | None = None,
    ) -> str:
        """
        执行一次非流式完整输出调用。
        """


@dataclass(frozen=True)
class RetrievalJob:
    """
    单个法条检索任务。

    Attributes:
        index: 全局任务顺序号，用于并发完成后恢复确定性合并顺序。
        issue_index: 所属法律事项下标。
        issue_title: 所属法律事项标题。
        retrieval_type: 检索类型，取 semantic 或 keyword。
        query: 实际检索文本；关键词检索时是关键词拼接文本。
        legal_name: 法律名称过滤条件；为空表示不限制法律名称。
        keywords: 关键词兜底检索使用的词列表。
    """

    index: int
    issue_index: int
    issue_title: str
    retrieval_type: str
    query: str
    legal_name: str
    keywords: list[str]


@dataclass(frozen=True)
class RetrievalJobResult:
    """
    单个法条检索任务的执行结果。

    Attributes:
        job: 对应的检索任务。
        evidences: 已转换为统一模型的法条证据。
        warnings: 该任务产生的非致命告警。
    """

    job: RetrievalJob
    evidences: list[LegalArticleEvidence]
    warnings: list[str]


class LegalCaseStateUpdater:
    """
    多轮法律案件状态更新子任务。

    Args:
        llm: 支持 complete() 的 LLM 客户端。
        options: 子任务调用参数。默认使用低温低推理，原因是这里追求 JSON 稳定性。
    """

    def __init__(
        self,
        llm: SupportsComplete,
        *,
        options: LLMCallOptions | None = None,
    ) -> None:
        self.llm = llm
        self.options = options or STRUCTURED_SUBTASK_OPTIONS

    def update(
        self,
        *,
        previous_state: LegalCaseState,
        public_messages: list[Message],
        user_input: str,
    ) -> LegalStateUpdate:
        """
        根据本轮用户输入更新案件状态。

        Args:
            previous_state: 上一轮已提交的案件状态。
            public_messages: 公开主会话历史，只包含 system/user/assistant。
            user_input: 本轮用户原始输入。

        Returns:
            LegalStateUpdate: 更新后的案件状态和变化说明。
        """

        messages = [
            system_message(STATE_UPDATE_SYSTEM_PROMPT),
            user_message(
                build_state_update_user_prompt(
                    previous_state=previous_state,
                    public_messages=public_messages,
                    user_input=user_input,
                )
            ),
        ]
        raw_response = self.llm.complete(messages, options=self.options)
        data = parse_json_object(raw_response)
        state_data = data.get("state") if isinstance(data.get("state"), dict) else data
        state = legal_case_state_from_dict(
            state_data,
            fallback=previous_state,
            version=previous_state.version + 1,
        )
        return LegalStateUpdate(
            state=state,
            newly_added_facts=normalize_string_list(data.get("newly_added_facts"), max_items=20),
            changed_facts=normalize_string_list(data.get("changed_facts"), max_items=20),
            warnings=normalize_string_list(data.get("warnings"), max_items=20),
            should_pause_for_supplement=parse_bool(data.get("should_pause_for_supplement")),
            pause_reason=normalize_text(data.get("pause_reason")),
            supplement_questions=normalize_string_list(data.get("supplement_questions"), max_items=5),
            supplement_evidence_gaps=normalize_string_list(data.get("supplement_evidence_gaps"), max_items=5),
        )


class LegalCaseRagSubtask:
    """
    “案情拆解 + 多重 query RAG”合并子任务。

    Args:
        planner: 已有法律 query planner。
        retriever: 已有法条检索器。
        max_queries_per_issue: 每个 issue 最多执行多少条语义 query。
        query_top_k: 每条语义 query 返回多少候选。
        keyword_top_k: 关键词兜底返回多少候选。
        issue_evidence_limit: 单个 issue 最多保留多少条证据。
        total_evidence_limit: 全案最多保留多少条证据。
        retrieval_workers: 本地检索并发 worker 数；设为 1 时使用串行执行。
    """

    def __init__(
        self,
        *,
        planner: LegalQueryPlanner,
        retriever: LegalArticleRetriever,
        max_queries_per_issue: int = DEFAULT_MAX_QUERIES_PER_ISSUE,
        query_top_k: int = DEFAULT_QUERY_TOP_K,
        keyword_top_k: int = DEFAULT_KEYWORD_TOP_K,
        issue_evidence_limit: int = DEFAULT_ISSUE_EVIDENCE_LIMIT,
        total_evidence_limit: int = DEFAULT_TOTAL_EVIDENCE_LIMIT,
        retrieval_workers: int = DEFAULT_RETRIEVAL_WORKERS,
    ) -> None:
        self.planner = planner
        self.retriever = retriever
        self.max_queries_per_issue = max(1, int(max_queries_per_issue))
        self.query_top_k = max(1, int(query_top_k))
        self.keyword_top_k = max(1, int(keyword_top_k))
        self.issue_evidence_limit = max(1, int(issue_evidence_limit))
        self.total_evidence_limit = max(1, int(total_evidence_limit))
        self.retrieval_workers = max(1, int(retrieval_workers))

    def preload_resources(self) -> None:
        """
        预加载 RAG 子任务依赖的本地检索资源。

        这里只预热 retriever，不调用 query planner。原因是 planner 依赖具体案情，不能在用户输入前执行；
        但 BGE-M3、Chroma 校验和关键词索引可以提前准备。
        """

        self.retriever.preload(include_keyword_index=True)

    def run(
        self,
        *,
        case_text: str,
        state: LegalCaseState,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> LegalCaseRagResult:
        """
        执行案情拆解和多 query 法条检索。

        Args:
            case_text: 本轮用户原始输入。
            state: 最新案件状态。
            on_event: 可选实时事件回调。用于 CLI 在长耗时检索前立即打印进度。

        Returns:
            LegalCaseRagResult: issue 级和全案级检索证据。
        """

        planning_text = build_planner_case_text(case_text=case_text, state=state)
        query_plan = self.planner.plan(planning_text)
        warnings = list(query_plan.warnings)
        jobs, used_queries_by_issue, used_keywords_by_issue = self._build_retrieval_jobs(query_plan)
        if jobs:
            self._emit_retrieval_batch_started(jobs=jobs, on_event=on_event)
            # 并发检索前先完成本地资源预热。原因是 retriever 内部有模型、Chroma 和关键词索引的懒加载状态，
            # 若多个 worker 首次同时触发初始化，容易把一次性加载成本放大甚至造成底层资源竞争。
            self.retriever.preload(include_keyword_index=True)
        job_results = self._run_retrieval_jobs(jobs)
        results_by_issue = group_retrieval_results_by_issue(job_results)

        issue_results: list[LegalIssueRagResult] = []
        all_evidence_map: dict[str, LegalArticleEvidence] = {}
        for issue_index, issue in enumerate(query_plan.issues):
            issue_map: dict[str, LegalArticleEvidence] = {}
            issue_warnings: list[str] = []
            for result in results_by_issue.get(issue_index, []):
                issue_warnings.extend(result.warnings)
                for evidence in result.evidences:
                    merge_evidence(issue_map, evidence)

            issue_evidences = sort_evidences(
                issue_map.values(),
                positive_terms=issue.positive_terms,
                negative_terms=issue.negative_terms,
            )[: self.issue_evidence_limit]
            issue_result = LegalIssueRagResult(
                issue=issue.issue,
                facts=list(issue.facts),
                used_queries=used_queries_by_issue.get(issue_index, []),
                used_keywords=used_keywords_by_issue.get(issue_index, []),
                evidences=issue_evidences,
                warnings=issue_warnings,
            )
            issue_results.append(issue_result)
            warnings.extend(issue_warnings)

            for evidence in issue_evidences:
                merge_evidence(all_evidence_map, evidence)

        # 全案级重排使用全部 issue 的加权词。原因是跨事项证据没有单一归属，
        # 用全量正/反向词做轻量调序比只用第一个事项的词更公平。
        all_positive_terms = [term for issue in query_plan.issues for term in issue.positive_terms]
        all_negative_terms = [term for issue in query_plan.issues for term in issue.negative_terms]
        evidences = sort_evidences(
            all_evidence_map.values(),
            positive_terms=all_positive_terms,
            negative_terms=all_negative_terms,
        )[: self.total_evidence_limit]
        return LegalCaseRagResult(
            query_plan=query_plan,
            issue_results=issue_results,
            evidences=evidences,
            warnings=warnings,
        )

    def _build_retrieval_jobs(
        self,
        query_plan: LegalQueryPlan,
    ) -> tuple[list[RetrievalJob], dict[int, list[str]], dict[int, list[str]]]:
        """
        根据 query plan 构造可并发执行的检索任务。

        Args:
            query_plan: 法律案情检索计划。

        Returns:
            tuple: 检索任务、issue 使用过的语义 query、issue 使用过的关键词。
        """

        jobs: list[RetrievalJob] = []
        used_queries_by_issue: dict[int, list[str]] = {}
        used_keywords_by_issue: dict[int, list[str]] = {}
        for issue_index, issue in enumerate(query_plan.issues):
            legal_names = issue.preferred_legal_names[:DEFAULT_MAX_LEGAL_NAME_FILTERS] or [""]
            issue_queries = list(issue.queries[: self.max_queries_per_issue])
            used_queries_by_issue[issue_index] = issue_queries
            for query in issue_queries:
                for legal_name in legal_names:
                    jobs.append(
                        RetrievalJob(
                            index=len(jobs),
                            issue_index=issue_index,
                            issue_title=issue.issue,
                            retrieval_type="semantic",
                            query=query,
                            legal_name=legal_name,
                            keywords=[],
                        )
                    )

            issue_keywords = list(issue.positive_terms)
            used_keywords_by_issue[issue_index] = issue_keywords
            if issue_keywords:
                jobs.append(
                    RetrievalJob(
                        index=len(jobs),
                        issue_index=issue_index,
                        issue_title=issue.issue,
                        retrieval_type="keyword",
                        query=" ".join(issue_keywords),
                        legal_name=legal_names[0] if legal_names and legal_names[0] else "",
                        keywords=issue_keywords,
                    )
                )
        return jobs, used_queries_by_issue, used_keywords_by_issue

    def _emit_retrieval_batch_started(
        self,
        *,
        jobs: list[RetrievalJob],
        on_event: Callable[[AgentEvent], None] | None,
    ) -> None:
        """
        在主线程发出一次粗粒度检索开始事件。

        这里不从 worker 线程逐条推送事件。原因是 Web 端只需要知道检索已开始，逐条 query 会制造噪音，
        还会让事件回调和本轮 events 列表承受并发写入风险。
        """

        first_job = jobs[0]
        emit_optional_event(
            on_event,
            AgentEvent(
                type="legal_rag_query_started",
                data={
                    "retrieval_type": "batch",
                    "issue": first_job.issue_title,
                    "query": f"并发执行 {len(jobs)} 个检索任务",
                    "legal_name": first_job.legal_name,
                    "job_count": len(jobs),
                },
            ),
        )

    def _run_retrieval_jobs(self, jobs: list[RetrievalJob]) -> list[RetrievalJobResult]:
        """
        执行检索任务并保持结果顺序稳定。

        Args:
            jobs: 待执行的检索任务。

        Returns:
            list[RetrievalJobResult]: 按 job.index 排序后的检索结果。
        """

        if not jobs:
            return []
        if self.retrieval_workers <= 1 or len(jobs) == 1:
            return [self._execute_retrieval_job(job) for job in jobs]

        worker_count = min(self.retrieval_workers, len(jobs))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="legal-rag") as executor:
            # executor.map 会按输入顺序产出结果；这里仍排序一次，防止后续替换执行器时破坏确定性。
            results = list(executor.map(self._execute_retrieval_job, jobs))
        return sorted(results, key=lambda item: item.job.index)

    def _execute_retrieval_job(self, job: RetrievalJob) -> RetrievalJobResult:
        """
        执行单个检索任务。

        Args:
            job: 检索任务。

        Returns:
            RetrievalJobResult: 当前任务的证据和非致命告警。
        """

        if job.retrieval_type == "keyword":
            return self._execute_keyword_job(job)
        return self._execute_semantic_job(job)

    def _execute_semantic_job(self, job: RetrievalJob) -> RetrievalJobResult:
        """
        执行单条语义检索任务。
        """

        warnings: list[str] = []
        evidences: list[LegalArticleEvidence] = []
        try:
            result = self.retriever.search_legal_articles(
                query=job.query,
                top_k=self.query_top_k,
                legal_name=job.legal_name,
                category="",
                article_no="",
                source_type="law_article",
                include_neighbors=False,
            )
        except Exception as error:  # pragma: no cover - 真实检索环境异常需要进入 warnings，而不是吞掉调试信息。
            return RetrievalJobResult(job=job, evidences=[], warnings=[f"语义检索失败：{job.query}；原因：{error}"])

        if not result.get("ok"):
            warnings.append(f"语义检索未成功：{job.query}；原因：{result.get('error', '未知错误')}")
            return RetrievalJobResult(job=job, evidences=[], warnings=warnings)

        for index, item in enumerate(safe_result_items(result)):
            evidence = evidence_from_retrieval_item(
                item,
                issue=job.issue_title,
                source_query=job.query,
                retrieval_type="semantic",
            )
            # 每条 query 的前几名保底保留，之后的弱相关命中按分数线剔除。
            # 原因是多 query 并发检索会把大量长尾低分条文送进合并池，挤占最终 prompt 里
            # 真正相关的法条位置；保底名额则防止冷门案由整体低分时证据被清空。
            if (
                index >= SEMANTIC_JOB_MIN_KEEP
                and evidence.score is not None
                and evidence.score < MIN_SEMANTIC_EVIDENCE_SCORE
            ):
                continue
            evidences.append(evidence)
        return RetrievalJobResult(job=job, evidences=evidences, warnings=warnings)

    def _execute_keyword_job(self, job: RetrievalJob) -> RetrievalJobResult:
        """
        执行关键词兜底检索任务。
        """

        warnings: list[str] = []
        evidences: list[LegalArticleEvidence] = []
        try:
            result = self.retriever.search_legal_articles_by_keyword(
                keywords=job.keywords,
                top_k=self.keyword_top_k,
                legal_name=job.legal_name,
                category="",
                article_no="",
                source_type="law_article",
                include_neighbors=False,
                match_mode="any",
            )
        except Exception as error:  # pragma: no cover - 同语义检索，真实环境错误转为业务 warning。
            return RetrievalJobResult(job=job, evidences=[], warnings=[f"关键词检索失败：{', '.join(job.keywords)}；原因：{error}"])

        if not result.get("ok"):
            warnings.append(f"关键词检索未成功：{', '.join(job.keywords)}；原因：{result.get('error', '未知错误')}")
            return RetrievalJobResult(job=job, evidences=[], warnings=warnings)

        for item in safe_result_items(result):
            evidences.append(
                evidence_from_retrieval_item(
                    item,
                    issue=job.issue_title,
                    source_query=job.query,
                    retrieval_type="keyword",
                )
            )
        return RetrievalJobResult(job=job, evidences=evidences, warnings=warnings)


class LegalDeterministicWebSearchSubtask:
    """
    确定性公网案例与司法实践检索子任务。

    Args:
        tool_runner: 复用现有 ToolRegistry.run("web_search", args) 的工具执行器。
        max_queries: 每轮最多执行多少条公网检索。默认三条固定 query，分别面向相似案例、
            司法解释/权威规定和裁判规则/司法实践；三条并发执行，不增加链路串行耗时。
        result_limit_per_query: 每条 query 重排后最多保留多少个结果进入最终 prompt。
        web_search_workers: 公网检索并发 worker 数。
    """

    def __init__(
        self,
        *,
        tool_runner: SupportsToolRun,
        max_queries: int = DEFAULT_WEB_SEARCH_MAX_QUERIES,
        result_limit_per_query: int = DEFAULT_WEB_SEARCH_RESULT_LIMIT,
        web_search_workers: int = DEFAULT_WEB_SEARCH_WORKERS,
    ) -> None:
        self.tool_runner = tool_runner
        self.max_queries = max(1, int(max_queries))
        self.result_limit_per_query = max(1, int(result_limit_per_query))
        self.web_search_workers = max(1, int(web_search_workers))

    def run(
        self,
        *,
        user_input: str,
        state: LegalCaseState,
        rag: LegalCaseRagResult | None = None,
        risks: list[LegalRiskFinding] | None = None,
        catalog: LegalAnalysisCatalog | None = None,
        next_action: LegalNextAction | None = None,
    ) -> LegalWebSearchResearchResult:
        """
        执行固定目的的公网检索，并把工具失败降级为 warning。

        这里不用再让模型决定是否检索。原因是相似案例和司法实践对法律咨询答复很常用，
        用确定性 query 可以避免最终回答阶段遗漏公网补充材料，同时保持 query 数可控。

        rag 及之后的参数都允许为 None。原因是主链路会在案件状态更新完成后立即后台启动
        公网检索，让它和本地 RAG、综合分析并行；此时这些结果还没有产出。
        """

        query_specs = build_deterministic_web_search_queries(
            user_input=user_input,
            state=state,
            rag=rag,
            risks=risks or [],
            catalog=catalog or LegalAnalysisCatalog(),
            next_action=next_action or LegalNextAction(),
            max_queries=self.max_queries,
        )
        warnings: list[str] = []
        query_results: list[LegalWebSearchQueryResult] = []
        # 跨 query 去重使用 url 和规范化标题双 key。原因是同一篇文章经常被多个站点转载，
        # 只按 url 去重会让三条 query 的结果里出现三份同题内容。
        seen_keys: set[str] = set()
        raw_results = self._run_web_search_query_specs(query_specs)

        for spec, raw_result in zip(query_specs, raw_results, strict=False):
            purpose = spec["purpose"]
            query = spec["query"]
            if isinstance(raw_result, Exception):
                message = f"公网检索失败：{purpose}；原因：{raw_result}"
                warnings.append(message)
                query_results.append(
                    LegalWebSearchQueryResult(purpose=purpose, query=query, ok=False, results=[], error=str(raw_result))
                )
                continue

            if not isinstance(raw_result, dict):
                message = f"公网检索返回格式异常：{purpose}"
                warnings.append(message)
                query_results.append(
                    LegalWebSearchQueryResult(purpose=purpose, query=query, ok=False, results=[], error=message)
                )
                continue

            ok = bool(raw_result.get("ok"))
            error = normalize_text(raw_result.get("error"))
            items = web_search_items_from_tool_result(
                raw_result,
                seen_keys=seen_keys,
                limit=self.result_limit_per_query,
            )
            if not ok:
                message = f"公网检索未成功：{purpose}；原因：{error or '未知错误'}"
                warnings.append(message)
            query_results.append(
                LegalWebSearchQueryResult(purpose=purpose, query=query, ok=ok, results=items, error=error)
            )

        return LegalWebSearchResearchResult(query_results=query_results, warnings=warnings)

    def _run_web_search_query_specs(self, query_specs: list[dict[str, str]]) -> list[dict[str, Any] | Exception]:
        """
        执行公网检索 query，并保持返回顺序与 query_specs 一致。

        worker 线程只负责调用工具，不做 URL 去重。原因是去重需要全局顺序，放回主线程可以
        避免共享状态竞争，并保证测试和 UI 展示顺序稳定。
        """

        if not query_specs:
            return []
        if self.web_search_workers <= 1 or len(query_specs) == 1:
            return [self._run_one_web_search_query(spec) for spec in query_specs]

        worker_count = min(self.web_search_workers, len(query_specs))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="legal-web-search") as executor:
            return list(executor.map(self._run_one_web_search_query, query_specs))

    def _run_one_web_search_query(self, spec: dict[str, str]) -> dict[str, Any] | Exception:
        """
        执行单条公网检索 query，异常作为值返回供主线程降级为 warning。
        """

        arguments = {
            "query": spec["query"],
            "count": DEFAULT_WEB_SEARCH_COUNT,
            "summary": True,
            "freshness": "noLimit",
        }
        # include/exclude 由 query 规格按目的携带：司法解释限定权威站点，其余排除低质站点。
        if spec.get("include"):
            arguments["include"] = spec["include"]
        if spec.get("exclude"):
            arguments["exclude"] = spec["exclude"]
        try:
            raw_result = self.tool_runner.run("web_search", arguments)
        except Exception as error:  # pragma: no cover - 真实工具异常同样应降级。
            return error
        return raw_result


class LegalCaseAnalyzer:
    """
    风险识别、案情目录和下一步动作的合并分析子任务。

    原先这三步是三次串行 LLM 调用，但输入几乎相同（案件状态 + 法条证据摘要），拆开只会
    放大每轮咨询的串行延迟。合并成一次结构化调用后，每轮可省约两次 LLM 往返；对外事件仍由
    session 按三个独立步骤发出，前端展示协议保持不变。

    Args:
        llm: 支持 complete() 的 LLM 客户端。
        options: 子任务调用参数。默认使用放宽 max_tokens 的低温低推理配置。
    """

    def __init__(
        self,
        llm: SupportsComplete,
        *,
        options: LLMCallOptions | None = None,
    ) -> None:
        self.llm = llm
        self.options = options or CASE_ANALYSIS_OPTIONS

    def analyze(self, *, state: LegalCaseState, rag: LegalCaseRagResult) -> LegalCaseAnalysis:
        """
        一次性完成风险识别、案情目录和下一步动作判断。

        Args:
            state: 最新案件状态。
            rag: 本轮法条 RAG 结果。

        Returns:
            LegalCaseAnalysis: 风险项、案情目录和下一步动作的合并结果。

        Raises:
            ValueError: LLM 输出不是合法 JSON 或 risks 字段不是列表时抛出。
        """

        messages = [
            system_message(CASE_ANALYSIS_SYSTEM_PROMPT),
            user_message(build_case_analysis_user_prompt(state=state, rag=rag)),
        ]
        raw_response = self.llm.complete(messages, options=self.options)
        data = parse_json_object(raw_response)

        raw_risks = data.get("risks", [])
        if not isinstance(raw_risks, list):
            raise ValueError("综合分析结果字段 risks 必须是列表。")
        risks = [risk_from_dict(item) for item in raw_risks if isinstance(item, dict)]

        catalog_data = data.get("catalog") if isinstance(data.get("catalog"), dict) else {}
        catalog = LegalAnalysisCatalog(
            case_points=normalize_string_list(catalog_data.get("case_points"), max_items=12),
            legal_concepts=normalize_string_list(catalog_data.get("legal_concepts"), max_items=12),
            follow_up_questions=normalize_string_list(catalog_data.get("follow_up_questions"), max_items=12),
        )

        action_data = data.get("next_action") if isinstance(data.get("next_action"), dict) else {}
        action = str(action_data.get("action", NEXT_ACTION_ASK_FOLLOWUP)).strip()
        if action not in VALID_NEXT_ACTIONS:
            # 非法 action 降级为追问。原因是追问比错误地给出结论更安全。
            action = NEXT_ACTION_ASK_FOLLOWUP
        next_action = LegalNextAction(
            action=action,
            reasons=normalize_string_list(action_data.get("reasons"), max_items=8),
            questions_to_ask=normalize_string_list(action_data.get("questions_to_ask"), max_items=8),
            should_correct_previous_answer=parse_bool(action_data.get("should_correct_previous_answer")),
        )
        return LegalCaseAnalysis(risks=risks, catalog=catalog, next_action=next_action)


def build_state_update_user_prompt(
    *,
    previous_state: LegalCaseState,
    public_messages: list[Message],
    user_input: str,
) -> str:
    """
    构造案件状态更新子任务的用户提示词。
    """

    return f"""
请更新案件状态。

【上一轮结构化案件状态】
{json.dumps(asdict(previous_state), ensure_ascii=False, indent=2)}

【最近公开对话】
{json.dumps(compact_public_messages(public_messages), ensure_ascii=False, indent=2)}

【用户本轮输入】
{user_input}

【输出 JSON 结构】
{{
  "state": {{
    "summary": "string",
    "parties": ["string"],
    "timeline": ["string"],
    "confirmed_facts": ["string"],
    "disputed_facts": ["string"],
    "adverse_facts": ["string"],
    "contradictions": ["string"],
    "evidence_gaps": ["string"],
    "user_goals": ["string"],
    "legal_concepts": ["string"],
    "follow_up_questions": ["string"]
  }},
  "newly_added_facts": ["string"],
  "changed_facts": ["string"],
  "warnings": ["string"],
  "should_pause_for_supplement": false,
  "pause_reason": "string",
  "supplement_questions": ["string"],
  "supplement_evidence_gaps": ["string"]
}}
""".strip()


def build_planner_case_text(*, case_text: str, state: LegalCaseState) -> str:
    """
    构造传给现有 LegalQueryPlanner 的案情文本。

    Args:
        case_text: 本轮用户原文。
        state: 最新案件状态。

    Returns:
        str: 合并后的中性案情文本。
    """

    return f"""
【本轮用户输入】
{case_text}

【当前案件摘要】
{state.summary}

【已确认事实】
{format_bullets(state.confirmed_facts)}

【争议事实和矛盾】
{format_bullets(state.disputed_facts + state.contradictions)}

【用户目标】
{format_bullets(state.user_goals)}
""".strip()


def build_case_analysis_user_prompt(*, state: LegalCaseState, rag: LegalCaseRagResult) -> str:
    """
    构造综合分析（风险 + 目录 + 下一步动作）提示词。
    """

    return f"""
请基于案件状态和法条证据，一次性完成风险识别、案情目录和下一步动作判断。

【案件状态】
{json.dumps(asdict(state), ensure_ascii=False, indent=2)}

【检索到的法条证据摘要】
{json.dumps(compact_rag_for_prompt(rag), ensure_ascii=False, indent=2)}

【输出 JSON 结构】
{{
  "risks": [
    {{
      "type": "adverse_fact|contradiction|missing_evidence|legal_uncertainty",
      "severity": "high|medium|low",
      "fact": "触发风险的事实或缺口",
      "reason": "为什么影响案件判断",
      "suggestion": "下一步应如何澄清或补强"
    }}
  ],
  "catalog": {{
    "case_points": ["string"],
    "legal_concepts": ["string"],
    "follow_up_questions": ["string"]
  }},
  "next_action": {{
    "action": "answer_now|ask_followup|search_more|cannot_answer",
    "reasons": ["string"],
    "questions_to_ask": ["string"],
    "should_correct_previous_answer": false
  }}
}}
""".strip()


def legal_case_state_from_dict(
    data: Any,
    *,
    fallback: LegalCaseState | None = None,
    version: int = 0,
) -> LegalCaseState:
    """
    从宽松 JSON object 构造 LegalCaseState。

    Args:
        data: LLM 输出中的 state object。
        fallback: 字段缺失时使用的上一轮状态。
        version: 本次状态版本号。

    Returns:
        LegalCaseState: 规范化后的案件状态。
    """

    source = data if isinstance(data, dict) else {}
    fallback = fallback or LegalCaseState()
    return LegalCaseState(
        summary=normalize_text(source.get("summary")) or fallback.summary,
        parties=normalize_string_list(source.get("parties"), max_items=12) or list(fallback.parties),
        timeline=normalize_string_list(source.get("timeline"), max_items=20) or list(fallback.timeline),
        confirmed_facts=normalize_string_list(source.get("confirmed_facts"), max_items=30)
        or list(fallback.confirmed_facts),
        disputed_facts=normalize_string_list(source.get("disputed_facts"), max_items=20)
        or list(fallback.disputed_facts),
        adverse_facts=normalize_string_list(source.get("adverse_facts"), max_items=20)
        or list(fallback.adverse_facts),
        contradictions=normalize_string_list(source.get("contradictions"), max_items=20)
        or list(fallback.contradictions),
        evidence_gaps=normalize_string_list(source.get("evidence_gaps"), max_items=20)
        or list(fallback.evidence_gaps),
        user_goals=normalize_string_list(source.get("user_goals"), max_items=12) or list(fallback.user_goals),
        legal_concepts=normalize_string_list(source.get("legal_concepts"), max_items=16)
        or list(fallback.legal_concepts),
        follow_up_questions=normalize_string_list(source.get("follow_up_questions"), max_items=12)
        or list(fallback.follow_up_questions),
        version=version,
    )


def risk_from_dict(data: dict[str, Any]) -> LegalRiskFinding:
    """
    从 LLM JSON object 构造风险项。
    """

    return LegalRiskFinding(
        type=normalize_text(data.get("type")) or "legal_uncertainty",
        severity=normalize_severity(data.get("severity")),
        fact=normalize_text(data.get("fact")),
        reason=normalize_text(data.get("reason")),
        suggestion=normalize_text(data.get("suggestion")),
    )


def merge_state_with_analysis(
    *,
    state: LegalCaseState,
    risks: list[LegalRiskFinding],
    catalog: LegalAnalysisCatalog,
) -> LegalCaseState:
    """
    把风险识别和案情目录结果合并回案件状态。

    这样做的原因是后续轮次需要继承结构化风险和追问目录，但仍然不把内部 LLM 原始对话写入主会话。
    """

    adverse_facts = merge_text_lists(
        state.adverse_facts,
        [risk.fact for risk in risks if risk.type == "adverse_fact" and risk.fact],
        limit=20,
    )
    contradictions = merge_text_lists(
        state.contradictions,
        [risk.fact for risk in risks if risk.type == "contradiction" and risk.fact],
        limit=20,
    )
    evidence_gaps = merge_text_lists(
        state.evidence_gaps,
        [risk.fact for risk in risks if risk.type == "missing_evidence" and risk.fact],
        limit=20,
    )
    legal_concepts = merge_text_lists(state.legal_concepts, catalog.legal_concepts, limit=16)
    follow_up_questions = merge_text_lists(state.follow_up_questions, catalog.follow_up_questions, limit=12)
    return replace(
        state,
        adverse_facts=adverse_facts,
        contradictions=contradictions,
        evidence_gaps=evidence_gaps,
        legal_concepts=legal_concepts,
        follow_up_questions=follow_up_questions,
    )


def evidence_from_retrieval_item(
    item: dict[str, Any],
    *,
    issue: str,
    source_query: str,
    retrieval_type: str,
) -> LegalArticleEvidence:
    """
    把 LegalArticleRetriever 返回的 result item 转成统一证据模型。
    """

    legal_name = normalize_text(item.get("legal_name"))
    article_no = normalize_text(item.get("article_no"))
    citation = normalize_text(item.get("citation")) or f"《{legal_name}》{article_no}"
    text = normalize_text(item.get("text")) or normalize_text(item.get("document"))
    return LegalArticleEvidence(
        citation=citation,
        legal_name=legal_name,
        article_no=article_no,
        text=text,
        issue=issue,
        source_query=source_query,
        retrieval_type=retrieval_type,
        score=safe_float(item.get("score")),
        hit_count=1,
    )


def merge_evidence(
    evidence_map: dict[str, LegalArticleEvidence],
    evidence: LegalArticleEvidence,
) -> None:
    """
    把证据合并进去重 map。

    重复命中时增加 hit_count，并保留较高 score。这样多 query 同时命中的条文会在重排中靠前。
    """

    key = evidence_key(evidence)
    existing = evidence_map.get(key)
    if existing is None:
        evidence_map[key] = evidence
        return

    evidence_map[key] = replace(
        existing,
        hit_count=existing.hit_count + evidence.hit_count,
        score=max_optional_float(existing.score, evidence.score),
    )


def evidence_key(evidence: LegalArticleEvidence) -> str:
    """
    构造法条证据去重 key。
    """

    if evidence.legal_name and evidence.article_no:
        return f"law::{evidence.legal_name}::{evidence.article_no}"
    if evidence.citation:
        return f"citation::{evidence.citation}"
    return f"text::{evidence.text[:120]}"


def evidence_rank_score(
    evidence: LegalArticleEvidence,
    *,
    positive_terms: list[str] | tuple[str, ...] = (),
    negative_terms: list[str] | tuple[str, ...] = (),
) -> float:
    """
    计算法条证据的融合重排分数。

    Args:
        evidence: 待打分的法条证据。
        positive_terms: planner 建议加权的正向关键词。
        negative_terms: planner 建议降权的反向关键词。

    Returns:
        float: 融合后的排序分数，越大越靠前。

    Why:
        以检索分数为主体，多 query 重复命中和正向词只做小幅加分，反向词小幅减分。
        这样单次高分的精准条文不会被多次低分的泛化条文压下去，同时 planner 已经产出的
        positive_terms/negative_terms 真正参与重排，而不是只停留在计划里。
    """

    base = evidence.score if evidence.score is not None else 0.0
    fused = base + EVIDENCE_EXTRA_HIT_BONUS * min(max(evidence.hit_count - 1, 0), EVIDENCE_EXTRA_HIT_BONUS_CAP)

    if positive_terms or negative_terms:
        text_blob = f"{evidence.citation} {evidence.text}"
        positive_hits = sum(1 for term in dict.fromkeys(positive_terms) if term and term in text_blob)
        fused += min(positive_hits * EVIDENCE_POSITIVE_TERM_BONUS, EVIDENCE_POSITIVE_TERM_BONUS_CAP)
        negative_hits = sum(1 for term in dict.fromkeys(negative_terms) if term and term in text_blob)
        fused -= min(negative_hits * EVIDENCE_NEGATIVE_TERM_PENALTY, EVIDENCE_NEGATIVE_TERM_PENALTY_CAP)
    return fused


def sort_evidences(
    evidences: Any,
    *,
    positive_terms: list[str] | tuple[str, ...] = (),
    negative_terms: list[str] | tuple[str, ...] = (),
) -> list[LegalArticleEvidence]:
    """
    对法条证据按融合分数重排。

    Args:
        evidences: 待排序证据集合。
        positive_terms: 参与加权的正向关键词。
        negative_terms: 参与降权的反向关键词。

    Returns:
        list[LegalArticleEvidence]: 分数从高到低排序的证据列表，同分时按 citation 稳定排序。
    """

    return sorted(
        list(evidences),
        key=lambda item: (
            -evidence_rank_score(item, positive_terms=positive_terms, negative_terms=negative_terms),
            item.citation,
        ),
    )


def group_retrieval_results_by_issue(
    results: list[RetrievalJobResult],
) -> dict[int, list[RetrievalJobResult]]:
    """
    按 issue 下标归并检索结果，并保持每个 issue 内的任务顺序稳定。

    Args:
        results: 所有检索任务结果。

    Returns:
        dict[int, list[RetrievalJobResult]]: issue 下标到检索结果列表的映射。
    """

    grouped: dict[int, list[RetrievalJobResult]] = {}
    for result in sorted(results, key=lambda item: item.job.index):
        grouped.setdefault(result.job.issue_index, []).append(result)
    return grouped


def build_deterministic_web_search_queries(
    *,
    user_input: str,
    state: LegalCaseState,
    rag: LegalCaseRagResult | None = None,
    risks: list[LegalRiskFinding] | None = None,
    catalog: LegalAnalysisCatalog | None = None,
    next_action: LegalNextAction | None = None,
    max_queries: int = DEFAULT_WEB_SEARCH_MAX_QUERIES,
) -> list[dict[str, str]]:
    """
    构造每轮固定执行的公网检索 query。

    Args:
        user_input: 用户本轮原始输入。
        state: 最新案件状态。
        rag: 已检索到的本地法条证据；提前并行启动时可为 None。
        risks: 当前风险项；提前并行启动时可为 None。
        catalog: 案情目录；提前并行启动时可为 None。
        next_action: 下一步动作；提前并行启动时可为 None。
        max_queries: 最多返回多少条 query。

    Returns:
        list[dict[str, str]]: 包含 purpose 和 query 的检索规格列表。

    Why:
        query 核心词只取最关键的少量事实和剥掉“可能涉及”前缀的法律概念。搜索引擎对
        长串事实拼接的匹配质量很差，短而聚焦的关键词组合更容易命中裁判文书、司法解释
        和权威解读；三条 query 分别锁定相似案例、司法解释/权威规定和裁判规则/实务口径。
    """

    catalog = catalog or LegalAnalysisCatalog()
    concepts = [
        term
        for term in (normalize_legal_concept_term(item) for item in (state.legal_concepts or catalog.legal_concepts))
        if term
    ][:3]
    fact_source = state.confirmed_facts or catalog.case_points
    facts = [truncate_text(normalize_text(item), 30) for item in fact_source[:2] if normalize_text(item)]
    core_text = normalize_text(" ".join([*facts, *concepts]))
    if not core_text:
        core_text = normalize_text(state.summary) or normalize_text(user_input)
    core_text = truncate_text(core_text, 60)

    # 案情提到赔偿或量刑诉求时，把实务 query 引向对应的标准/口径，而不是泛化普法文章。
    hint_blob = " ".join([state.summary, *state.user_goals, *facts, *concepts, user_input])
    practice_suffix = ""
    if COMPENSATION_HINT_PATTERN.search(hint_blob):
        practice_suffix = " 赔偿标准"
    elif SENTENCING_HINT_PATTERN.search(hint_blob):
        practice_suffix = " 量刑标准"

    specs = [
        {
            "purpose": "similar_cases",
            "query": truncate_text(f"{core_text} 判决 典型案例 裁判文书", 100),
            "exclude": WEB_SEARCH_EXCLUDE_SITES,
        },
        {
            "purpose": "judicial_interpretation",
            "query": truncate_text(f"{core_text} 司法解释 最高人民法院 规定 理解与适用", 100),
            "include": WEB_SEARCH_AUTHORITATIVE_INCLUDE_SITES,
        },
        {
            "purpose": "judicial_practice",
            "query": truncate_text(f"{core_text} 裁判规则 裁判要旨 司法实践{practice_suffix}", 100),
            "exclude": WEB_SEARCH_EXCLUDE_SITES,
        },
    ]
    return specs[: max(1, int(max_queries))]


def normalize_legal_concept_term(value: Any) -> str:
    """
    把案件状态里的法律概念转成适合公网搜索的短词。

    Args:
        value: 原始概念文本，例如“可能涉及二倍工资”。

    Returns:
        str: 剥掉推测性前缀后的概念词，例如“二倍工资”；无有效内容时返回空字符串。
    """

    text = normalize_text(value)
    if not text:
        return ""
    text = LEGAL_CONCEPT_PREFIX_PATTERN.sub("", text).strip()
    return truncate_text(text, 24)


def web_search_items_from_tool_result(
    result: dict[str, Any],
    *,
    seen_keys: set[str],
    limit: int,
) -> list[LegalWebSearchItem]:
    """
    从 web_search 工具结果中提取去重、重排后的精简条目。

    Args:
        result: web_search 工具的结构化返回。
        seen_keys: 跨 query 共享的去重 key 集合（url 和规范化标题）；保留条目的 key 会写回该集合。
        limit: 重排后最多保留多少条。

    Returns:
        list[LegalWebSearchItem]: 按权威度分数从高到低排列的结果条目。
    """

    candidates: list[LegalWebSearchItem] = []
    candidate_keys: list[set[str]] = []
    batch_seen: set[str] = set()
    for item in safe_result_items(result):
        url = normalize_text(item.get("url"))
        if not url:
            continue
        keys = {f"url::{url}"}
        title_key = normalize_title_for_dedup(item.get("title"))
        if title_key:
            keys.add(f"title::{title_key}")
        if keys & seen_keys or keys & batch_seen:
            continue
        batch_seen.update(keys)
        candidates.append(
            LegalWebSearchItem(
                title=truncate_text(normalize_text(item.get("title")), 120),
                url=url,
                snippet=truncate_text(normalize_text(item.get("snippet")), 300),
                summary=truncate_text(normalize_text(item.get("summary")), 500),
                site_name=truncate_text(normalize_text(item.get("site_name")), 80),
                display_url=truncate_text(normalize_text(item.get("display_url")), 120),
                date_published=truncate_text(normalize_text(item.get("date_published")), 40),
                date_last_crawled=truncate_text(normalize_text(item.get("date_last_crawled")), 40),
                authority_level=classify_web_authority_level(url),
            )
        )
        candidate_keys.append(keys)

    kept_items = rank_and_filter_web_items(candidates, limit=limit)
    kept_ids = {id(item) for item in kept_items}
    for item, keys in zip(candidates, candidate_keys, strict=False):
        # 只把保留条目的 key 写回全局集合。被重排淘汰的条目允许在后续 query 中再次竞争，
        # 因为它在另一条 query 的候选池里可能是相对最优结果。
        if id(item) in kept_ids:
            seen_keys.update(keys)
    return kept_items


def rank_and_filter_web_items(
    items: list[LegalWebSearchItem],
    *,
    limit: int,
) -> list[LegalWebSearchItem]:
    """
    对公网检索候选按权威度重排，并压制低置信度站点。

    Args:
        items: 已去重的候选条目，按搜索引擎原始顺序排列。
        limit: 最多保留多少条。

    Returns:
        list[LegalWebSearchItem]: 重排后的条目。低置信度站点只有在非低置信度结果不足
        WEB_LOW_AUTHORITY_BACKFILL_LIMIT 条时才回填，避免资料栏被内容农场文章占满。
    """

    scored = sorted(
        enumerate(items),
        key=lambda pair: (-score_web_search_item(pair[1]), pair[0]),
    )
    preferred = [item for _, item in scored if item.authority_level != "low"]
    low_quality = [item for _, item in scored if item.authority_level == "low"]

    keep_limit = max(1, int(limit))
    kept = preferred[:keep_limit]
    # 回填目标同时受 limit 约束，避免 limit=1 时反而因回填输出两条。
    backfill_target = min(keep_limit, WEB_LOW_AUTHORITY_BACKFILL_LIMIT)
    if len(kept) < backfill_target:
        kept.extend(low_quality[: backfill_target - len(kept)])
    return kept


def score_web_search_item(item: LegalWebSearchItem) -> float:
    """
    计算单条公网结果的权威度分数。

    Args:
        item: 已分级的公网结果条目。

    Returns:
        float: 域名基础分 + 权威内容特征词加分 + 规范案号加分。
    """

    score = WEB_AUTHORITY_DOMAIN_SCORES.get(item.authority_level, 0.0)
    text_blob = " ".join([item.title, item.snippet, item.summary, item.site_name])
    keyword_hits = sum(1 for keyword in WEB_AUTHORITY_CONTENT_KEYWORDS if keyword in text_blob)
    score += min(keyword_hits * WEB_AUTHORITY_KEYWORD_BONUS, WEB_AUTHORITY_KEYWORD_BONUS_CAP)
    if WEB_CASE_NUMBER_PATTERN.search(text_blob):
        score += WEB_CASE_NUMBER_BONUS
    return score


def classify_web_authority_level(url: str) -> str:
    """
    按域名后缀判断公网结果的权威度分级。

    Args:
        url: 结果链接。

    Returns:
        str: high、medium、low 或 normal。只依据域名判断，不采信页面内容自我声明。
    """

    host = extract_url_host(url)
    if not host:
        return "normal"
    for suffix in HIGH_AUTHORITY_DOMAIN_SUFFIXES:
        if host_matches_domain_suffix(host, suffix):
            return "high"
    for suffix in MEDIUM_AUTHORITY_DOMAIN_SUFFIXES:
        if host_matches_domain_suffix(host, suffix):
            return "medium"
    for suffix in LOW_AUTHORITY_DOMAIN_SUFFIXES:
        if host_matches_domain_suffix(host, suffix):
            return "low"
    return "normal"


def extract_url_host(url: str) -> str:
    """
    从 URL 中提取小写主机名，去掉端口和用户信息。
    """

    match = re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://([^/?#]+)", str(url or "").strip())
    if not match:
        return ""
    host = match.group(1)
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    return host.split(":", 1)[0].strip().lower()


def host_matches_domain_suffix(host: str, suffix: str) -> bool:
    """
    判断主机名是否等于某域名后缀或属于其子域名。

    整段匹配而不是简单 endswith，可防止 fakegov.cn 这类拼接域名冒充 gov.cn。
    """

    return host == suffix or host.endswith(f".{suffix}")


def normalize_title_for_dedup(title: Any) -> str:
    """
    生成用于跨 query 标题去重的规范化 key。

    去掉空白和标点后取小写；过短标题区分度不足，返回空字符串表示不参与标题去重。
    """

    text = normalize_text(title)
    if not text:
        return ""
    normalized = re.sub(r"[\W_]+", "", text).lower()
    if len(normalized) < 6:
        return ""
    return normalized[:80]


def compact_web_research_for_prompt(
    web_research: LegalWebSearchResearchResult | None,
    *,
    max_items_per_query: int = 5,
) -> dict[str, Any]:
    """
    压缩公网检索结果，供最终回答 prompt 使用。

    只保留标题、链接、摘录和摘要等必要字段。这样做的原因是公网搜索结果可能很长，
    最终回答只需要可引用来源和简短上下文，不应把工具原始响应塞进 prompt。
    """

    if web_research is None:
        return {"query_results": [], "warnings": []}
    return {
        "query_results": [
            {
                "purpose": item.purpose,
                "query": item.query,
                "ok": item.ok,
                "error": item.error,
                "results": [asdict(result) for result in item.results[:max_items_per_query]],
            }
            for item in web_research.query_results
        ],
        "warnings": web_research.warnings[:8],
    }


def compact_rag_for_prompt(rag: LegalCaseRagResult, *, max_items: int = 12) -> dict[str, Any]:
    """
    压缩 RAG 结果，避免内部 prompt 过长。
    """

    return {
        "issues": [
            {
                "issue": issue.issue,
                "facts": issue.facts,
                "used_queries": issue.used_queries,
                "evidence_count": len(issue.evidences),
            }
            for issue in rag.issue_results
        ],
        "evidences": [evidence_to_prompt_dict(item) for item in rag.evidences[:max_items]],
        "warnings": rag.warnings[:8],
    }


def evidence_to_prompt_dict(evidence: LegalArticleEvidence, *, text_limit: int = 500) -> dict[str, Any]:
    """
    把法条证据压缩成最终 prompt 需要的字段。
    """

    return {
        "citation": evidence.citation,
        "legal_name": evidence.legal_name,
        "article_no": evidence.article_no,
        "text": truncate_text(evidence.text, text_limit),
        "issue": evidence.issue,
        "source_query": evidence.source_query,
        "retrieval_type": evidence.retrieval_type,
        "hit_count": evidence.hit_count,
    }


def query_plan_to_prompt_dict(plan: LegalQueryPlan) -> dict[str, Any]:
    """
    转换 query plan，默认不暴露 raw_response。
    """

    return plan_to_dict(plan, include_raw_response=False)


def compact_public_messages(messages: list[Message], *, max_messages: int = 6) -> list[dict[str, str]]:
    """
    压缩公开历史，只给状态更新器看最近几轮。
    """

    compacted: list[dict[str, str]] = []
    for message in messages[-max_messages:]:
        role = str(message.get("role", ""))
        if role == "system":
            continue
        compacted.append(
            {
                "role": role,
                "content": truncate_text(str(message.get("content", "")), 500),
            }
        )
    return compacted


def safe_result_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    安全读取检索结果中的 results 列表。
    """

    items = result.get("results", [])
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def normalize_string_list(value: Any, *, max_items: int) -> list[str]:
    """
    宽松规范化字符串列表。

    与 query planner 的严格校验不同，这里的 LLM 子任务结果属于业务状态，字段缺失时应尽量降级为空列表。
    """

    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = [value]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        if len(normalized) >= max_items:
            break
        text = normalize_text(item)
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


def merge_text_lists(first: list[str], second: list[str], *, limit: int) -> list[str]:
    """
    合并两个文本列表并去重保序。
    """

    return normalize_string_list([*first, *second], max_items=limit)


def normalize_text(value: Any) -> str:
    """
    规范化单个文本值。
    """

    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return " ".join(text.split())


def normalize_severity(value: Any) -> str:
    """
    规范化风险强度。
    """

    severity = normalize_text(value).lower()
    if severity in {"high", "medium", "low"}:
        return severity
    return "medium"


def parse_bool(value: Any) -> bool:
    """
    宽松解析布尔值。
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "是", "需要"}
    return bool(value)


def safe_float(value: Any) -> float | None:
    """
    安全转换浮点数。
    """

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def max_optional_float(first: float | None, second: float | None) -> float | None:
    """
    返回两个可空浮点数中较大的一个。
    """

    if first is None:
        return second
    if second is None:
        return first
    return max(first, second)


def truncate_text(text: str, limit: int) -> str:
    """
    截断过长文本，避免 prompt 膨胀。
    """

    if len(text) <= limit:
        return text
    return f"{text[:limit]}……"


def format_bullets(items: list[str]) -> str:
    """
    把列表格式化为简短项目符号文本。
    """

    if not items:
        return "- 暂无"
    return "\n".join(f"- {item}" for item in items)


def emit_optional_event(
    on_event: Callable[[AgentEvent], None] | None,
    event: AgentEvent,
) -> None:
    """
    安全触发可选实时事件回调。

    Args:
        on_event: 调用方传入的回调；为空时不做任何事。
        event: 要推送的事件。
    """

    if on_event is not None:
        on_event(event)


__all__ = [
    "CASE_ANALYSIS_OPTIONS",
    "STRUCTURED_SUBTASK_OPTIONS",
    "WEB_SEARCH_PURPOSE_LABELS",
    "LegalCaseAnalyzer",
    "LegalCaseRagSubtask",
    "LegalCaseStateUpdater",
    "LegalDeterministicWebSearchSubtask",
    "SupportsToolRun",
    "build_deterministic_web_search_queries",
    "classify_web_authority_level",
    "compact_rag_for_prompt",
    "compact_web_research_for_prompt",
    "evidence_rank_score",
    "evidence_to_prompt_dict",
    "merge_state_with_analysis",
    "query_plan_to_prompt_dict",
    "rank_and_filter_web_items",
    "score_web_search_item",
    "sort_evidences",
]
