"""
跨会话案件记忆：知识沉淀与记忆检索。

本模块解决的问题：每个法律咨询会话结束后，其结构化案件知识（摘要、关键事实、诉求、
法律概念）随会话沉睡在快照里；新会话即使咨询同类问题也无法利用。这里把每轮成功提交的
case_state 确定性地蒸馏成一条“案件记忆”，落盘到独立目录，并提供关键词打分检索，
供新会话在链路开始前唤起相关历史咨询背景。

设计取舍：

1. 沉淀不发起任何 LLM 调用。case_state 本身就是状态更新器（LLM）蒸馏过的结构化知识，
   直接提取即可；这样保持“一轮成功链路共 4 次 LLM 调用”的既有约束不变。
2. 检索用关键词子串打分而不是 embedding。本地记忆规模是几十条量级，关键词方案零模型
   依赖、毫秒级返回、结果可解释（能给出命中的词），符合项目“保持简单”的偏好。
3. 一个会话对应一个记忆文件（upsert 覆盖），会话进展时记忆随最新 case_state 演进；
   删除会话时删除对应记忆，避免已删除会话的知识残留在召回结果里。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable

from agent_system.legal_consultation.subtasks import (
    normalize_legal_concept_term,
    normalize_string_list,
    normalize_text,
    truncate_text,
)
from agent_system.storage.session_store import (
    SESSION_ID_PATTERN,
    SessionStoreError,
    atomic_write_json,
    read_json_file,
    utc_now_text,
)

MEMORY_SCHEMA_VERSION = "legal_case_memory.v1"
DEFAULT_RECALL_TOP_K = 3
MAX_MEMORY_KEYWORDS = 24

# 检索关键词停用词：这些壳词几乎命中任何案情输入，只会制造无意义召回。
# 只挡最通用的表述词，不挡“二倍工资”“借条”这类有区分度的实体法概念。
MEMORY_KEYWORD_STOPWORDS = {
    "用户",
    "本人",
    "对方",
    "当事人",
    "相关",
    "情况",
    "问题",
    "法律",
    "咨询",
    "纠纷",
    "案件",
    "事项",
}


class MemoryStoreError(Exception):
    """
    案件记忆存储层错误。

    与 SessionStoreError 分开定义的原因是：记忆属于“锦上添花”的辅助能力，调用方（Web 层）
    需要按类型把记忆失败降级为非致命提示，而不是和会话持久化失败混在一起处理。
    """


@dataclass(frozen=True)
class LegalCaseMemory:
    """
    一条跨会话案件记忆。

    Attributes:
        session_id: 来源会话 ID，同时作为记忆文件名。
        title: 记忆标题，通常取会话标题或案情摘要截断。
        summary: 案情中性摘要，来自 case_state.summary。
        key_facts: 关键已确认事实。
        user_goals: 用户诉求。
        legal_concepts: 可能涉及的法律概念（保留“可能涉及”原始表述，展示用）。
        keywords: 检索关键词（已剥掉推测性前缀的短词，匹配用）。
        turn_count: 沉淀时会话已完成的轮数。
        created_at: 首次沉淀时间，upsert 时保留。
        updated_at: 最近一次沉淀时间。
    """

    session_id: str
    title: str = ""
    summary: str = ""
    key_facts: list[str] = field(default_factory=list)
    user_goals: list[str] = field(default_factory=list)
    legal_concepts: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    turn_count: int = 0
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class RecalledCaseMemory:
    """
    一条被唤起的历史记忆及其匹配信息。

    Attributes:
        memory: 命中的记忆条目。
        score: 关键词命中得分（命中词长度之和，长词更具体、权重自然更高）。
        matched_keywords: 实际命中的关键词，用于解释“为什么召回它”。
    """

    memory: LegalCaseMemory
    score: int
    matched_keywords: list[str] = field(default_factory=list)


class MemoryStore:
    """
    基于本地文件的案件记忆存储。

    Args:
        base_dir: 记忆根目录，例如 `data/memory`；不存在时自动创建。
        now_provider: 可选时间源，返回 ISO 时间字符串。默认取 UTC 当前时间；
            测试注入固定时间即可验证 created_at/updated_at 语义，不必 mock 模块函数。
    """

    def __init__(self, base_dir: str | Path, *, now_provider: Callable[[], str] | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._now = now_provider or utc_now_text

    def save(self, memory: LegalCaseMemory) -> LegalCaseMemory:
        """
        以 upsert 语义保存一条记忆：同会话覆盖旧文件，created_at 保留首次值。

        Args:
            memory: 待保存的记忆条目；时间戳字段由本方法统一盖章。

        Returns:
            LegalCaseMemory: 已补全时间戳的最终落盘内容。

        Raises:
            MemoryStoreError: session_id 非法或写盘失败时抛出。
        """

        path = self._memory_path(memory.session_id)
        existing: LegalCaseMemory | None = None
        if path.is_file():
            try:
                existing = self._read_memory_file(path)
            except MemoryStoreError:
                # 旧文件损坏时直接用新内容覆盖：记忆可以从最新快照完整重建，没有抢救价值。
                existing = None

        now = self._now()
        created_at = (existing.created_at if existing else "") or memory.created_at or now
        stamped = replace(memory, created_at=created_at, updated_at=now)
        payload = {"schema_version": MEMORY_SCHEMA_VERSION, **asdict(stamped)}
        try:
            atomic_write_json(path, payload)
        except SessionStoreError as error:
            raise MemoryStoreError(str(error)) from error
        return stamped

    def load(self, session_id: str) -> LegalCaseMemory | None:
        """
        按会话 ID 读取记忆。

        Returns:
            LegalCaseMemory | None: 记忆不存在时返回 None。

        Raises:
            MemoryStoreError: session_id 非法或文件损坏时抛出。
        """

        path = self._memory_path(session_id)
        if not path.is_file():
            return None
        return self._read_memory_file(path)

    def load_all(self) -> list[LegalCaseMemory]:
        """
        读取全部记忆，按最近更新倒序。

        单个文件损坏时跳过而不是整体失败，保证一个坏文件不会让记忆检索完全不可用。
        """

        memories: list[LegalCaseMemory] = []
        for path in self.base_dir.glob("*.json"):
            if not SESSION_ID_PATTERN.match(path.stem):
                continue
            try:
                memories.append(self._read_memory_file(path))
            except MemoryStoreError:
                continue
        memories.sort(key=lambda item: item.updated_at, reverse=True)
        return memories

    def delete(self, session_id: str) -> None:
        """
        删除指定会话的记忆文件；文件不存在时静默返回（幂等）。

        Raises:
            MemoryStoreError: session_id 非法或删除失败时抛出。
        """

        path = self._memory_path(session_id)
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            raise MemoryStoreError(f"删除记忆 {session_id} 失败：{error}") from error

    def search(
        self,
        query_text: str,
        *,
        exclude_session_id: str | None = None,
        top_k: int = DEFAULT_RECALL_TOP_K,
    ) -> list[RecalledCaseMemory]:
        """
        用关键词子串命中打分检索历史记忆。

        Args:
            query_text: 检索文本，通常是用户本轮原始输入。
            exclude_session_id: 需要排除的会话（当前会话自己的记忆已在 case_state 里）。
            top_k: 最多返回条数，限制注入 prompt 的历史背景规模。

        Returns:
            list[RecalledCaseMemory]: 按得分降序（平分时最近更新优先）的召回结果；
            无命中或输入为空时返回空列表，不做兜底放水。
        """

        query = normalize_text(query_text)
        if not query:
            return []

        recalls: list[RecalledCaseMemory] = []
        for memory in self.load_all():
            if exclude_session_id and memory.session_id == exclude_session_id:
                continue
            matched = [keyword for keyword in memory.keywords if keyword and keyword in query]
            if not matched:
                continue
            # 得分取命中词长度之和：长词（如“违法解除劳动合同”）比短词更能说明两案相关，
            # 简单求和即可让更具体的记忆自然排前，不需要额外调权。
            score = sum(len(keyword) for keyword in matched)
            recalls.append(RecalledCaseMemory(memory=memory, score=score, matched_keywords=matched))

        # 两次稳定排序实现“得分降序、平分时最近更新优先”：先按时间倒序，再按得分倒序。
        recalls.sort(key=lambda item: item.memory.updated_at, reverse=True)
        recalls.sort(key=lambda item: item.score, reverse=True)
        return recalls[: max(0, int(top_k))]

    def _memory_path(self, session_id: str) -> Path:
        """
        校验 session_id 后返回记忆文件路径，防止路径穿越。
        """

        if not SESSION_ID_PATTERN.match(str(session_id or "")):
            raise MemoryStoreError(f"非法 session_id：{session_id!r}")
        return self.base_dir / f"{session_id}.json"

    def _read_memory_file(self, path: Path) -> LegalCaseMemory:
        """
        读取并校验单个记忆文件。

        Raises:
            MemoryStoreError: JSON 损坏或 schema 版本不兼容时抛出。
        """

        try:
            data = read_json_file(path)
        except SessionStoreError as error:
            raise MemoryStoreError(str(error)) from error
        version = str(data.get("schema_version") or "")
        if version != MEMORY_SCHEMA_VERSION:
            raise MemoryStoreError(f"{path.name} schema 版本不兼容：{version or '缺失'}（期望 {MEMORY_SCHEMA_VERSION}）。")
        return legal_case_memory_from_dict(data, fallback_session_id=path.stem)


def legal_case_memory_from_dict(data: dict[str, Any], *, fallback_session_id: str = "") -> LegalCaseMemory:
    """
    从字典安全构建记忆条目：逐字段规范化，未知字段忽略。

    这样旧版本文件多出或缺少字段时仍能得到合法对象，不会因为 schema 漂移让记忆库不可用。
    """

    valid_fields = {item.name for item in fields(LegalCaseMemory)}
    raw = {key: value for key, value in data.items() if key in valid_fields}
    return LegalCaseMemory(
        session_id=normalize_text(raw.get("session_id")) or fallback_session_id,
        title=normalize_text(raw.get("title")),
        summary=normalize_text(raw.get("summary")),
        key_facts=normalize_string_list(raw.get("key_facts"), max_items=8),
        user_goals=normalize_string_list(raw.get("user_goals"), max_items=5),
        legal_concepts=normalize_string_list(raw.get("legal_concepts"), max_items=8),
        keywords=normalize_string_list(raw.get("keywords"), max_items=MAX_MEMORY_KEYWORDS),
        turn_count=safe_turn_count(raw.get("turn_count")),
        created_at=normalize_text(raw.get("created_at")),
        updated_at=normalize_text(raw.get("updated_at")),
    )


def build_case_memory_from_snapshot(
    session_id: str,
    snapshot: dict[str, Any],
    *,
    title: str = "",
    turn_count: int = 0,
) -> LegalCaseMemory | None:
    """
    从会话快照确定性构建案件记忆，不发起任何 LLM 调用。

    Args:
        session_id: 来源会话 ID。
        snapshot: `export_snapshot()` 产出的快照，需含 `case_state` 字段。
        title: 可选记忆标题，通常传会话标题；为空时用案情摘要截断兜底。
        turn_count: 会话已完成轮数。

    Returns:
        LegalCaseMemory | None: 案件状态没有可沉淀内容（无摘要、事实和法律概念）或快照
        缺少 case_state 时返回 None，调用方直接跳过保存即可。
    """

    case_state = snapshot.get("case_state") if isinstance(snapshot, dict) else None
    if not isinstance(case_state, dict):
        return None

    summary = normalize_text(case_state.get("summary"))
    key_facts = normalize_string_list(case_state.get("confirmed_facts"), max_items=8)
    user_goals = normalize_string_list(case_state.get("user_goals"), max_items=5)
    legal_concepts = normalize_string_list(case_state.get("legal_concepts"), max_items=8)
    if not summary and not key_facts and not legal_concepts:
        return None

    memory_title = truncate_text(normalize_text(title), 40) or truncate_text(summary, 30)
    return LegalCaseMemory(
        session_id=str(session_id or ""),
        title=memory_title,
        summary=truncate_text(summary, 200),
        key_facts=[truncate_text(item, 80) for item in key_facts],
        user_goals=[truncate_text(item, 60) for item in user_goals],
        legal_concepts=[truncate_text(item, 40) for item in legal_concepts],
        keywords=extract_memory_keywords(case_state),
        turn_count=max(0, int(turn_count or 0)),
    )


def extract_memory_keywords(case_state: dict[str, Any]) -> list[str]:
    """
    从案件状态提取检索关键词。

    法律概念剥掉“可能涉及/涉嫌”等推测性前缀后是最有区分度的匹配词；当事人和诉求作为
    补充信号。只做去重、长度和停用词过滤，不做分词——关键词本身已经是结构化短语，
    子串匹配足够用。
    """

    candidates: list[str] = []
    for value in case_state.get("legal_concepts") or []:
        candidates.append(normalize_legal_concept_term(value))
    for source_field in ("parties", "user_goals"):
        for value in case_state.get(source_field) or []:
            candidates.append(truncate_text(normalize_text(value), 24))

    keywords: list[str] = []
    for word in candidates:
        if len(word) < 2 or word in MEMORY_KEYWORD_STOPWORDS or word in keywords:
            continue
        keywords.append(word)
        if len(keywords) >= MAX_MEMORY_KEYWORDS:
            break
    return keywords


def recalled_memories_to_prompt_payload(recalls: list[RecalledCaseMemory]) -> list[dict[str, Any]]:
    """
    把召回结果转换为可注入 prompt 和事件的白名单负载。

    只保留展示与理解案情所需字段，不透出 session_id、score 等内部标识，
    这样业务会话层和 Web 层拿到的就是可以直接使用的安全结构。
    """

    payload: list[dict[str, Any]] = []
    for item in recalls:
        memory = item.memory
        payload.append(
            {
                "title": truncate_text(memory.title, 60),
                "summary": truncate_text(memory.summary, 200),
                "key_facts": memory.key_facts[:5],
                "user_goals": memory.user_goals[:3],
                "legal_concepts": memory.legal_concepts[:6],
                "updated_at": memory.updated_at,
            }
        )
    return payload


def safe_turn_count(value: Any) -> int:
    """
    安全转换轮次计数，异常输入回落为 0。
    """

    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "DEFAULT_RECALL_TOP_K",
    "MEMORY_SCHEMA_VERSION",
    "LegalCaseMemory",
    "MemoryStore",
    "MemoryStoreError",
    "RecalledCaseMemory",
    "build_case_memory_from_snapshot",
    "extract_memory_keywords",
    "recalled_memories_to_prompt_payload",
]
