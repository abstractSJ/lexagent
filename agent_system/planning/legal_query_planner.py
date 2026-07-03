"""
法律案情检索规划器。

本模块负责把一段原始案情文本拆解为若干“单个法律事项”，并为每个事项生成多条适合后续
向量检索的中文 query。它的职责是做“检索前规划”，而不是直接回答法律问题。

设计原则：
1. 只做事实拆解和 query 组生成，不做违法性判断或责任结论判断。
2. 输出 JSON 只是中间计划，真正送入向量库的仍然是一条条普通字符串 query。
3. 尽量高召回：宁可保留可能相关事项，也不要在检索前过早排除。
4. 尽量防止模型臆测：禁止输出具体条号、刑期、罚金、没收财产等检索前结论。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Protocol
import re

from agent_system.config import LLMCallOptions, load_llm_config
from agent_system.llm.messages import Message, system_message, user_message
from agent_system.llm.openai_client import OpenAIChatClient


# 使用独立低温配置来做结构化检索规划。
# 原因是这里追求的是字段稳定和 JSON 可解析性，而不是对话时的表达丰富度。
PLANNER_TEMPERATURE = 0.1
PLANNER_REASONING_EFFORT = "low"
DEFAULT_MAX_REPAIR_ATTEMPTS = 1
DEFAULT_PLANNER_LLM_OPTIONS = LLMCallOptions(
    temperature=PLANNER_TEMPERATURE,
    reasoning_effort=PLANNER_REASONING_EFFORT,
)

# 这些上限是为了控制一次规划的输出规模，避免模型给出过长、重复或无法消费的计划。
MAX_GLOBAL_QUERIES = 5
MAX_ISSUES = 8
MAX_FACTS_PER_ISSUE = 6
MAX_PREFERRED_LEGAL_NAMES = 5
MAX_QUERIES_PER_ISSUE = 5
MAX_TERMS_PER_ISSUE = 8
MIN_RECOMMENDED_QUERIES = 3

# 这一类表达代表模型已经在检索前提前下了法律结论，会污染后续检索方向。
# 因此这里做本地拒绝或剔除，宁可让模型重试，也不允许把“结论”当成 query 送给向量库。
ARTICLE_NO_PATTERN = re.compile(r"第[一二三四五六七八九十百千万零〇\d]+条")
FORBIDDEN_QUERY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"不构成犯罪|构成犯罪"), "包含检索前犯罪结论"),
    (re.compile(r"属于违法|不属于违法|违法行为|不违法"), "包含检索前违法结论"),
    (re.compile(r"有期徒刑|无期徒刑|死刑|拘役|管制"), "包含模型猜测的刑罚结果"),
    (re.compile(r"罚金|没收财产|缓刑"), "包含模型猜测的处罚结果"),
    (re.compile(r"赔偿\s*\d+[元块万]"), "包含模型猜测的具体金额结果"),
]

LEGAL_QUERY_PLANNER_SYSTEM_PROMPT = """
你是“中华人民共和国法律检索规划器”。

你的唯一任务，是把用户输入的案情文本拆解为若干“可能需要检索的单个法律事项”，并为每个事项生成一组适合后续向量检索的中文 query。

你不是法律裁判者，不负责判断行为是否违法、是否构成犯罪、应适用哪一条法、是否成立责任，也不负责给出法律意见。

你必须严格遵守以下规则：

一、工作目标
1. 你的目标是提高后续法条检索的召回率与准确率。
2. 你要尽量把案情中的不同法律事项拆开，而不是把多个事项揉成一个 query。
3. 只要某个事实可能与权利、义务、责任、赔偿、处罚、身份、财产、程序、婚姻、劳动、合同、公司、行政、刑事有关，就应当生成检索事项。
4. 即使你不确定该事实是否违法，也必须保留并生成检索事项，不得因为“不确定”而省略。

二、禁止事项
1. 不得补充、猜测或改写用户未提供的事实。
2. 不得预判具体法条条号。
3. 不得预判具体罪名、违法结论、法律责任结论。
4. 不得预判具体处罚结果，例如“三年以下有期徒刑”“罚金”“没收财产”等。
5. 不得使用英美法或其他国家法律体系来判断问题，只能面向中华人民共和国现行法律体系做检索规划。
6. 不得输出解释性文字、markdown、代码块、前后说明，只能输出合法 JSON。

三、query 生成规则
1. 每个事项生成 3 到 5 条互补 query。
2. query 应尽量简短、清晰、单一事项化，不要完整复述整段案情。
3. query 应优先使用以下几类表达：
   - 事实式表达：直接描述行为或事实
   - 概念式表达：概括成法律概念，但不能编造不存在的法律概念
   - 问题式表达：保留用户“怎么处理/怎么处罚/怎么办”的提问形式
   - 安全规范表达：尽量贴近中国法条常见表述，但不得加入你猜测的处罚结果
4. 同一事项的多条 query 中，至少有一条使用规范法律用语改写口语表述（例如“没签合同”改写为“未订立书面劳动合同”，“被辞退”改写为“解除劳动合同”），因为向量库中的法条使用规范表述，规范化 query 的召回质量明显更高；同时保留一条贴近用户原话的 query 兜底。
5. query 中可以包含“刑事责任”“民事责任”“赔偿责任”“处理”“分割”等责任类型词，但不得包含你猜测出的刑期、罚金、没收财产等结果词。
6. 如果能够较高把握地判断优先法律名称，可填写 preferred_legal_names；否则返回空数组。
7. positive_terms 只写能够帮助后续重排的核心正向词。
8. negative_terms 只在存在明显混淆项时填写；如果没有明显混淆项，返回空数组。

四、输出格式
你必须输出一个 JSON object，结构如下：
{
  "global_queries": ["..."],
  "issues": [
    {
      "issue": "单个法律事项名称",
      "facts": ["原始事实或最小规范化事实"],
      "preferred_legal_names": ["法律名称1", "法律名称2"],
      "queries": ["query1", "query2", "query3"],
      "positive_terms": ["term1", "term2"],
      "negative_terms": ["term1", "term2"]
    }
  ]
}

五、质量要求
1. 宁可多保留可能相关事项，也不要过早排除。
2. query 之间要有互补性，避免完全重复。
3. 每个事项必须保持边界清晰，不要把多个事项混成一个事项。
4. 输出必须是严格合法 JSON。
""".strip()


class SupportsComplete(Protocol):
    """
    支持单次 complete() 调用的最小 LLM 协议。

    使用 Protocol 的原因，是让测试代码可以传入 fake LLM，而不必真的初始化远程客户端。
    """

    def complete(
        self,
        messages: list[Message],
        *,
        options: LLMCallOptions | None = None,
    ) -> str:
        """
        接收 Chat-style messages 并返回完整文本。
        """


@dataclass(frozen=True)
class LegalIssueQuery:
    """
    单个法律事项的检索计划。

    Attributes:
        issue: 事项标题，用中性事实语言描述待检索的问题，不作违法性结论。
        facts: 与该事项相关的关键事实列表，只保留原始案情中已有事实。
        preferred_legal_names: 优先检索的法律名称列表；不确定时可以为空列表。
        queries: 面向向量检索的自然语言查询列表，每条 query 都会被逐条向量化检索。
        positive_terms: 建议在后续重排中加权的正向关键词。
        negative_terms: 建议在后续重排中降权或区分的反向关键词。
    """

    issue: str
    facts: list[str]
    preferred_legal_names: list[str]
    queries: list[str]
    positive_terms: list[str]
    negative_terms: list[str]


@dataclass(frozen=True)
class LegalQueryPlan:
    """
    一段案情对应的完整检索计划。

    Attributes:
        global_queries: 面向整段案情的兜底 query 列表。
        issues: 拆解得到的单个法律事项列表。
        warnings: 解析或规范化过程中产生的非致命提示。
        raw_response: LLM 原始输出文本，便于调试 prompt 效果。
    """

    global_queries: list[str]
    issues: list[LegalIssueQuery]
    warnings: list[str]
    raw_response: str = ""


class LegalQueryPlanError(ValueError):
    """
    法律 query planner 生成或校验失败。
    """

    def __init__(
        self,
        message: str,
        *,
        raw_response: str = "",
        errors: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.errors = list(errors or [])


class LegalQueryPlanner:
    """
    一次性法律案情检索规划器。

    该类是无状态工具：输入一段案情，输出一份结构化 query plan。
    它不保存多轮历史，也不直接执行向量检索。

    Args:
        llm: 支持 complete(messages, options=...) 的 LLM 客户端；为空时使用项目默认 LLM 客户端。
        max_repair_attempts: JSON 解析或字段校验失败后的修复重试次数。
    """

    def __init__(
        self,
        llm: SupportsComplete | None = None,
        *,
        max_repair_attempts: int = DEFAULT_MAX_REPAIR_ATTEMPTS,
        llm_options: LLMCallOptions | None = None,
    ) -> None:
        """
        初始化 query planner。

        Args:
            llm: 可选 LLM 客户端。
            max_repair_attempts: 最大修复重试次数。
            llm_options: 可选单次调用覆盖参数。为 None 时使用规划器默认低温参数。
        """

        self.llm = llm or build_query_planner_llm()
        self.max_repair_attempts = max(0, int(max_repair_attempts))
        self.llm_options = llm_options or DEFAULT_PLANNER_LLM_OPTIONS

    def plan(self, case_text: str) -> LegalQueryPlan:
        """
        根据案情文本生成检索计划。

        Args:
            case_text: 原始案情文本。

        Returns:
            LegalQueryPlan: 结构化检索计划。

        Raises:
            LegalQueryPlanError: LLM 输出无法解析、字段严重缺失或修复后仍不合法时抛出。
        """

        case_text = normalize_text(case_text)
        if not case_text:
            raise LegalQueryPlanError("case_text 不能为空。")

        messages = [
            system_message(LEGAL_QUERY_PLANNER_SYSTEM_PROMPT),
            user_message(build_planning_user_prompt(case_text)),
        ]

        raw_response = self.llm.complete(messages, options=self.llm_options)
        try:
            data = parse_json_object(raw_response)
            return validate_and_normalize_plan(data, raw_response=raw_response)
        except ValueError as error:
            original_errors = [str(error)]

        if self.max_repair_attempts <= 0:
            raise LegalQueryPlanError(
                "法律 query planner 输出无法解析或校验失败。",
                raw_response=raw_response,
                errors=original_errors,
            )

        latest_raw_response = raw_response
        latest_errors = original_errors
        for _ in range(self.max_repair_attempts):
            repair_messages = [
                system_message(LEGAL_QUERY_PLANNER_SYSTEM_PROMPT),
                user_message(
                    build_repair_user_prompt(
                        case_text=case_text,
                        raw_response=latest_raw_response,
                        errors=latest_errors,
                    )
                ),
            ]
            latest_raw_response = self.llm.complete(
                repair_messages,
                options=self.llm_options,
            )

            try:
                data = parse_json_object(latest_raw_response)
                return validate_and_normalize_plan(data, raw_response=latest_raw_response)
            except ValueError as error:
                latest_errors = [str(error)]

        raise LegalQueryPlanError(
            "法律 query planner 在修复后仍然无法输出合法计划。",
            raw_response=latest_raw_response,
            errors=latest_errors,
        )


def build_query_planner_llm() -> OpenAIChatClient:
    """
    创建法律 query planner 使用的默认 LLM 客户端。

    Returns:
        OpenAIChatClient: 使用项目默认 LLM 配置初始化的客户端。

    说明:
        规划器的低温、低推理强度不再通过复制整份 LLMConfig 实现，
        而是由 LegalQueryPlanner.llm_options 在单次 complete() 调用时覆盖。
        这样能让 planner 与普通聊天、Agent 工具调用共享同一个“单次调用变参”机制。
    """

    return OpenAIChatClient(config=load_llm_config())


def build_planning_user_prompt(case_text: str) -> str:
    """
    构造初次规划时发送给模型的用户提示词。

    Args:
        case_text: 原始案情文本。

    Returns:
        str: 用户提示词。
    """

    return f"""
请根据以下案情生成法律检索计划。

【案情】
{case_text}

【输出要求】
1. 只输出 JSON，不要输出任何解释、前言、结尾或 Markdown 代码块。
2. 不要判断是否违法或是否构成犯罪。
3. 不要猜测条号、罪名、刑期、拘役、管制、罚金、没收财产等具体法律后果。
4. 只做事项拆解和 query 组生成。
5. 若存在多个事项，必须拆开列出。
6. global_queries 保留 2 到 5 条；issues 保留 1 到 8 个；每个 issue 的 queries 保留 3 到 5 条。
7. preferred_legal_names 只有在较高把握时才填写，例如“刑法”“民法典”；不确定时返回空数组。

【目标 JSON 结构】
{{
  "global_queries": ["string"],
  "issues": [
    {{
      "issue": "单个法律事项名称",
      "facts": ["案情事实或最小规范化事实"],
      "preferred_legal_names": ["刑法", "民法典"],
      "queries": ["query1", "query2", "query3"],
      "positive_terms": ["term1"],
      "negative_terms": ["term2"]
    }}
  ]
}}
""".strip()


def build_repair_user_prompt(
    *,
    case_text: str,
    raw_response: str,
    errors: list[str],
) -> str:
    """
    构造 JSON 修复请求。

    Args:
        case_text: 原始案情文本。
        raw_response: 上一次模型输出。
        errors: 本地解析或校验错误列表。

    Returns:
        str: 修复提示词。
    """

    error_text = "\n".join(f"- {item}" for item in errors) if errors else "- 未知错误"

    return f"""
你上一轮输出未通过本地 JSON 解析或字段校验。请只修复输出结构，不要新增法律结论。

【原始案情】
{case_text}

【上一轮输出】
{raw_response}

【本地发现的问题】
{error_text}

【修复要求】
1. 只输出合法 JSON，不要输出任何解释、Markdown 或代码块。
2. 保留“法律检索规划器”的角色，不要给出违法性判断、罪名判断或具体处罚结果。
3. 不要输出具体条号、刑期、拘役、管制、罚金、没收财产等猜测性法律后果。
4. 顶层必须是 object，且必须包含 global_queries 和 issues。
5. 每个 issue 必须包含：issue、facts、preferred_legal_names、queries、positive_terms、negative_terms。
6. 若某条 query 含有具体条号或处罚结果，请改写为更中性的检索 query，而不是删除整个事项。
""".strip()


def extract_json_object(text: str) -> str:
    """
    从模型输出中提取第一个完整 JSON object 文本。

    Args:
        text: LLM 原始输出。

    Returns:
        str: 第一个完整 JSON object 字符串。

    Raises:
        ValueError: 文本中找不到完整 JSON object 时抛出。
    """

    stripped = text.strip()
    if not stripped:
        raise ValueError("LLM 输出为空，无法提取 JSON。")

    # 这里不依赖正则“整块匹配”，而是做括号平衡扫描。
    # 原因是模型偶尔会在 JSON 前后夹带解释文字，用平衡扫描容错更稳。
    start_index = stripped.find("{")
    if start_index < 0:
        raise ValueError("LLM 输出中没有找到 JSON object 起始符号 '{'。")

    depth = 0
    in_string = False
    escaped = False
    object_start = -1

    for index in range(start_index, len(stripped)):
        char = stripped[index]

        if escaped:
            escaped = False
            continue

        if in_string:
            if char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            if depth == 0:
                object_start = index
            depth += 1
            continue

        if char == "}":
            depth -= 1
            if depth == 0 and object_start >= 0:
                return stripped[object_start : index + 1]
            if depth < 0:
                break

    raise ValueError("LLM 输出中的 JSON object 不完整，未找到匹配的结束符号 '}'。")


def parse_json_object(text: str) -> dict[str, Any]:
    """
    解析模型输出中的 JSON object。

    Args:
        text: LLM 原始输出。

    Returns:
        dict[str, Any]: 解析后的对象。

    Raises:
        ValueError: 不是合法 JSON object 时抛出。
    """

    json_text = extract_json_object(text)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"LLM 输出中的 JSON 语法不合法：{error}") from error

    if not isinstance(data, dict):
        raise ValueError("LLM 输出的顶层 JSON 必须是 object。")

    return data


def validate_and_normalize_plan(
    data: dict[str, Any],
    *,
    raw_response: str = "",
) -> LegalQueryPlan:
    """
    校验并规范化模型生成的检索计划。

    Args:
        data: 模型输出经 json.loads 后的对象。
        raw_response: LLM 原始输出文本。

    Returns:
        LegalQueryPlan: 规范化后的检索计划。

    Raises:
        ValueError: 顶层结构或关键字段不合法时抛出。
    """

    warnings: list[str] = []

    if "global_queries" not in data:
        raise ValueError("顶层缺少必填字段 global_queries。")
    if "issues" not in data:
        raise ValueError("顶层缺少必填字段 issues。")

    global_queries = normalize_string_list(
        data.get("global_queries"),
        field_name="global_queries",
        max_items=MAX_GLOBAL_QUERIES,
        warnings=warnings,
    )
    global_queries = filter_safe_queries(
        global_queries,
        field_name="global_queries",
        warnings=warnings,
    )
    if not global_queries:
        warnings.append("global_queries 为空；后续检索将只能依赖 issue 级 query。")

    raw_issues = data.get("issues")
    if not isinstance(raw_issues, list):
        raise ValueError("顶层字段 issues 必须是列表。")
    if not raw_issues:
        raise ValueError("issues 不能为空。")

    issues: list[LegalIssueQuery] = []
    for index, issue_value in enumerate(raw_issues):
        if len(issues) >= MAX_ISSUES:
            warnings.append(f"issues 数量超过上限 {MAX_ISSUES}，后续事项已被截断。")
            break
        issues.append(validate_issue(issue_value, index=index, warnings=warnings))

    if not issues:
        raise ValueError("issues 中没有任何合法事项。")

    return LegalQueryPlan(
        global_queries=global_queries,
        issues=issues,
        warnings=warnings,
        raw_response=raw_response,
    )


def validate_issue(value: Any, *, index: int, warnings: list[str]) -> LegalIssueQuery:
    """
    校验并规范化单个 issue。

    Args:
        value: 原始 issue 对象。
        index: issue 在列表中的位置。
        warnings: 共享警告列表。

    Returns:
        LegalIssueQuery: 规范化后的单个事项。

    Raises:
        ValueError: issue 缺少核心字段或内容严重不合法时抛出。
    """

    if not isinstance(value, dict):
        raise ValueError(f"第 {index + 1} 个 issue 必须是 object。")

    issue = normalize_text(value.get("issue"))
    if not issue:
        raise ValueError(f"第 {index + 1} 个 issue 缺少非空 issue 标题。")

    forbidden_reason = first_forbidden_reason(issue)
    if forbidden_reason:
        raise ValueError(f"第 {index + 1} 个 issue 标题不安全：{forbidden_reason}。")
    if ARTICLE_NO_PATTERN.search(issue):
        raise ValueError(f"第 {index + 1} 个 issue 标题包含具体条号，不符合规划器要求。")

    facts = normalize_string_list(
        value.get("facts"),
        field_name=f"issues[{index}].facts",
        max_items=MAX_FACTS_PER_ISSUE,
        warnings=warnings,
    )
    if not facts:
        warnings.append(f"第 {index + 1} 个 issue 没有保留下任何 facts。")

    preferred_legal_names = normalize_string_list(
        value.get("preferred_legal_names"),
        field_name=f"issues[{index}].preferred_legal_names",
        max_items=MAX_PREFERRED_LEGAL_NAMES,
        warnings=warnings,
    )

    queries = normalize_string_list(
        value.get("queries"),
        field_name=f"issues[{index}].queries",
        max_items=MAX_QUERIES_PER_ISSUE,
        warnings=warnings,
    )
    queries = filter_safe_queries(
        queries,
        field_name=f"issues[{index}].queries",
        warnings=warnings,
    )
    if not queries:
        raise ValueError(f"第 {index + 1} 个 issue 没有任何合法 query。")
    if len(queries) < MIN_RECOMMENDED_QUERIES:
        warnings.append(
            f"第 {index + 1} 个 issue 的合法 query 少于 {MIN_RECOMMENDED_QUERIES} 条，后续召回可能偏弱。"
        )

    positive_terms = normalize_string_list(
        value.get("positive_terms"),
        field_name=f"issues[{index}].positive_terms",
        max_items=MAX_TERMS_PER_ISSUE,
        warnings=warnings,
    )
    positive_terms = filter_safe_terms(
        positive_terms,
        field_name=f"issues[{index}].positive_terms",
        warnings=warnings,
    )
    if not positive_terms:
        warnings.append(f"第 {index + 1} 个 issue 没有任何合法 positive_terms。")

    negative_terms = normalize_string_list(
        value.get("negative_terms"),
        field_name=f"issues[{index}].negative_terms",
        max_items=MAX_TERMS_PER_ISSUE,
        warnings=warnings,
    )
    negative_terms = filter_safe_terms(
        negative_terms,
        field_name=f"issues[{index}].negative_terms",
        warnings=warnings,
    )

    return LegalIssueQuery(
        issue=issue,
        facts=facts,
        preferred_legal_names=preferred_legal_names,
        queries=queries,
        positive_terms=positive_terms,
        negative_terms=negative_terms,
    )


def normalize_string_list(
    value: Any,
    *,
    field_name: str,
    max_items: int,
    warnings: list[str],
) -> list[str]:
    """
    规范化字符串列表字段。

    Args:
        value: 原始字段值。
        field_name: 便于报错和警告定位的字段名。
        max_items: 最大保留项数。
        warnings: 共享警告列表。

    Returns:
        list[str]: 去空、去重、截断后的字符串列表。

    Raises:
        ValueError: 字段不是列表时抛出。
    """

    if not isinstance(value, list):
        raise ValueError(f"字段 {field_name} 必须是列表。")

    normalized: list[str] = []
    seen: set[str] = set()

    for item in value:
        if len(normalized) >= max_items:
            warnings.append(f"字段 {field_name} 超过上限 {max_items}，后续内容已截断。")
            break

        if isinstance(item, (dict, list)):
            warnings.append(f"字段 {field_name} 中存在非字符串复合结构，已忽略。")
            continue

        text = normalize_text(item)
        if not text:
            continue
        if text in seen:
            continue

        normalized.append(text)
        seen.add(text)

    return normalized


def normalize_text(value: Any) -> str:
    """
    规范化单条文本。

    Args:
        value: 原始值。

    Returns:
        str: 去首尾空白并压缩内部空白后的文本。
    """

    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return " ".join(text.split())


def filter_safe_queries(
    queries: list[str],
    *,
    field_name: str,
    warnings: list[str],
) -> list[str]:
    """
    过滤不适合直接送入向量检索的 query。

    Args:
        queries: 原始 query 列表。
        field_name: 字段名，用于警告定位。
        warnings: 共享警告列表。

    Returns:
        list[str]: 保留下来的安全 query 列表。
    """

    safe_queries: list[str] = []
    seen: set[str] = set()

    for query in queries:
        reason = first_forbidden_reason(query)
        if reason:
            warnings.append(f"字段 {field_name} 中的 query 已剔除：{reason}。")
            continue
        if ARTICLE_NO_PATTERN.search(query):
            warnings.append(f"字段 {field_name} 中的 query 已剔除：包含具体条号。")
            continue
        if query in seen:
            continue
        safe_queries.append(query)
        seen.add(query)

    return safe_queries


def filter_safe_terms(
    terms: list[str],
    *,
    field_name: str,
    warnings: list[str],
) -> list[str]:
    """
    过滤不安全的 term 列表。

    Args:
        terms: 原始 term 列表。
        field_name: 字段名。
        warnings: 共享警告列表。

    Returns:
        list[str]: 过滤后的 term 列表。
    """

    safe_terms: list[str] = []
    seen: set[str] = set()

    for term in terms:
        reason = first_forbidden_reason(term)
        if reason:
            warnings.append(f"字段 {field_name} 中的 term 已剔除：{reason}。")
            continue
        if ARTICLE_NO_PATTERN.search(term):
            warnings.append(f"字段 {field_name} 中的 term 已剔除：包含具体条号。")
            continue
        if term in seen:
            continue
        safe_terms.append(term)
        seen.add(term)

    return safe_terms


def first_forbidden_reason(text: str) -> str | None:
    """
    返回文本命中的第一条禁止原因。

    Args:
        text: 待检查文本。

    Returns:
        str | None: 命中时返回原因，否则返回 None。
    """

    for pattern, reason in FORBIDDEN_QUERY_PATTERNS:
        if pattern.search(text):
            return reason
    return None


def plan_to_dict(
    plan: LegalQueryPlan,
    *,
    include_raw_response: bool = False,
) -> dict[str, Any]:
    """
    把 LegalQueryPlan 转换为可 JSON 序列化的字典。

    Args:
        plan: 结构化检索计划。
        include_raw_response: 是否保留 raw_response。

    Returns:
        dict[str, Any]: 适合直接 json.dumps 的字典。
    """

    data = asdict(plan)
    if not include_raw_response:
        data.pop("raw_response", None)
    return data


def plan_legal_queries(
    case_text: str,
    *,
    llm: SupportsComplete | None = None,
    max_repair_attempts: int = DEFAULT_MAX_REPAIR_ATTEMPTS,
    llm_options: LLMCallOptions | None = None,
) -> LegalQueryPlan:
    """
    一次性生成法律 query plan。

    Args:
        case_text: 原始案情文本。
        llm: 可选 LLM 客户端或 fake LLM。
        max_repair_attempts: 解析或校验失败时的修复重试次数。
        llm_options: 可选单次调用覆盖参数。为 None 时使用规划器默认低温参数。

    Returns:
        LegalQueryPlan: 结构化检索计划。
    """

    planner = LegalQueryPlanner(
        llm=llm,
        max_repair_attempts=max_repair_attempts,
        llm_options=llm_options,
    )
    return planner.plan(case_text)


__all__ = [
    "DEFAULT_MAX_REPAIR_ATTEMPTS",
    "LegalIssueQuery",
    "LegalQueryPlan",
    "LegalQueryPlanError",
    "LegalQueryPlanner",
    "LEGAL_QUERY_PLANNER_SYSTEM_PROMPT",
    "build_planning_user_prompt",
    "build_query_planner_llm",
    "build_repair_user_prompt",
    "extract_json_object",
    "parse_json_object",
    "plan_legal_queries",
    "plan_to_dict",
    "validate_and_normalize_plan",
]
