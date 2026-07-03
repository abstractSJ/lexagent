"""
法条检索 Agent 工具。

本模块把 LegalArticleRetriever 包装成 Responses API function tool，供 AgentRunner 统一调度。
工具内部使用懒加载检索器，避免创建 AgentSession 时就加载 BGE-M3 模型。
"""

from __future__ import annotations

from typing import Any

from agent_system.agent.tools import LocalTool
from agent_system.retrieval.legal_retriever import LegalArticleRetriever, build_legal_retriever


def build_legal_tools(retriever: LegalArticleRetriever | None = None) -> list[LocalTool]:
    """
    创建法律检索工具列表。

    Args:
        retriever: 可选法条检索器。测试或特殊入口可以传入自定义实例；为空时首次调用工具再创建。

    Returns:
        list[LocalTool]: 可注册到 ToolRegistry 的工具列表。
    """

    cached_retriever = retriever

    def get_retriever() -> LegalArticleRetriever:
        """
        获取懒加载的法条检索器。

        Returns:
            LegalArticleRetriever: 法条检索器实例。
        """

        nonlocal cached_retriever
        if cached_retriever is None:
            cached_retriever = build_legal_retriever()
        return cached_retriever

    def search_legal_articles(arguments: dict[str, Any]) -> dict[str, Any]:
        """
        执行语义法条检索工具。

        Args:
            arguments: 模型传入的工具参数。

        Returns:
            dict[str, Any]: 检索结果或错误信息。
        """

        query = str(arguments.get("query", "")).strip()
        if not query:
            return {"ok": False, "error": "query 不能为空。", "results": []}

        top_k = parse_optional_int(arguments.get("top_k"))
        include_neighbors = parse_bool(arguments.get("include_neighbors"))
        retriever_instance = get_retriever()
        return retriever_instance.search_legal_articles(
            query=query,
            top_k=top_k,
            legal_name=str(arguments.get("legal_name", "")).strip(),
            category=str(arguments.get("category", "")).strip(),
            article_no=str(arguments.get("article_no", "")).strip(),
            source_type=str(arguments.get("source_type", "law_article")).strip() or "law_article",
            include_neighbors=include_neighbors,
        )

    def search_legal_articles_by_keyword(arguments: dict[str, Any]) -> dict[str, Any]:
        """
        执行关键词精确匹配法条工具。

        Args:
            arguments: 模型传入的工具参数。

        Returns:
            dict[str, Any]: 关键词匹配结果或错误信息。
        """

        keywords = arguments.get("keywords", "")
        top_k = parse_optional_int(arguments.get("top_k"))
        include_neighbors = parse_bool(arguments.get("include_neighbors"))
        retriever_instance = get_retriever()
        return retriever_instance.search_legal_articles_by_keyword(
            keywords=keywords,
            top_k=top_k,
            legal_name=str(arguments.get("legal_name", "")).strip(),
            category=str(arguments.get("category", "")).strip(),
            article_no=str(arguments.get("article_no", "")).strip(),
            source_type=str(arguments.get("source_type", "law_article")).strip() or "law_article",
            include_neighbors=include_neighbors,
            match_mode=str(arguments.get("match_mode", "all")).strip() or "all",
        )

    return [
        LocalTool(
            name="search_legal_articles",
            description=(
                "检索本地 Chroma 法条向量库，返回与法律咨询问题相关的正式法条、条号、原文和引用格式。"
                "用户询问法律依据、赔偿责任、程序、权利义务或具体条文时优先使用。"
                "这是语义检索，适合自然语言问题和同义表达。"
            ),
            parameters=_build_parameters(
                properties={
                    "query": {
                        "type": "string",
                        "description": "用户法律问题或检索关键词，例如：公司不签劳动合同怎么办。",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量，常规法律问题建议 15；复杂行为或多法律关系问题可取 20-30，最大值会由本地配置限制。",
                    },
                    "legal_name": {
                        "type": "string",
                        "description": "可选法律名称过滤，例如：劳动合同法、刑法、公司法；不限制时传空字符串。",
                    },
                    "category": {
                        "type": "string",
                        "description": "可选分类过滤，例如：社会法、民法商法、刑法；不限制时传空字符串。",
                    },
                    "article_no": {
                        "type": "string",
                        "description": "可选条号过滤，例如：第八十二条；不限制时传空字符串。",
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["law_article"],
                        "description": "内容类型，第一版固定传 law_article。",
                    },
                    "include_neighbors": {
                        "type": "boolean",
                        "description": "是否返回相邻条文。问题涉及连续条文、前后定义或例外时传 true。",
                    },
                },
                required=[
                    "query",
                    "top_k",
                    "legal_name",
                    "category",
                    "article_no",
                    "source_type",
                    "include_neighbors",
                ],
            ),
            handler=search_legal_articles,
        ),
        LocalTool(
            name="search_legal_articles_by_keyword",
            description=(
                "按关键词精确匹配本地法条原文和元数据，用来补足语义 RAG 可能漏召回的条文。"
                "当用户给出明确关键词、法条原文片段、法律名、条号，或语义检索结果不够直接时使用。"
                "可用 match_mode=all 要求全部关键词命中，也可用 any 扩大召回。"
            ),
            parameters=_build_parameters(
                properties={
                    "keywords": {
                        "type": "string",
                        "description": "关键词字符串，多个关键词用空格、逗号、顿号或分号分隔，例如：未签 劳动合同 二倍工资。",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量，常规取 10-15；需要扩大关键词兜底召回时可取 20-30，最大值会由本地配置限制。",
                    },
                    "legal_name": {
                        "type": "string",
                        "description": "可选法律名称过滤，例如：劳动合同法；不限制时传空字符串。",
                    },
                    "category": {
                        "type": "string",
                        "description": "可选分类过滤，例如：社会法、民法商法、刑法；不限制时传空字符串。",
                    },
                    "article_no": {
                        "type": "string",
                        "description": "可选条号过滤，例如：第八十二条；不限制时传空字符串。",
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["law_article"],
                        "description": "内容类型，第一版固定传 law_article。",
                    },
                    "include_neighbors": {
                        "type": "boolean",
                        "description": "是否返回相邻条文。问题涉及连续条文、前后定义或例外时传 true。",
                    },
                    "match_mode": {
                        "type": "string",
                        "enum": ["all", "any"],
                        "description": "all 表示全部关键词必须命中，精确但可能少；any 表示任一关键词命中，召回更宽。",
                    },
                },
                required=[
                    "keywords",
                    "top_k",
                    "legal_name",
                    "category",
                    "article_no",
                    "source_type",
                    "include_neighbors",
                    "match_mode",
                ],
            ),
            handler=search_legal_articles_by_keyword,
        ),
    ]


def parse_optional_int(value: Any) -> int | None:
    """
    安全解析可选整数。

    Args:
        value: 待解析值。

    Returns:
        int | None: 解析成功时返回整数，空值或非法值返回 None。
    """

    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_bool(value: Any) -> bool:
    """
    安全解析布尔值。

    Args:
        value: 待解析值。

    Returns:
        bool: 解析后的布尔值。
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "是", "需要"}
    return bool(value)


def _build_parameters(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """
    构造严格 JSON Schema 参数对象。

    Args:
        properties: 参数属性定义。
        required: 必填字段列表。

    Returns:
        dict[str, Any]: Responses function tool 使用的 JSON Schema。
    """

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
