"""
法条检索业务封装。

LegalArticleRetriever 把“查询向量化、Chroma 召回、metadata 过滤、相邻条文补全、引用格式化”封装成
一个面向 Agent 工具调用的稳定接口。Agent 层只需要调用 search_legal_articles，不需要知道底层向量库细节。
"""

from __future__ import annotations

import re
from typing import Any

from agent_system.config import (
    load_chroma_config,
    load_embedding_config,
    load_retrieval_config,
)
from agent_system.retrieval.chroma_store import LegalChromaStore
from agent_system.retrieval.embedding import LocalBGEEmbeddingModel
from agent_system.retrieval.legal_preprocess import LegalArticleDocument, build_legal_documents


class LegalArticleRetriever:
    """
    法条语义检索器。

    该类在初始化时只保存配置，embedding 模型会在首次查询时懒加载。
    这样做的原因是 BGE-M3 模型较大，普通工具调用 demo 启动时不应该立刻加载模型，
    只有真正检索法条时才承担模型加载成本。
    """

    def __init__(self) -> None:
        """
        初始化法条检索器配置。
        """

        self.embedding_config = load_embedding_config()
        self.chroma_config = load_chroma_config()
        self.retrieval_config = load_retrieval_config()
        self._embedder: LocalBGEEmbeddingModel | None = None
        self._store: LegalChromaStore | None = None
        self._validated = False
        self._document_index: dict[int, LegalArticleDocument] | None = None

    def preload(self, *, include_keyword_index: bool = True) -> None:
        """
        预加载法条检索所需的本地资源。

        Args:
            include_keyword_index: 是否同时预构建关键词检索索引。

        说明：
            默认检索器采用懒加载，是为了让普通脚本启动更快。但交互式法律咨询 CLI 更需要“第一轮
            用户输入后立即可用”，所以这里提供显式预热入口，把 BGE-M3、Chroma collection 校验和
            可选关键词索引提前完成。
        """

        self._ensure_ready()
        if include_keyword_index:
            self._get_document_index()

    def search_legal_articles(
        self,
        *,
        query: str,
        top_k: int | None = None,
        legal_name: str = "",
        category: str = "",
        article_no: str = "",
        source_type: str = "law_article",
        include_neighbors: bool | None = None,
    ) -> dict[str, Any]:
        """
        检索相关法条。

        Args:
            query: 用户问题或检索关键词。
            top_k: 返回结果数量。为空时使用配置默认值。
            legal_name: 可选法律名称过滤，例如“劳动合同法”。
            category: 可选分类过滤，例如“社会法”。
            article_no: 可选条号过滤，例如“第八十二条”。
            source_type: 内容类型过滤，默认只查正式法条。
            include_neighbors: 是否返回同一标题下的相邻条文。

        Returns:
            dict[str, Any]: 适合直接返回给 Agent 的结构化检索结果。
        """

        query = query.strip()
        if not query:
            return {"ok": False, "error": "query 不能为空。", "results": []}

        limit = self._normalize_top_k(top_k)
        include_neighbors = (
            self.retrieval_config.include_neighbors_default
            if include_neighbors is None
            else bool(include_neighbors)
        )
        source_type = source_type.strip() or self.retrieval_config.default_source_type
        article_no = normalize_article_no(article_no)

        self._ensure_ready()
        assert self._embedder is not None
        assert self._store is not None

        where = build_where_filter(
            legal_name=legal_name,
            category=category,
            article_no=article_no,
            source_type=source_type,
        )
        candidate_k = max(limit, self.retrieval_config.candidate_k)
        query_embedding = self._embedder.embed_query(query)
        raw_result = self._store.query_by_embedding(
            query_embedding=query_embedding,
            n_results=candidate_k,
            where=where,
        )

        results = self._format_results(raw_result, top_k=limit, include_neighbors=include_neighbors)
        return {
            "ok": True,
            "query": query,
            "top_k": limit,
            "where": where or {},
            "results": results,
        }

    def search_legal_articles_by_keyword(
        self,
        *,
        keywords: str | list[Any],
        top_k: int | None = None,
        legal_name: str = "",
        category: str = "",
        article_no: str = "",
        source_type: str = "law_article",
        include_neighbors: bool | None = None,
        match_mode: str = "all",
    ) -> dict[str, Any]:
        """
        按关键词精确匹配法条。

        Args:
            keywords: 关键词字符串或关键词列表。字符串中可用空格、逗号、顿号或分号分隔多个关键词。
            top_k: 返回结果数量。为空时使用配置默认值。
            legal_name: 可选法律名称过滤，例如“劳动合同法”。
            category: 可选分类过滤，例如“社会法”。
            article_no: 可选条号过滤，例如“第八十二条”。
            source_type: 内容类型过滤，默认只查正式法条。
            include_neighbors: 是否返回同一标题下的相邻条文。
            match_mode: 关键词匹配模式。all 表示必须命中全部关键词，any 表示命中任一关键词即可。

        Returns:
            dict[str, Any]: 适合直接返回给 Agent 的结构化关键词匹配结果。
        """

        normalized_keywords = normalize_keywords(keywords)
        if not normalized_keywords:
            return {"ok": False, "error": "keywords 不能为空。", "results": []}

        limit = self._normalize_top_k(top_k)
        include_neighbors = (
            self.retrieval_config.include_neighbors_default
            if include_neighbors is None
            else bool(include_neighbors)
        )
        source_type = source_type.strip() or self.retrieval_config.default_source_type
        article_no = normalize_article_no(article_no)
        normalized_match_mode = "any" if match_mode == "any" else "all"

        document_index = self._get_document_index()
        candidates: list[tuple[int, int, dict[str, Any]]] = []

        for document in sorted(
            document_index.values(),
            key=lambda item: safe_int(item.metadata.get("record_index"), default=0),
        ):
            metadata = document.metadata
            if not metadata_matches_filters(
                metadata,
                legal_name=legal_name,
                category=category,
                article_no=article_no,
                source_type=source_type,
            ):
                continue

            matched_keywords = match_keywords_in_document(document, normalized_keywords)
            matched_count = len(matched_keywords)
            if normalized_match_mode == "all" and matched_count != len(normalized_keywords):
                continue
            if normalized_match_mode == "any" and matched_count == 0:
                continue

            matched_field_count = sum(len(match["fields"]) for match in matched_keywords)
            result = format_result_item(
                rank=0,
                item_id=document.id,
                metadata=dict(metadata),
                document=document.document,
                distance=None,
            )
            result["score"] = round(matched_count / len(normalized_keywords), 4)
            result["matched_keywords"] = matched_keywords
            result["match_count"] = matched_count
            result["match_mode"] = normalized_match_mode
            if include_neighbors:
                result["neighbors"] = self._build_neighbors(dict(metadata))
            candidates.append((matched_count, matched_field_count, result))

        candidates.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                safe_int(item[2].get("record_index"), default=0),
            )
        )
        results = [item[2] for item in candidates[:limit]]
        for index, result in enumerate(results, start=1):
            result["rank"] = index

        return {
            "ok": True,
            "keywords": normalized_keywords,
            "match_mode": normalized_match_mode,
            "top_k": limit,
            "where": build_where_filter(
                legal_name=legal_name,
                category=category,
                article_no=article_no,
                source_type=source_type,
            )
            or {},
            "results": results,
        }

    def _ensure_ready(self) -> None:
        """
        懒加载 embedding 模型和 Chroma store，并校验 collection 元信息。
        """

        if self._store is None:
            self._store = LegalChromaStore(self.chroma_config)

        if not self._validated:
            self._store.validate_collection(
                embedding_model=self.embedding_config.model_name,
                schema_version=self.chroma_config.schema_version,
                source_type=self.retrieval_config.default_source_type,
            )
            self._validated = True

        if self._embedder is None:
            self._embedder = LocalBGEEmbeddingModel(self.embedding_config)

    def _normalize_top_k(self, top_k: int | None) -> int:
        """
        规范化 top_k，避免过小或过大导致工具输出不可控。

        Args:
            top_k: 用户或模型传入的结果数量。

        Returns:
            int: 限制在合理范围内的结果数量。
        """

        if top_k is None:
            top_k = self.retrieval_config.top_k
        try:
            value = int(top_k)
        except (TypeError, ValueError):
            value = self.retrieval_config.top_k
        return max(1, min(value, self.retrieval_config.max_top_k))

    def _format_results(
        self,
        raw_result: dict[str, Any],
        *,
        top_k: int,
        include_neighbors: bool,
    ) -> list[dict[str, Any]]:
        """
        将 Chroma 原始结果转为 Agent 友好的列表。

        Args:
            raw_result: Chroma query 返回值。
            top_k: 最终返回数量。
            include_neighbors: 是否补充相邻条文。

        Returns:
            list[dict[str, Any]]: 检索结果列表。
        """

        ids = first_result_list(raw_result.get("ids"))
        documents = first_result_list(raw_result.get("documents"))
        metadatas = first_result_list(raw_result.get("metadatas"))
        distances = first_result_list(raw_result.get("distances"))

        results: list[dict[str, Any]] = []
        for index, item_id in enumerate(ids[:top_k]):
            metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            document = documents[index] if index < len(documents) else ""
            distance = distances[index] if index < len(distances) else None
            result = format_result_item(
                rank=index + 1,
                item_id=str(item_id),
                metadata=metadata,
                document=str(document),
                distance=distance,
            )
            if include_neighbors:
                result["neighbors"] = self._build_neighbors(metadata)
            results.append(result)

        return results

    def _get_document_index(self) -> dict[int, LegalArticleDocument]:
        """
        获取按 record_index 建立的法条文档索引。

        Returns:
            dict[int, LegalArticleDocument]: record_index 到法条文档的映射。
        """

        if self._document_index is None:
            # 关键词匹配和相邻条文都只需要原始法条文本，不需要向量模型。
            # 因此这里直接从原始 JSON 构建轻量索引，避免为了精确匹配额外加载 BGE-M3。
            documents = build_legal_documents(
                self.chroma_config.data_path,
                include_source_types=(self.retrieval_config.default_source_type,),
            )
            self._document_index = {
                int(document.metadata["record_index"]): document for document in documents
            }
        return self._document_index

    def _build_neighbors(self, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """
        根据 metadata 中的 prev/next record_index 构造相邻条文。

        Args:
            metadata: 命中条文的 metadata。

        Returns:
            list[dict[str, Any]]: 相邻条文列表。
        """

        document_index = self._get_document_index()
        neighbors: list[dict[str, Any]] = []
        for direction, key in (("prev", "prev_record_index"), ("next", "next_record_index")):
            record_index = safe_int(metadata.get(key), default=-1)
            if record_index < 0:
                continue

            neighbor = document_index.get(record_index)
            if neighbor is None:
                continue

            neighbors.append(
                {
                    "direction": direction,
                    "id": neighbor.id,
                    "record_index": neighbor.metadata.get("record_index", -1),
                    "category": neighbor.metadata.get("category", ""),
                    "legal_name": neighbor.metadata.get("legal_name", ""),
                    "title": neighbor.metadata.get("title", ""),
                    "article_no": neighbor.metadata.get("article_no", ""),
                    "text": neighbor.metadata.get("text", ""),
                    "citation": neighbor.metadata.get("citation", ""),
                }
            )

        return neighbors


def build_legal_retriever() -> LegalArticleRetriever:
    """
    创建默认法条检索器。

    Returns:
        LegalArticleRetriever: 已读取项目配置的检索器实例。
    """

    return LegalArticleRetriever()


def build_where_filter(
    *,
    legal_name: str = "",
    category: str = "",
    article_no: str = "",
    source_type: str = "law_article",
) -> dict[str, Any] | None:
    """
    构造 Chroma metadata where 过滤条件。

    Args:
        legal_name: 法律名称过滤。
        category: 分类过滤。
        article_no: 条号过滤。
        source_type: 内容类型过滤。

    Returns:
        dict[str, Any] | None: Chroma where 条件；没有任何条件时返回 None。
    """

    conditions: list[dict[str, Any]] = []
    if source_type.strip():
        conditions.append({"source_type": source_type.strip()})
    if legal_name.strip():
        conditions.append({"legal_name": legal_name.strip()})
    if category.strip():
        conditions.append({"category": category.strip()})
    if article_no.strip():
        conditions.append({"article_no": article_no.strip()})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


_KEYWORD_SEPARATOR_RE = re.compile(r"[\s,，、;；]+")


def normalize_article_no(article_no: str) -> str:
    """
    规范化用户输入的条号。

    Args:
        article_no: 用户或模型传入的条号。

    Returns:
        str: 与 metadata 中 article_no 尽量一致的条号。
    """

    value = article_no.strip()
    if not value:
        return ""
    if value.startswith("第"):
        return value
    if value.endswith("条"):
        return f"第{value}"
    return value


def normalize_keywords(keywords: str | list[Any]) -> list[str]:
    """
    规范化关键词输入。

    Args:
        keywords: 关键词字符串或关键词列表。字符串中可用空格、逗号、顿号或分号分隔。

    Returns:
        list[str]: 去重后的关键词列表，保持首次出现顺序。
    """

    if isinstance(keywords, str):
        raw_parts = _KEYWORD_SEPARATOR_RE.split(keywords.strip())
    elif isinstance(keywords, list):
        raw_parts = [str(item).strip() for item in keywords]
    else:
        raw_parts = [str(keywords).strip()]

    normalized: list[str] = []
    seen: set[str] = set()
    for part in raw_parts:
        keyword = part.strip()
        if not keyword or keyword in seen:
            continue
        seen.add(keyword)
        normalized.append(keyword)
    return normalized


def metadata_matches_filters(
    metadata: dict[str, Any],
    *,
    legal_name: str,
    category: str,
    article_no: str,
    source_type: str,
) -> bool:
    """
    判断法条 metadata 是否满足工具过滤条件。

    Args:
        metadata: 法条 metadata。
        legal_name: 法律名称过滤。
        category: 分类过滤。
        article_no: 条号过滤。
        source_type: 内容类型过滤。

    Returns:
        bool: 满足所有非空过滤条件时返回 True。
    """

    filters = {
        "legal_name": legal_name.strip(),
        "category": category.strip(),
        "article_no": article_no.strip(),
        "source_type": source_type.strip(),
    }
    for key, expected in filters.items():
        if expected and str(metadata.get(key, "")) != expected:
            return False
    return True


def match_keywords_in_document(
    document: LegalArticleDocument,
    keywords: list[str],
) -> list[dict[str, Any]]:
    """
    在法条文档的多个字段中匹配关键词。

    Args:
        document: 法条文档。
        keywords: 已规范化的关键词列表。

    Returns:
        list[dict[str, Any]]: 每个命中关键词及其命中字段。
    """

    metadata = document.metadata
    field_texts = {
        "legal_name": str(metadata.get("legal_name", "")),
        "title": str(metadata.get("title", "")),
        "article_no": str(metadata.get("article_no", "")),
        "text": str(metadata.get("text", "")),
        "document": document.document,
    }

    matches: list[dict[str, Any]] = []
    for keyword in keywords:
        fields = [field for field, text in field_texts.items() if keyword in text]
        if fields:
            matches.append({"keyword": keyword, "fields": fields})
    return matches


def first_result_list(value: Any) -> list[Any]:
    """
    读取 Chroma query 返回值中的第一组结果。

    Args:
        value: Chroma 返回的二维列表字段。

    Returns:
        list[Any]: 第一组结果；字段缺失时返回空列表。
    """

    if not isinstance(value, list) or not value:
        return []
    first = value[0]
    if not isinstance(first, list):
        return []
    return first


def format_result_item(
    *,
    rank: int,
    item_id: str,
    metadata: dict[str, Any],
    document: str,
    distance: Any,
) -> dict[str, Any]:
    """
    格式化单条检索结果。

    Args:
        rank: 排名。
        item_id: Chroma 文档 ID。
        metadata: Chroma metadata。
        document: Chroma document 文本。
        distance: Chroma 距离值。

    Returns:
        dict[str, Any]: Agent 可读的检索结果。
    """

    legal_name = str(metadata.get("legal_name", ""))
    article_no = str(metadata.get("article_no", ""))
    citation = str(metadata.get("citation") or f"《{legal_name}》{article_no}")
    distance_value = safe_float(distance)

    return {
        "rank": rank,
        "id": item_id,
        "score": cosine_score(distance_value),
        "distance": distance_value,
        "record_index": safe_int(metadata.get("record_index"), default=-1),
        "category": str(metadata.get("category", "")),
        "legal_name": legal_name,
        "title": str(metadata.get("title", "")),
        "version_date": str(metadata.get("version_date", "")),
        "article_no": article_no,
        "source_type": str(metadata.get("source_type", "")),
        "text": str(metadata.get("text", "")),
        "document": document,
        "citation": citation,
    }


def cosine_score(distance: float | None) -> float | None:
    """
    将 cosine distance 粗略转换为 0~1 分数。

    Args:
        distance: Chroma 返回的距离。

    Returns:
        float | None: 分数，越大表示越相似。
    """

    if distance is None:
        return None
    return round(max(0.0, min(1.0, 1.0 - distance)), 4)


def safe_float(value: Any) -> float | None:
    """
    安全转换 float。

    Args:
        value: 待转换值。

    Returns:
        float | None: 转换失败时返回 None。
    """

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any, *, default: int = 0) -> int:
    """
    安全转换 int。

    Args:
        value: 待转换值。
        default: 转换失败时使用的默认值。

    Returns:
        int: 转换结果。
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return default
