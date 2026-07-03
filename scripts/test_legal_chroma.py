"""
检验本地 Chroma 法条向量库。

运行方式：
    python scripts/test_legal_chroma.py
    python scripts/test_legal_chroma.py --metadata-only
    python scripts/test_legal_chroma.py --query "公司不签劳动合同怎么办" --legal-name "劳动合同法" --top-k 15

这个脚本只测试本地向量库和本地 embedding 检索链路，不调用 LLM，也不会请求外部大模型接口。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    # 允许用户直接通过 `python scripts/test_legal_chroma.py` 运行脚本。
    # 原因是脚本位于 scripts 子目录，直接运行时 Python 默认不会把项目根目录加入模块搜索路径。
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_system.config import (  # noqa: E402
    load_chroma_config,
    load_embedding_config,
    load_retrieval_config,
)
from agent_system.retrieval.chroma_store import LegalChromaStore  # noqa: E402
from agent_system.retrieval.legal_retriever import LegalArticleRetriever  # noqa: E402


# 测试脚本默认也返回较多条文，保持与 Agent 工具的检索策略一致。
# 原因是法律问题往往需要 LLM 同时看到核心条文、相关条文和可能的排除项，过少结果容易让模型漏判。
DEFAULT_TEST_TOP_K = 15


@dataclass(frozen=True)
class QueryCase:
    """
    单个向量检索测试用例。

    Attributes:
        query: 用户问题或检索关键词。
        legal_name: 可选法律名称过滤；为空表示不限制法律名称。
        category: 可选分类过滤；为空表示不限制分类。
        article_no: 可选条号过滤；为空表示不限制条号。
        top_k: 返回结果数量。
        include_neighbors: 是否展示命中条文的相邻条文。
    """

    query: str
    legal_name: str = ""
    category: str = ""
    article_no: str = ""
    top_k: int = DEFAULT_TEST_TOP_K
    include_neighbors: bool = False


def main() -> None:
    """
    执行 Chroma collection 元信息校验、样本文档查看和语义检索测试。
    """

    args = parse_args()
    chroma_config = load_chroma_config()
    embedding_config = load_embedding_config()
    retrieval_config = load_retrieval_config()

    print("===== 本地 Chroma 法条向量库测试 =====")
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"持久化目录：{chroma_config.persist_directory}")
    print(f"Collection：{chroma_config.collection_name}")
    print(f"Embedding 模型：{embedding_config.model_name}")
    print(f"Embedding 设备：{embedding_config.device or 'auto'}")
    print(f"默认 top_k：{retrieval_config.top_k}")

    store = LegalChromaStore(chroma_config)
    collection = store.get_collection()

    print_collection_summary(collection)
    print_sample_documents(collection, sample_size=args.sample_size)
    validate_collection(store, embedding_config.model_name, chroma_config.schema_version, retrieval_config.default_source_type)

    if args.metadata_only:
        print("\n===== 元信息测试完成 =====")
        print("已完成 collection 读取、文档数量检查、metadata 完整性校验；未加载 BGE-M3，因此没有执行语义检索。")
        return

    cases = build_query_cases(args)
    run_query_cases(cases)

    print("\n===== 测试完成 =====")
    print("如果上面的结果能返回相关法条，说明 Chroma 持久化库、BGE-M3 查询向量化和检索封装都可以正常工作。")


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 用户传入的测试参数。
    """

    parser = argparse.ArgumentParser(
        description="检验本地 Chroma 法条向量库是否完整、可读、可语义检索。"
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="只检查 collection 元信息和样本文档，不加载 BGE-M3，不执行语义检索。",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="展示 collection 中前几条样本文档，默认 3 条。",
    )
    parser.add_argument(
        "--query",
        default="",
        help="自定义检索问题；不传时运行内置的几个 smoke test。",
    )
    parser.add_argument(
        "--legal-name",
        default="",
        help="可选法律名称过滤，例如：劳动合同法、刑法、民法典。",
    )
    parser.add_argument(
        "--category",
        default="",
        help="可选分类过滤，例如：社会法、刑法、民法商法。",
    )
    parser.add_argument(
        "--article-no",
        default="",
        help="可选条号过滤，例如：第八十二条。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TEST_TOP_K,
        help=f"每个查询返回的法条数量，默认 {DEFAULT_TEST_TOP_K} 条；复杂问题可手动调到 20-30 条。",
    )
    parser.add_argument(
        "--include-neighbors",
        action="store_true",
        help="同时展示命中条文的相邻条文，便于检查连续条文上下文。",
    )
    return parser.parse_args()


def print_collection_summary(collection: Any) -> None:
    """
    打印 collection 的基本状态。

    Args:
        collection: Chroma collection 对象。
    """

    metadata = collection.metadata or {}

    print("\n===== Collection 基本信息 =====")
    print(f"文档数量：{collection.count()}")
    print(f"build_complete：{metadata.get('build_complete')}")
    print(f"document_count(metadata)：{metadata.get('document_count')}")
    print(f"embedding_model：{metadata.get('embedding_model')}")
    print(f"schema_version：{metadata.get('schema_version')}")
    print(f"source_type：{metadata.get('source_type')}")
    print(f"source_file_hash：{metadata.get('source_file_hash')}")
    print(f"build_time：{metadata.get('build_time')}")


def print_sample_documents(collection: Any, *, sample_size: int) -> None:
    """
    从 collection 中取少量原始样本文档，确认持久化数据确实可读。

    Args:
        collection: Chroma collection 对象。
        sample_size: 展示样本文档数量。
    """

    sample_size = max(0, sample_size)
    if sample_size == 0:
        return

    count = collection.count()
    if count == 0:
        print("\n===== 样本文档 =====")
        print("collection 为空，无法展示样本文档。")
        return

    sample = collection.get(
        limit=min(sample_size, count),
        include=["documents", "metadatas"],
    )
    ids = sample.get("ids") or []
    documents = sample.get("documents") or []
    metadatas = sample.get("metadatas") or []

    print("\n===== 样本文档 =====")
    for index, item_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        document = str(documents[index]) if index < len(documents) else ""
        print(f"\n[{index + 1}] id={item_id}")
        print(f"    引用：{metadata.get('citation', '')}")
        print(f"    分类：{metadata.get('category', '')}")
        print(f"    法律：{metadata.get('legal_name', '')}")
        print(f"    条号：{metadata.get('article_no', '')}")
        print(f"    文档：{shorten(document, 180)}")


def validate_collection(
    store: LegalChromaStore,
    embedding_model: str,
    schema_version: str,
    source_type: str,
) -> None:
    """
    执行正式的 collection 完整性校验。

    Args:
        store: Chroma 存储封装。
        embedding_model: 当前查询配置中的 embedding 模型名。
        schema_version: 当前代码期望的 schema 版本。
        source_type: 当前 collection 应包含的内容类型。
    """

    print("\n===== 完整性校验 =====")
    store.validate_collection(
        embedding_model=embedding_model,
        schema_version=schema_version,
        source_type=source_type,
    )
    print("校验通过：embedding_model、schema_version、build_complete、source_file_hash 和文档数量均正常。")


def build_query_cases(args: argparse.Namespace) -> list[QueryCase]:
    """
    根据命令行参数构造检索测试用例。

    Args:
        args: 命令行参数。

    Returns:
        list[QueryCase]: 待执行的检索测试用例。
    """

    if args.query.strip():
        return [
            QueryCase(
                query=args.query.strip(),
                legal_name=args.legal_name.strip(),
                category=args.category.strip(),
                article_no=args.article_no.strip(),
                top_k=args.top_k,
                include_neighbors=args.include_neighbors,
            )
        ]

    # 默认用例覆盖劳动、刑事、民事三类常见问题。
    # 这样做的原因是单个问题偶然命中不能充分说明库可用；多领域 smoke test 更容易发现过滤或检索异常。
    return [
        QueryCase(
            query="公司不签劳动合同怎么办",
            legal_name="劳动合同法",
            top_k=args.top_k,
            include_neighbors=args.include_neighbors,
        ),
        QueryCase(
            query="故意伤害他人造成轻伤会怎么处罚",
            legal_name="刑法",
            top_k=args.top_k,
            include_neighbors=args.include_neighbors,
        ),
        QueryCase(
            query="离婚时夫妻共同财产如何分割",
            legal_name="民法典",
            top_k=args.top_k,
            include_neighbors=args.include_neighbors,
        ),
    ]


def run_query_cases(cases: list[QueryCase]) -> None:
    """
    执行语义检索测试并打印结果。

    Args:
        cases: 待执行的检索测试用例。
    """

    print("\n===== 语义检索测试 =====")
    print("正在加载 BGE-M3 并生成查询向量，首次运行可能需要等待模型加载。")

    retriever = LegalArticleRetriever()
    for index, case in enumerate(cases, start=1):
        print(f"\n--- 查询 {index}: {case.query} ---")
        if case.legal_name:
            print(f"法律名称过滤：{case.legal_name}")
        if case.category:
            print(f"分类过滤：{case.category}")
        if case.article_no:
            print(f"条号过滤：{case.article_no}")

        result = retriever.search_legal_articles(
            query=case.query,
            top_k=case.top_k,
            legal_name=case.legal_name,
            category=case.category,
            article_no=case.article_no,
            include_neighbors=case.include_neighbors,
        )
        print_query_result(result)


def print_query_result(result: dict[str, Any]) -> None:
    """
    打印单次检索结果。

    Args:
        result: LegalArticleRetriever 返回的结构化检索结果。
    """

    if not result.get("ok"):
        print(f"检索失败：{result.get('error', '未知错误')}")
        return

    rows = result.get("results") or []
    print(f"返回数量：{len(rows)}")
    print(f"where：{result.get('where')}")

    if not rows:
        print("没有返回结果。可以尝试去掉 legal_name/category/article_no 过滤，或换一个更接近法条表达的问题。")
        return

    for row in rows:
        print(f"\n#{row.get('rank')} score={row.get('score')} distance={row.get('distance')}")
        print(f"引用：{row.get('citation')}")
        print(f"分类：{row.get('category')} | 法律：{row.get('legal_name')} | 条号：{row.get('article_no')}")
        print(f"原文：{shorten(str(row.get('text', '')), 360)}")

        neighbors = row.get("neighbors") or []
        for neighbor in neighbors:
            print(
                f"  相邻[{neighbor.get('direction')}]："
                f"{neighbor.get('citation')} {shorten(str(neighbor.get('text', '')), 160)}"
            )


def shorten(text: str, limit: int) -> str:
    """
    截断长文本，避免测试输出刷屏。

    Args:
        text: 原始文本。
        limit: 最大展示字符数。

    Returns:
        str: 适合命令行展示的短文本。
    """

    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


if __name__ == "__main__":
    main()
