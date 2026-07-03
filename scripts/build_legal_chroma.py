"""
构建本地 Chroma 法条向量库。

运行方式：
    python scripts/build_legal_chroma.py

脚本会读取 `data/最核心法条_9k.json`，按“单条正式法条 = 一个向量文档”的粒度生成 BGE-M3 embedding，
并持久化写入 `data/chroma/` 下的 `legal_articles_v1` collection。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    # 允许用户直接通过 `python scripts/build_legal_chroma.py` 运行脚本。
    # 原因是脚本目录不在包根目录下，显式补充项目根路径可以避免导入 agent_system 失败。
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_system.config import (  # noqa: E402
    load_chroma_config,
    load_embedding_config,
    load_retrieval_config,
)
from agent_system.retrieval.chroma_store import LegalChromaStore  # noqa: E402
from agent_system.retrieval.embedding import LocalBGEEmbeddingModel  # noqa: E402
from agent_system.retrieval.legal_preprocess import (  # noqa: E402
    LegalArticleDocument,
    build_legal_documents,
    calculate_file_sha256,
)


def main() -> None:
    """
    执行法条向量库构建流程。
    """

    embedding_config = load_embedding_config()
    chroma_config = load_chroma_config()
    retrieval_config = load_retrieval_config()

    print("===== 构建本地 Chroma 法条向量库 =====")
    print(f"原始数据：{chroma_config.data_path}")
    print(f"持久化目录：{chroma_config.persist_directory}")
    print(f"Collection：{chroma_config.collection_name}")
    print(f"Embedding 模型：{embedding_config.model_name}")
    print(f"Embedding 设备：{embedding_config.device or 'auto'}")
    print(f"Embedding batch size：{embedding_config.batch_size}")

    documents = build_legal_documents(
        chroma_config.data_path,
        include_source_types=(retrieval_config.default_source_type,),
    )
    if not documents:
        raise RuntimeError("没有可入库的正式法条文档，请检查原始数据和 source_type 规则。")

    print(f"待入库正式法条数：{len(documents)}")
    print_build_statistics(documents)

    source_file_hash = calculate_file_sha256(chroma_config.data_path)
    collection_metadata = {
        "embedding_provider": embedding_config.provider,
        "embedding_model": embedding_config.model_name,
        "schema_version": chroma_config.schema_version,
        "data_source": chroma_config.data_path,
        "source_file_hash": source_file_hash,
        "document_count": len(documents),
        "source_type": retrieval_config.default_source_type,
        "build_complete": False,
        "build_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        # 建库时使用 cosine 距离；BGE 向量同时做归一化，检索分数更稳定。
        "hnsw:space": "cosine",
    }

    store = LegalChromaStore(chroma_config)
    store.reset_collection(collection_metadata)

    embedder = LocalBGEEmbeddingModel(embedding_config)
    for batch_index, batch in enumerate(iter_batches(documents, embedding_config.batch_size), start=1):
        texts = [document.document for document in batch]
        embeddings = embedder.embed_documents(texts, show_progress_bar=False)
        store.upsert_documents(batch, embeddings)
        print(f"已写入批次 {batch_index}，累计 {min(batch_index * embedding_config.batch_size, len(documents))}/{len(documents)}")

    collection = store.get_collection()
    collection_count = collection.count()
    if collection_count != len(documents):
        raise RuntimeError(
            f"Chroma 写入数量不完整：collection={collection_count}, expected={len(documents)}。"
        )

    # 只有在数量校验通过后才把 build_complete 标为 true。
    # 原因是建库过程中断时，检索端必须拒绝使用半成品 collection，避免遗漏大量法条而不自知。
    complete_metadata = dict(collection_metadata)
    complete_metadata["build_complete"] = True

    # Chroma 把 hnsw:space 视为 collection 创建阶段的索引参数，创建后不允许通过 modify 再提交。
    # 即使值仍然是 cosine，部分版本也会把它判定为“修改距离函数”并抛错。
    # 因此完成标记阶段只更新普通业务 metadata，距离函数保留创建 collection 时已经写入的配置。
    complete_metadata.pop("hnsw:space", None)
    collection.modify(metadata=complete_metadata)

    print("===== 构建完成 =====")
    print(f"Collection 文档数：{collection_count}")
    print("后续可运行：python legal_agent_demo.py")


def iter_batches(
    documents: list[LegalArticleDocument],
    batch_size: int,
) -> Iterator[list[LegalArticleDocument]]:
    """
    按固定大小切分文档列表。

    Args:
        documents: 待切分文档列表。
        batch_size: 每批数量。

    Yields:
        list[LegalArticleDocument]: 当前批次文档。
    """

    for start in range(0, len(documents), batch_size):
        yield documents[start : start + batch_size]


def print_build_statistics(documents: list[LegalArticleDocument]) -> None:
    """
    打印建库前统计信息。

    Args:
        documents: 已解析的正式法条文档列表。
    """

    category_counter = Counter(str(document.metadata.get("category", "")) for document in documents)
    law_counter = Counter(str(document.metadata.get("legal_name", "")) for document in documents)

    print("分类分布 Top 5：")
    for category, count in category_counter.most_common(5):
        print(f"  - {category}: {count}")

    print("法律名称分布 Top 5：")
    for law_name, count in law_counter.most_common(5):
        print(f"  - {law_name}: {count}")


if __name__ == "__main__":
    main()
