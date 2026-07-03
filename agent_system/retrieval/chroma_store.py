"""
Chroma 持久化存储封装。

本模块只负责和 Chroma 交互：创建 collection、写入向量、执行向量检索和校验 collection 元信息。
业务层的“法条怎么清洗、怎么引用、怎么给 Agent 返回结果”放在 legal_retriever 中处理。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from agent_system.config import ChromaConfig
from agent_system.retrieval.legal_preprocess import (
    LegalArticleDocument,
    build_legal_documents,
    calculate_file_sha256,
)


class LegalChromaStore:
    """
    法条 Chroma collection 访问器。

    Args:
        config: Chroma 持久化配置。

    Raises:
        RuntimeError: 未安装 chromadb 时抛出。
    """

    def __init__(self, config: ChromaConfig) -> None:
        """
        初始化 Chroma persistent client。

        Args:
            config: Chroma 配置对象。
        """

        self.config = config
        Path(config.persist_directory).mkdir(parents=True, exist_ok=True)

        try:
            import chromadb
        except ImportError as error:
            raise RuntimeError("缺少 chromadb 依赖。请先执行：pip install -r requirements.txt") from error

        self._client = chromadb.PersistentClient(path=config.persist_directory)

    def reset_collection(self, metadata: dict[str, str | int | float | bool]) -> Any:
        """
        删除并重建 collection。

        Args:
            metadata: collection metadata。应包含 embedding_model、schema_version 和 hnsw:space 等字段。

        Returns:
            Any: Chroma collection 对象。
        """

        try:
            self._client.delete_collection(self.config.collection_name)
        except Exception as error:
            # Chroma 不同版本对“不存在”的异常类型不完全一致，所以这里用错误文本做兼容判断。
            # 只吞掉 collection 不存在这一类可预期情况；数据库锁、权限等真实错误继续抛出，避免建库失败原因被隐藏。
            message = str(error).lower()
            if "does not exist" not in message and "not found" not in message:
                raise

        return self._client.create_collection(
            name=self.config.collection_name,
            metadata=dict(metadata),
        )

    def get_collection(self) -> Any:
        """
        获取已存在的 collection。

        Returns:
            Any: Chroma collection 对象。

        Raises:
            RuntimeError: collection 尚未建立时抛出。
        """

        try:
            return self._client.get_collection(self.config.collection_name)
        except Exception as error:
            raise RuntimeError(
                f"没有找到 Chroma collection：{self.config.collection_name}。"
                "请先执行：python scripts/build_legal_chroma.py"
            ) from error

    def upsert_documents(
        self,
        documents: Sequence[LegalArticleDocument],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """
        写入一批法条文档和对应向量。

        Args:
            documents: 法条文档列表。
            embeddings: 与 documents 一一对应的向量列表。

        Raises:
            ValueError: 文档数量和向量数量不一致时抛出。
        """

        if len(documents) != len(embeddings):
            raise ValueError("documents 和 embeddings 数量必须一致。")
        if not documents:
            return

        collection = self.get_collection()
        collection.upsert(
            ids=[document.id for document in documents],
            documents=[document.document for document in documents],
            metadatas=[document.metadata for document in documents],
            embeddings=[list(embedding) for embedding in embeddings],
        )

    def validate_collection(
        self,
        *,
        embedding_model: str,
        schema_version: str,
        source_type: str = "law_article",
    ) -> None:
        """
        校验 collection 的关键元信息和完整性。

        Args:
            embedding_model: 当前查询阶段使用的 embedding 模型名。
            schema_version: 当前代码期望的数据 schema 版本。
            source_type: 当前 collection 应包含的内容类型。

        Raises:
            RuntimeError: collection 使用的模型、schema、数据版本或文档数量与当前配置不一致时抛出。
        """

        collection = self.get_collection()
        metadata = collection.metadata or {}

        actual_embedding_model = metadata.get("embedding_model")
        if actual_embedding_model != embedding_model:
            raise RuntimeError(
                "Chroma collection 的 embedding 模型与当前配置不一致："
                f"collection={actual_embedding_model!r}, current={embedding_model!r}。"
                "请删除旧库或重新执行：python scripts/build_legal_chroma.py"
            )

        actual_schema_version = metadata.get("schema_version")
        if actual_schema_version != schema_version:
            raise RuntimeError(
                "Chroma collection 的 schema_version 与当前配置不一致："
                f"collection={actual_schema_version!r}, current={schema_version!r}。"
                "请重新执行：python scripts/build_legal_chroma.py"
            )

        if metadata.get("build_complete") is not True:
            raise RuntimeError(
                "Chroma collection 尚未完整构建完成。"
                "请重新执行：python scripts/build_legal_chroma.py"
            )

        expected_source_hash = calculate_file_sha256(self.config.data_path)
        actual_source_hash = metadata.get("source_file_hash")
        if actual_source_hash != expected_source_hash:
            raise RuntimeError(
                "Chroma collection 的原始数据 hash 与当前数据文件不一致。"
                "请重新执行：python scripts/build_legal_chroma.py"
            )

        expected_count = len(build_legal_documents(self.config.data_path, include_source_types=(source_type,)))
        metadata_count = metadata.get("document_count")
        collection_count = collection.count()
        if metadata_count != expected_count or collection_count != expected_count:
            raise RuntimeError(
                "Chroma collection 文档数量不完整或与当前数据不一致："
                f"metadata={metadata_count!r}, collection={collection_count}, expected={expected_count}。"
                "请重新执行：python scripts/build_legal_chroma.py"
            )

    def query_by_embedding(
        self,
        *,
        query_embedding: Sequence[float],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        使用已生成的查询向量执行 Chroma 检索。

        Args:
            query_embedding: 查询向量。
            n_results: 召回数量。
            where: 可选 metadata 过滤条件。

        Returns:
            dict[str, Any]: Chroma 原始查询结果。
        """

        collection = self.get_collection()
        count = collection.count()
        if count <= 0:
            raise RuntimeError("Chroma collection 为空，请重新执行：python scripts/build_legal_chroma.py")

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [list(query_embedding)],
            "n_results": max(1, min(n_results, count)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        return collection.query(**query_kwargs)
