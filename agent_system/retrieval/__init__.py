"""
法条 RAG 检索层对外入口。

该包提供三类能力：
1. 预处理原始法条 JSON；
2. 使用本地 BGE-M3 生成 embedding 并写入 Chroma；
3. 在 Agent 工具中检索和格式化相关法条。
"""

from agent_system.retrieval.legal_retriever import LegalArticleRetriever, build_legal_retriever

__all__ = [
    "LegalArticleRetriever",
    "build_legal_retriever",
]
