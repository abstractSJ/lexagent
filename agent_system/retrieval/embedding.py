"""
本地 embedding 模型封装。

第一版使用 sentence-transformers 加载 `BAAI/bge-m3`。封装这一层的目的，是让建库脚本和查询工具
共享同一套 embedding 参数，避免“建库模型”和“查询模型”不一致导致向量空间错位。
"""

from __future__ import annotations

from typing import Sequence
import os

from agent_system.config import EmbeddingConfig


class LocalBGEEmbeddingModel:
    """
    本地 BGE-M3 embedding 模型。

    Args:
        config: embedding 配置，包含模型名、设备、批大小和归一化开关。

    Raises:
        RuntimeError: 未安装 sentence-transformers 或模型加载失败时抛出。
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """
        初始化本地 embedding 模型。

        Args:
            config: embedding 配置对象。
        """

        self.config = config
        self._configure_torch_runtime(config)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "缺少 sentence-transformers 依赖。请先执行：pip install -r requirements.txt"
            ) from error

        model_kwargs = {}
        if config.device:
            # 默认配置使用 cpu。原因是 BGE-M3 在部分 Windows + CUDA 环境下可能触发显卡驱动重置或黑屏，
            # 第一版优先保证建库稳定；确认显存和驱动稳定后，可通过 LEGAL_EMBEDDING_DEVICE=cuda 主动启用 GPU。
            model_kwargs["device"] = config.device

        try:
            self._model = SentenceTransformer(config.model_name, **model_kwargs)
        except Exception as error:
            raise RuntimeError(
                f"加载本地 embedding 模型失败：{config.model_name}。"
                "如果是首次运行，请确认网络可以下载 HuggingFace 模型，或提前把模型缓存到本地。"
            ) from error

    def _configure_torch_runtime(self, config: EmbeddingConfig) -> None:
        """
        配置 PyTorch CPU 推理线程。

        Args:
            config: embedding 配置对象。

        说明：
            BGE-M3 建库是长时间本地推理任务。默认限制线程数可以降低 CPU 满载、温度过高、
            系统无响应和笔记本黑屏重启的概率；如果机器散热和电源充足，可通过环境变量调大。
        """

        if config.torch_num_threads is None:
            return

        # 这些环境变量需要尽量在 torch 初始化前设置。这里用 setdefault，避免覆盖用户已经显式设置的值。
        os.environ.setdefault("OMP_NUM_THREADS", str(config.torch_num_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(config.torch_num_threads))

        try:
            import torch
        except ImportError:
            return

        torch.set_num_threads(config.torch_num_threads)
        try:
            torch.set_num_interop_threads(max(1, min(2, config.torch_num_threads)))
        except RuntimeError:
            # set_num_interop_threads 在并行运行时初始化后不能重复设置。
            # 这种情况下继续使用当前 PyTorch 配置即可，不影响功能正确性。
            pass

    def embed_documents(self, texts: Sequence[str], *, show_progress_bar: bool = False) -> list[list[float]]:
        """
        批量生成文档向量。

        Args:
            texts: 待向量化的文档文本列表。
            show_progress_bar: 是否显示 sentence-transformers 自带进度条。

        Returns:
            list[list[float]]: 每个文本对应一个 dense embedding。
        """

        if not texts:
            return []

        embeddings = self._model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        return embeddings.astype("float32").tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        生成查询向量。

        Args:
            query: 用户检索问题或关键词。

        Returns:
            list[float]: 查询向量。
        """

        query = query.strip()
        if not query:
            raise ValueError("query 不能为空。")

        return self.embed_documents([query])[0]
