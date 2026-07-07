"""
项目配置模块。

当前项目处于学习阶段，所以这里直接使用 Python 配置文件保存主要参数，
而不是引入额外的 YAML 配置系统。这样做的好处是代码更少、跳转更直接、类型更清晰。

注意：本文件会进 git，不要在这里写真实 API Key。真实 Key 放环境变量，
或放 agent_system/config_local.py（已被 .gitignore 排除，模板见 config_local.example.py）。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_float_env(name: str, default: float) -> float:
    """
    读取浮点型环境变量。

    Args:
        name: 环境变量名称。
        default: 未设置时使用的默认值。

    Returns:
        float: 解析后的浮点数。

    Raises:
        RuntimeError: 环境变量存在但不是数字时抛出。
    """

    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return float(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} 必须是数字。") from error


def read_int_env(name: str, default: int) -> int:
    """
    读取整数型环境变量。

    Args:
        name: 环境变量名称。
        default: 未设置时使用的默认值。

    Returns:
        int: 解析后的整数。

    Raises:
        RuntimeError: 环境变量存在但不是整数时抛出。
    """

    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} 必须是整数。") from error


def read_bool_env(name: str, default: bool) -> bool:
    """
    读取布尔型环境变量。

    Args:
        name: 环境变量名称。
        default: 未设置时使用的默认值。

    Returns:
        bool: 解析后的布尔值。

    Raises:
        RuntimeError: 环境变量存在但不是可识别布尔值时抛出。
    """

    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "是", "开启"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "否", "关闭"}:
        return False
    raise RuntimeError(f"{name} 必须是布尔值。")


@dataclass(frozen=True)
class LLMConfig:
    """
    LLM 调用配置。

    Attributes:
        api_key: OpenAI 或 OpenAI-compatible 服务的 API Key。
        model: 默认使用的模型名称。
        base_url: 可选的接口地址。官方 OpenAI 可以不填；兼容服务通常需要填写 /v1 地址。
        temperature: 默认采样温度。值越低越稳定，值越高越发散。
        reasoning_effort: Responses API 的推理强度。常见值为 low、medium、high；为 None 时不传该参数。
        timeout: 请求超时时间，单位为秒。
        max_tokens: 可选的最大生成 token 数。为 None 时不主动传递该参数。
    """

    api_key: str
    model: str
    base_url: Optional[str]
    temperature: float
    reasoning_effort: Optional[str]
    timeout: float
    max_tokens: Optional[int] = None


@dataclass(frozen=True)
class LLMCallOptions:
    """
    单次 LLM 调用覆盖参数。

    Attributes:
        model: 可选模型名。为 None 时沿用全局 LLMConfig.model。
        temperature: 可选采样温度。为 None 时沿用全局 LLMConfig.temperature。
        reasoning_effort: 可选推理强度。为 None 时沿用全局 LLMConfig.reasoning_effort。
        max_tokens: 可选最大输出 token 数。为 None 时沿用全局 LLMConfig.max_tokens。
        disable_reasoning: 是否显式禁用 reasoning 参数。
            之所以需要这个开关，是因为 None 在这里表示“继承默认配置”，
            不能同时再表示“本次请求不要传 reasoning”。
        disable_max_tokens: 是否显式禁用 max_output_tokens 参数。
            这样上层既可以保留项目默认上限，也可以在单次调用里主动取消上限。
    """

    model: Optional[str] = None
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    max_tokens: Optional[int] = None
    disable_reasoning: bool = False
    disable_max_tokens: bool = False


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    本地 embedding 模型配置。

    Attributes:
        provider: embedding 提供方。第一版固定为 local，表示使用本地 sentence-transformers 模型。
        model_name: 本地 embedding 模型名称或本地模型目录。
        device: 可选运行设备，例如 cpu、cuda。默认使用 cpu，优先稳定，避免本地 GPU 驱动异常导致黑屏。
        batch_size: 批量向量化大小。CPU 环境下不宜过大，避免内存占用过高。
        torch_num_threads: CPU 推理线程数。限制线程数可以降低满载、发热和系统卡死风险。
        normalize_embeddings: 是否对向量做 L2 归一化。配合 Chroma cosine 距离时建议开启。
    """

    provider: str
    model_name: str
    device: Optional[str]
    batch_size: int
    torch_num_threads: Optional[int]
    normalize_embeddings: bool


@dataclass(frozen=True)
class ChromaConfig:
    """
    Chroma 法条向量库配置。

    Attributes:
        data_path: 原始法条 JSON 文件路径。
        persist_directory: Chroma 持久化目录。
        collection_name: 主法条 collection 名称。
        schema_version: 当前入库 schema 版本。schema 变化时应重建 collection。
    """

    data_path: str
    persist_directory: str
    collection_name: str
    schema_version: str


@dataclass(frozen=True)
class RetrievalConfig:
    """
    法条检索默认参数。

    Attributes:
        top_k: 工具默认返回条数。法律问题常常涉及多个相关条文，因此默认给 LLM 较充分的候选依据。
        candidate_k: Chroma 初始召回条数。先多召回再裁剪，可以减少相关法条被过早漏掉的概率。
        max_top_k: 工具允许的最大返回条数，在尽量提供充分依据的同时避免一次输出过长。
        default_source_type: 默认检索内容类型。第一版只检索正式法条。
        include_neighbors_default: 是否默认返回相邻条文。
    """

    top_k: int
    candidate_k: int
    max_top_k: int
    default_source_type: str
    include_neighbors_default: bool


@dataclass(frozen=True)
class WebSearchConfig:
    """
    博查 Web Search 工具配置。

    Attributes:
        api_key: 博查 Web Search API Key。
        endpoint: 博查 Web Search 接口地址。
        timeout: 请求超时时间，单位为秒。
        default_count: 工具默认返回搜索结果数量。
        max_count: 工具允许的最大返回结果数量，用于防止一次工具结果过长。
        default_summary: 是否默认请求搜索结果摘要。
        default_freshness: 默认时间过滤范围，例如 noLimit、oneDay、oneWeek、oneMonth、oneYear。
    """

    api_key: str
    endpoint: str
    timeout: float
    default_count: int
    max_count: int
    default_summary: bool
    default_freshness: str


# 本地私密默认值：真实 API Key、内网地址等不进 git 的配置放在 config_local.py（已被
# .gitignore 排除），模板见 config_local.example.py。环境变量仍然最优先；两者都没有时
# 回退到 config.py 的内置默认值，由各 load_*_config() 在启动时校验并给出明确报错。
try:
    from agent_system.config_local import LOCAL_ENV_DEFAULTS
except ImportError:
    LOCAL_ENV_DEFAULTS: dict[str, str] = {}


def read_env_with_local_default(name: str, fallback: str = "") -> str:
    """
    读取字符串配置，缺省时依次回退 config_local.py 和内置默认值。

    Args:
        name: 环境变量名称，同时也是 config_local.py 中 LOCAL_ENV_DEFAULTS 的键名。
        fallback: 环境变量和本地私密配置都未提供时的默认值。

    Returns:
        str: 解析后的配置值。
    """

    raw_value = os.getenv(name)
    if raw_value is not None and raw_value.strip():
        return raw_value
    return str(LOCAL_ENV_DEFAULTS.get(name, fallback))


# 当前项目的直接配置区。
# LLM 配置保留原有 OpenAI-compatible 服务；如果设置了环境变量，则优先使用环境变量覆盖。
LLM_CONFIG = LLMConfig(
    api_key=read_env_with_local_default("AGENT_LLM_API_KEY"),
    base_url=read_env_with_local_default("AGENT_LLM_BASE_URL"),
    model=read_env_with_local_default("AGENT_LLM_MODEL", "gpt-5.5"),
    temperature=0.7,
    reasoning_effort="medium",
    timeout=read_float_env("AGENT_LLM_TIMEOUT", 180.0),
    max_tokens=None,
)


# 本地 BGE-M3 + Chroma 是法条库第一版默认方案。
# 建库和查询必须使用同一份 EmbeddingConfig；如果更换 model_name，需要重新执行建库脚本。
EMBEDDING_CONFIG = EmbeddingConfig(
    provider="local",
    model_name=os.getenv("LEGAL_EMBEDDING_MODEL", "BAAI/bge-m3"),
    # 默认使用 CPU。原因是本项目常在 Windows 本机学习环境运行，CUDA 驱动、显存和 PyTorch
    # 版本不匹配时容易导致启动卡顿、失败甚至系统不稳定；确认环境稳定后可显式设置
    # LEGAL_EMBEDDING_DEVICE=cuda 启用 GPU。
    device=(os.getenv("LEGAL_EMBEDDING_DEVICE") or "cpu").strip() or "cpu",
    batch_size=8,
    torch_num_threads=4,
    normalize_embeddings=True,
)


CHROMA_CONFIG = ChromaConfig(
    data_path=str(PROJECT_ROOT / "data" / "最核心法条_9k.json"),
    persist_directory=str(PROJECT_ROOT / "data" / "chroma"),
    collection_name="legal_articles_v1",
    schema_version="legal_articles_v1",
)


RETRIEVAL_CONFIG = RetrievalConfig(
    # 默认返回 15 条而不是少量条文。
    # 原因是一个法律行为可能同时涉及定义、构成要件、责任承担、程序、例外和处罚幅度；
    # 检索阶段宁可多给候选依据，再交给 LLM 在回答阶段筛选和组织。
    top_k=15,
    # 初始召回数量高于最终返回数量，给后续重排或规则过滤留出缓冲空间，降低漏召回风险。
    candidate_k=60,
    # 允许复杂问题最多取 30 条，兼顾“信息充分”和“上下文不过度膨胀”。
    max_top_k=30,
    default_source_type="law_article",
    include_neighbors_default=False,
)


WEB_SEARCH_CONFIG = WebSearchConfig(
    api_key=read_env_with_local_default("BOCHA_WEB_SEARCH_API_KEY"),
    endpoint=os.getenv("BOCHA_WEB_SEARCH_URL", "https://api.bocha.cn/v1/web-search"),
    timeout=read_float_env("BOCHA_WEB_SEARCH_TIMEOUT", 30.0),
    # 默认取 10 条结果，原因是 Web 搜索结果摘要可能较长；先保证上下文可控，再由模型按需二次搜索。
    default_count=read_int_env("BOCHA_WEB_SEARCH_COUNT", 10),
    # 博查文档常见上限为 50。这里也在本地限制一次工具输出规模，避免外部网页摘要挤占法律证据上下文。
    max_count=read_int_env("BOCHA_WEB_SEARCH_MAX_COUNT", 50),
    default_summary=read_bool_env("BOCHA_WEB_SEARCH_SUMMARY", True),
    default_freshness=os.getenv("BOCHA_WEB_SEARCH_FRESHNESS", "noLimit"),
)


def load_llm_config() -> LLMConfig:
    """
    加载 LLM 配置。

    Returns:
        LLMConfig: 当前项目使用的 LLM 配置对象。

    Raises:
        RuntimeError: 当 api_key 仍是占位符时抛出。
    """

    # 这里做一次启动前校验。
    # 原因是 API Key 如果没填，真正调用模型时才失败会比较绕；在加载配置时失败更容易定位问题。
    if not LLM_CONFIG.api_key.strip() or LLM_CONFIG.api_key.startswith(("请在这里填入", "请填入")):
        raise RuntimeError(
            "缺少 LLM API Key：请设置环境变量 AGENT_LLM_API_KEY，"
            "或参照 agent_system/config_local.example.py 创建 agent_system/config_local.py 并填入真实 Key。"
        )

    return LLM_CONFIG


def load_embedding_config() -> EmbeddingConfig:
    """
    加载本地 embedding 配置。

    Returns:
        EmbeddingConfig: 当前法条库使用的 embedding 配置。

    Raises:
        RuntimeError: 配置不合法时抛出。
    """

    if EMBEDDING_CONFIG.provider != "local":
        raise RuntimeError("当前实现只支持本地 embedding provider：local。")
    if not EMBEDDING_CONFIG.model_name.strip():
        raise RuntimeError("LEGAL_EMBEDDING_MODEL 不能为空。")

    raw_batch_size = os.getenv("LEGAL_EMBEDDING_BATCH_SIZE")
    batch_size = EMBEDDING_CONFIG.batch_size
    if raw_batch_size is not None:
        try:
            batch_size = int(raw_batch_size)
        except ValueError as error:
            raise RuntimeError("LEGAL_EMBEDDING_BATCH_SIZE 必须是正整数。") from error

    if batch_size < 1:
        raise RuntimeError("LEGAL_EMBEDDING_BATCH_SIZE 必须大于等于 1。")

    raw_torch_num_threads = os.getenv("LEGAL_TORCH_NUM_THREADS")
    torch_num_threads = EMBEDDING_CONFIG.torch_num_threads
    if raw_torch_num_threads is not None:
        try:
            torch_num_threads = int(raw_torch_num_threads)
        except ValueError as error:
            raise RuntimeError("LEGAL_TORCH_NUM_THREADS 必须是正整数。") from error

    if torch_num_threads is not None and torch_num_threads < 1:
        raise RuntimeError("LEGAL_TORCH_NUM_THREADS 必须大于等于 1。")

    return EmbeddingConfig(
        provider=EMBEDDING_CONFIG.provider,
        model_name=EMBEDDING_CONFIG.model_name,
        device=EMBEDDING_CONFIG.device,
        batch_size=batch_size,
        torch_num_threads=torch_num_threads,
        normalize_embeddings=EMBEDDING_CONFIG.normalize_embeddings,
    )


def load_chroma_config() -> ChromaConfig:
    """
    加载 Chroma 法条库配置。

    Returns:
        ChromaConfig: 当前 Chroma 配置。

    Raises:
        RuntimeError: 原始数据文件不存在或 collection 名称为空时抛出。
    """

    if not Path(CHROMA_CONFIG.data_path).exists():
        raise RuntimeError(f"法条数据文件不存在：{CHROMA_CONFIG.data_path}")
    if not CHROMA_CONFIG.collection_name.strip():
        raise RuntimeError("Chroma collection_name 不能为空。")
    if not CHROMA_CONFIG.schema_version.strip():
        raise RuntimeError("Chroma schema_version 不能为空。")
    return CHROMA_CONFIG


def load_retrieval_config() -> RetrievalConfig:
    """
    加载法条检索配置。

    Returns:
        RetrievalConfig: 当前检索默认参数。

    Raises:
        RuntimeError: 检索参数不合法时抛出。
    """

    if RETRIEVAL_CONFIG.top_k < 1:
        raise RuntimeError("RETRIEVAL_CONFIG.top_k 必须大于等于 1。")
    if RETRIEVAL_CONFIG.candidate_k < RETRIEVAL_CONFIG.top_k:
        raise RuntimeError("RETRIEVAL_CONFIG.candidate_k 必须大于等于 top_k。")
    if RETRIEVAL_CONFIG.max_top_k < RETRIEVAL_CONFIG.top_k:
        raise RuntimeError("RETRIEVAL_CONFIG.max_top_k 必须大于等于 top_k。")
    return RETRIEVAL_CONFIG


def load_web_search_config() -> WebSearchConfig:
    """
    加载博查 Web Search 配置。

    Returns:
        WebSearchConfig: 当前 Web Search 工具配置。

    Raises:
        RuntimeError: 配置不合法时抛出。
    """

    if not WEB_SEARCH_CONFIG.api_key.strip():
        raise RuntimeError("BOCHA_WEB_SEARCH_API_KEY 不能为空。")
    if not WEB_SEARCH_CONFIG.endpoint.strip():
        raise RuntimeError("BOCHA_WEB_SEARCH_URL 不能为空。")
    if WEB_SEARCH_CONFIG.timeout <= 0:
        raise RuntimeError("BOCHA_WEB_SEARCH_TIMEOUT 必须大于 0。")
    if WEB_SEARCH_CONFIG.default_count < 1:
        raise RuntimeError("BOCHA_WEB_SEARCH_COUNT 必须大于等于 1。")
    if WEB_SEARCH_CONFIG.max_count < WEB_SEARCH_CONFIG.default_count:
        raise RuntimeError("BOCHA_WEB_SEARCH_MAX_COUNT 必须大于等于 BOCHA_WEB_SEARCH_COUNT。")

    return WEB_SEARCH_CONFIG
