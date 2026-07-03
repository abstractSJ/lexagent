"""
博查 Web Search Agent 工具。

本模块把博查 Web Search HTTP API 包装成项目现有的 LocalTool。工具只返回整理后的搜索结果，
避免把外部接口的完整原始响应直接塞进模型上下文，保持最终回答阶段可控。
"""

from __future__ import annotations

import json
from typing import Any
import urllib.error
import urllib.request

from agent_system.agent.tools import LocalTool
from agent_system.config import WebSearchConfig, load_web_search_config


MAX_TITLE_CHARS = 200
MAX_SNIPPET_CHARS = 600
MAX_SUMMARY_CHARS = 1200
MAX_IMAGE_ITEMS = 5


def build_web_search_tools(config: WebSearchConfig | None = None) -> list[LocalTool]:
    """
    创建 Web Search 工具列表。

    Args:
        config: 可选博查 Web Search 配置。测试可以传入自定义配置；为空时首次调用工具再加载全局配置。

    Returns:
        list[LocalTool]: 可注册到 ToolRegistry 的工具列表。
    """

    cached_config = config

    def get_config() -> WebSearchConfig:
        """
        获取懒加载的 Web Search 配置。

        Returns:
            WebSearchConfig: 博查 Web Search 配置。
        """

        nonlocal cached_config
        if cached_config is None:
            cached_config = load_web_search_config()
        return cached_config

    def web_search(arguments: dict[str, Any]) -> dict[str, Any]:
        """
        执行博查 Web Search 查询。

        Args:
            arguments: 模型传入的工具参数。

        Returns:
            dict[str, Any]: 结构化搜索结果或错误信息。
        """

        query = str(arguments.get("query", "")).strip()
        if not query:
            return {"ok": False, "error": "query 不能为空。", "results": []}

        web_config = get_config()
        count = clamp_count(arguments.get("count"), web_config)
        freshness = str(arguments.get("freshness", "")).strip() or web_config.default_freshness
        summary = parse_bool_with_default(arguments.get("summary"), web_config.default_summary)
        include = str(arguments.get("include", "")).strip()
        exclude = str(arguments.get("exclude", "")).strip()

        payload: dict[str, Any] = {
            "query": query,
            "count": count,
            "summary": summary,
        }
        if freshness:
            payload["freshness"] = freshness
        # include/exclude 是博查官方的站点过滤参数（多域名用 | 或 , 分隔，上限 100 个）。
        # 服务端过滤比本地丢弃结果更划算：请求的 count 候选位不会被低质站点占用。
        if include:
            payload["include"] = include
        if exclude:
            payload["exclude"] = exclude

        try:
            response = post_web_search_request(config=web_config, payload=payload)
        except urllib.error.HTTPError as error:
            return build_http_error_result(error)
        except urllib.error.URLError as error:
            return {"ok": False, "error": f"Web Search 网络请求失败：{error.reason}", "results": []}
        except TimeoutError:
            return {"ok": False, "error": "Web Search 请求超时。", "results": []}
        except json.JSONDecodeError as error:
            return {"ok": False, "error": f"Web Search 响应不是合法 JSON：{error}", "results": []}

        return normalize_web_search_response(response)

    return [
        LocalTool(
            name="web_search",
            description=(
                "调用博查 Web Search 搜索公网信息。法律咨询中适合用于：相似案例、最高人民法院指导性案例/"
                "公报案例/典型案例、司法解释和新近规范性文件、地方裁判口径、量刑/赔偿/补偿区间、"
                "执行程序和实务操作细节、本地法条库覆盖不到或可能已经更新的信息。"
                "建议使用包含法域、案由、法院层级和关键词的中文查询，例如："
                "最高人民法院 指导性案例 交通事故 赔偿 责任比例；"
                "盗窃罪 数额较大 量刑 标准 司法解释；"
                "劳动仲裁 未签劳动合同 二倍工资 裁判规则。"
                "需要权威来源（司法解释、裁判文书、官方文件）时可用 include 限定站点，"
                "例如 court.gov.cn|chinacourt.org|faxin.cn；需要过滤低质量咨询站时可用 exclude 排除站点。"
                "返回结果包含标题、链接、站点、摘要和抓取时间；回答时必须区分公网资料与本地法条依据，"
                "不能把搜索摘要当作正式法条原文。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题，例如：2026年上海最低工资标准。",
                    },
                    "count": {
                        "type": "integer",
                        "description": "返回结果数量。常规取 5-10；需要更充分材料时可取更大值，最大值受本地配置限制。",
                    },
                    "freshness": {
                        "type": "string",
                        "description": "时间范围过滤；常用 noLimit、oneDay、oneWeek、oneMonth、oneYear，不限制时传 noLimit。",
                    },
                    "summary": {
                        "type": "boolean",
                        "description": "是否请求搜索结果摘要。需要快速阅读网页要点时传 true；只需要标题和链接时传 false。",
                    },
                    "include": {
                        "type": "string",
                        "description": "限定检索站点域名，多个用 | 分隔，例如 court.gov.cn|chinacourt.org；不限定时传空字符串。",
                    },
                    "exclude": {
                        "type": "string",
                        "description": "排除站点域名，多个用 | 分隔，用于过滤低质量来源；不排除时传空字符串。",
                    },
                },
                "required": ["query", "count", "freshness", "summary", "include", "exclude"],
                "additionalProperties": False,
            },
            handler=web_search,
        )
    ]


def post_web_search_request(*, config: WebSearchConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """
    向博查 Web Search 接口发送 JSON 请求。

    Args:
        config: 博查 Web Search 配置。
        payload: 请求体。

    Returns:
        dict[str, Any]: 接口返回的 JSON 对象。

    Raises:
        urllib.error.HTTPError: HTTP 状态码不是 2xx 时抛出。
        urllib.error.URLError: 网络连接失败时抛出。
        TimeoutError: 请求超时时抛出。
        json.JSONDecodeError: 响应体不是合法 JSON 时抛出。
    """

    request = urllib.request.Request(
        config.endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=config.timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    return json.loads(body)


def normalize_web_search_response(response: dict[str, Any]) -> dict[str, Any]:
    """
    整理博查 Web Search 原始响应。

    Args:
        response: 博查接口原始 JSON 响应。

    Returns:
        dict[str, Any]: 适合回传给模型的精简结果。
    """

    code = response.get("code")
    if code != 200:
        return {
            "ok": False,
            "code": code,
            "log_id": response.get("log_id"),
            # 博查错误响应体的字段名是 message，部分旧示例是 msg；两者都兜住，避免拿不到错误文本。
            "error": response.get("msg") or response.get("message") or "Web Search 接口返回非 200 code。",
            "results": [],
        }

    data = response.get("data") if isinstance(response.get("data"), dict) else {}
    web_pages = data.get("webPages") if isinstance(data.get("webPages"), dict) else {}
    query_context = data.get("queryContext") if isinstance(data.get("queryContext"), dict) else {}
    raw_results = web_pages.get("value") if isinstance(web_pages.get("value"), list) else []

    results = [normalize_web_page_item(item) for item in raw_results if isinstance(item, dict)]

    return {
        "ok": True,
        "code": code,
        "log_id": response.get("log_id"),
        "query": query_context.get("originalQuery"),
        "web_search_url": web_pages.get("webSearchUrl"),
        "total_estimated_matches": web_pages.get("totalEstimatedMatches"),
        "some_results_removed": web_pages.get("someResultsRemoved"),
        "results": results,
        "images": normalize_images(data.get("images")),
    }


def normalize_web_page_item(item: dict[str, Any]) -> dict[str, Any]:
    """
    整理单条网页搜索结果。

    Args:
        item: 博查返回的单条网页结果。

    Returns:
        dict[str, Any]: 精简后的网页结果。
    """

    return {
        "title": truncate_text(item.get("name"), MAX_TITLE_CHARS),
        "url": item.get("url"),
        "display_url": item.get("displayUrl"),
        "snippet": truncate_text(item.get("snippet"), MAX_SNIPPET_CHARS),
        # 外部网页 summary 可能非常长。这里截断是为了保护 Agent 上下文窗口，避免一次搜索吞掉法律证据和用户事实。
        "summary": truncate_text(item.get("summary"), MAX_SUMMARY_CHARS),
        "site_name": item.get("siteName"),
        "site_icon": item.get("siteIcon"),
        "date_published": item.get("datePublished"),
        "date_last_crawled": item.get("dateLastCrawled"),
        "cached_page_url": item.get("cachedPageUrl"),
        "language": item.get("language"),
    }


def normalize_images(images: Any) -> list[dict[str, Any]]:
    """
    整理图片搜索结果。

    Args:
        images: 博查响应中的 images 字段。

    Returns:
        list[dict[str, Any]]: 精简图片结果。
    """

    if not isinstance(images, dict):
        return []
    raw_items = images.get("value")
    if not isinstance(raw_items, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw_items[:MAX_IMAGE_ITEMS]:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "thumbnail_url": item.get("thumbnailUrl"),
                "content_url": item.get("contentUrl"),
                "host_page_url": item.get("hostPageUrl"),
                "host_page_display_url": item.get("hostPageDisplayUrl"),
            }
        )
    return normalized


def build_http_error_result(error: urllib.error.HTTPError) -> dict[str, Any]:
    """
    构造 HTTP 错误结果。

    Args:
        error: urllib 抛出的 HTTPError。

    Returns:
        dict[str, Any]: 可回传给模型的错误信息。
    """

    body = error.read().decode("utf-8", errors="replace")
    try:
        parsed_body: Any = json.loads(body)
    except json.JSONDecodeError:
        parsed_body = body[:1000]

    return {
        "ok": False,
        "status": error.code,
        "error": "Web Search HTTP 请求失败。",
        "body": parsed_body,
        "results": [],
    }


def clamp_count(value: Any, config: WebSearchConfig) -> int:
    """
    解析并限制搜索结果数量。

    Args:
        value: 模型传入的 count。
        config: Web Search 配置。

    Returns:
        int: 限制后的结果数量。
    """

    try:
        count = int(value)
    except (TypeError, ValueError):
        count = config.default_count
    if count < 1:
        return config.default_count
    return min(count, config.max_count)


def parse_bool_with_default(value: Any, default: bool) -> bool:
    """
    安全解析布尔值，并在空值时使用默认值。

    Args:
        value: 待解析值。
        default: 空值时使用的默认值。

    Returns:
        bool: 解析后的布尔值。
    """

    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on", "是", "需要"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "否", "不需要"}:
            return False
    return bool(value)


def truncate_text(value: Any, max_chars: int) -> str:
    """
    截断文本字段。

    Args:
        value: 待处理值。
        max_chars: 最大字符数。

    Returns:
        str: 截断后的字符串。
    """

    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


__all__ = [
    "build_web_search_tools",
    "clamp_count",
    "normalize_web_search_response",
    "post_web_search_request",
]
