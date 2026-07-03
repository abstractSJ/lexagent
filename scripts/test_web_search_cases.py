"""
测试博查 Web Search 对司法案例类问题的检索效果。

运行方式：
    python scripts/test_web_search_cases.py
    python scripts/test_web_search_cases.py --case-text "贩卖原味内衣内裤" --count 8
    python scripts/test_web_search_cases.py --case-text "贩卖原味内衣内裤" --compact

这个脚本直接复用 Agent 系统里注册给模型使用的 web_search LocalTool。
它的目标不是生成法律结论，而是观察公网搜索是否能召回判例、裁判文书、案例分析等材料。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    # 允许用户直接通过 `python scripts/test_web_search_cases.py` 运行脚本。
    # 原因是脚本位于 scripts 子目录，直接运行时 Python 默认不会把项目根目录加入模块搜索路径。
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_system.agent.web_search_tools import build_web_search_tools  # noqa: E402


DEFAULT_CASE_TEXT = "贩卖原味内衣内裤"
DEFAULT_COUNT = 8
DEFAULT_FRESHNESS = "noLimit"

# 用多组查询词测试同一个案情。
# 原因是司法案例在公网的标题表达很不稳定：有的写“判决书”，有的写“案例”，有的只写罪名。
# 单一 query 很容易误判 web_search 没效果，所以这里主动覆盖“案例/判例/裁判文书/罪名路径”几类说法。
QUERY_TEMPLATES = [
    "{case_text} 判例",
    "{case_text} 刑事 判决",
    "{case_text} 裁判文书",
    "{case_text} 寻衅滋事",
    "{case_text} 传播淫秽物品",
    "{case_text} 非法经营",
]

# 简单的命中提示词，只用于人工快速扫结果，不作为法律判断依据。
# 原因是搜索摘要可能来自媒体、论坛或法律咨询页，不能仅凭关键词就认定它是真实判例。
CASE_HINT_KEYWORDS = [
    "判决",
    "裁判",
    "法院",
    "案号",
    "刑事",
    "寻衅滋事",
    "传播淫秽物品",
    "非法经营",
    "治安管理处罚",
]


def main() -> None:
    """
    执行多组 web_search 查询，并打印适合人工判断检索质量的摘要。
    """

    args = parse_args()
    web_search_tool = build_web_search_tools()[0]
    query_results = []

    for query in build_queries(args.case_text):
        result = web_search_tool.handler(
            {
                "query": query,
                "count": args.count,
                "freshness": args.freshness,
                "summary": not args.no_summary,
            }
        )
        query_results.append({"query": query, "result": result})

    if args.compact:
        print(json.dumps(query_results, ensure_ascii=False, indent=2))
        return

    print_report(args.case_text, query_results)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    """

    parser = argparse.ArgumentParser(description="测试 web_search 对司法案例类关键词的召回效果。")
    parser.add_argument(
        "--case-text",
        default=DEFAULT_CASE_TEXT,
        help=f"待测试的案情关键词，默认：{DEFAULT_CASE_TEXT}。",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"每个 query 返回结果数量，默认 {DEFAULT_COUNT}。",
    )
    parser.add_argument(
        "--freshness",
        default=DEFAULT_FRESHNESS,
        help=f"时间范围过滤，例如 noLimit、oneYear、oneMonth，默认 {DEFAULT_FRESHNESS}。",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="不请求搜索结果 summary，只看标题、链接和 snippet。",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="输出完整 JSON，便于保存和后续分析。",
    )
    return parser.parse_args()


def build_queries(case_text: str) -> list[str]:
    """
    根据案情关键词构造多组搜索 query。

    Args:
        case_text: 案情关键词。

    Returns:
        list[str]: 去重后的搜索 query 列表。
    """

    normalized_case_text = case_text.strip() or DEFAULT_CASE_TEXT
    queries: list[str] = []
    seen: set[str] = set()
    for template in QUERY_TEMPLATES:
        query = template.format(case_text=normalized_case_text)
        if query not in seen:
            seen.add(query)
            queries.append(query)
    return queries


def print_report(case_text: str, query_results: list[dict[str, Any]]) -> None:
    """
    打印可读性更好的搜索效果报告。

    Args:
        case_text: 原始案情关键词。
        query_results: 每组 query 的 web_search 返回结果。
    """

    print("===== Web Search 司法案例检索测试 =====")
    print(f"案情关键词：{case_text}")
    print(f"查询组数：{len(query_results)}")

    for index, item in enumerate(query_results, start=1):
        query = item["query"]
        result = item["result"]
        print(f"\n===== Query {index}: {query} =====")

        if not result.get("ok"):
            print(f"搜索失败：{result.get('error') or result}")
            continue

        results = result.get("results") or []
        print(f"返回数量：{len(results)}")
        print(f"估计总命中：{result.get('total_estimated_matches')}")

        for result_index, web_item in enumerate(results, start=1):
            print_web_item(result_index, web_item)


def print_web_item(index: int, item: dict[str, Any]) -> None:
    """
    打印单条网页搜索结果。

    Args:
        index: 当前结果序号。
        item: web_search 返回的单条结果。
    """

    title = item.get("title") or ""
    url = item.get("url") or ""
    snippet = item.get("snippet") or ""
    summary = item.get("summary") or ""
    hints = collect_case_hints("\n".join([title, snippet, summary]))

    print(f"\n[{index}] {title}")
    print(f"URL：{url}")
    if item.get("site_name"):
        print(f"站点：{item.get('site_name')}")
    if hints:
        print(f"疑似案例关键词：{', '.join(hints)}")
    if snippet:
        print(f"Snippet：{snippet}")
    if summary:
        print(f"Summary：{summary[:500]}{'...' if len(summary) > 500 else ''}")


def collect_case_hints(text: str) -> list[str]:
    """
    从标题、摘要中提取疑似案例相关提示词。

    Args:
        text: 待检查文本。

    Returns:
        list[str]: 在文本中出现过的提示词。
    """

    return [keyword for keyword in CASE_HINT_KEYWORDS if keyword in text]


if __name__ == "__main__":
    main()
