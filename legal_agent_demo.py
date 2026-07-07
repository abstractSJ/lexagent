"""
法律咨询 Agent Demo。

运行前先构建本地向量库：
    python scripts/build_legal_chroma.py

然后运行本文件：
    python legal_agent_demo.py

该 Demo 使用现有 OpenAI-compatible LLM 负责理解问题和组织回答，使用本地 BGE-M3 + Chroma
检索正式法条。回答中会引用检索工具返回的 citation，避免模型凭空编造法条。
"""

from __future__ import annotations

import json
import os

from agent_system.agent import AgentEvent, create_agent_session
from agent_system.config import load_embedding_config
from agent_system.legal_consultation import create_legal_consultation_session


SYSTEM_PROMPT = """
你是一个中文法律咨询 Agent，负责根据本地法条库为用户提供信息参考。

规则：
1. 用户询问法律依据、赔偿责任、权利义务、诉讼/仲裁程序、行政/刑事责任或具体条文时，必须先调用 search_legal_articles 工具。
2. 调用工具时不要只取少量法条；常规问题 top_k 传 15，复杂行为、多法律关系或可能涉及多部法律的问题 top_k 传 20-30。
3. 如果用户明确提到法律名称或条号，应把 legal_name、article_no 参数传给工具；没有明确限制时传空字符串。
4. 工具返回的多条法条需要由你自行判断主次：优先引用直接依据，也可以说明间接相关或不适用的条文为什么不作为核心依据。
5. 不得编造法条名称、条号或条文内容；只能引用工具返回的 citation 和 text。
6. 如果工具没有检索到直接依据，应明确说明“当前法条库未检索到直接依据”，再给出谨慎的一般性分析。
7. 回答结构建议包含：结论、相关法条、适用说明、注意事项。
8. 结尾必须提示：以下内容仅作一般信息参考，不构成正式法律意见。
""".strip()


def print_event(event: AgentEvent) -> None:
    """
    打印法律 Agent 的执行过程。

    Args:
        event: 法律咨询会话返回的过程事件。
    """

    if event.type == "legal_step":
        status = "开始" if event.data.get("status") == "start" else str(event.data.get("status"))
        print(f"\n[步骤] {status}：{event.data.get('name')}", flush=True)
        return

    if event.type == "legal_rag_query_started":
        print("\n[检索中]")
        print(f"类型：{event.data.get('retrieval_type')}")
        print(f"事项：{event.data.get('issue')}")
        print(f"query：{event.data.get('query')}", flush=True)
        return

    if event.type == "error":
        print("\n[错误]")
        print(event.data.get("error"), flush=True)
        return

    if event.type == "case_state_updated":
        print("\n[案件状态]")
        print(f"版本：{event.data.get('version')}")
        print(f"摘要：{event.data.get('summary')}")
        if event.data.get("changed_facts"):
            print("修正事实：")
            print(json.dumps(event.data.get("changed_facts"), ensure_ascii=False, indent=2, default=str))
        return

    if event.type == "legal_missing_details_suggested":
        print("\n[可先补充的关键信息]")
        if event.data.get("questions"):
            print("追问：")
            print(json.dumps(event.data.get("questions"), ensure_ascii=False, indent=2, default=str))
        if event.data.get("evidence_gaps"):
            print("证据缺口：")
            print(json.dumps(event.data.get("evidence_gaps"), ensure_ascii=False, indent=2, default=str))
        return

    if event.type == "legal_supplement_required":
        print("\n[等待补充]")
        print(event.data.get("message") or "请先补充关键信息。", flush=True)
        return

    if event.type == "legal_case_rag_done":
        print("\n[案情拆解 + 多 query RAG]")
        print(f"事项数：{event.data.get('issue_count')}，证据数：{event.data.get('evidence_count')}")
        print(json.dumps(event.data.get("issues", []), ensure_ascii=False, indent=2, default=str))
        return

    if event.type == "legal_risk_analyzed":
        print("\n[风险识别]")
        print(f"风险数：{event.data.get('risk_count')}")
        return

    if event.type == "legal_analysis_catalog_built":
        print("\n[追问目录]")
        print(json.dumps(event.data.get("follow_up_questions", []), ensure_ascii=False, indent=2, default=str))
        return

    if event.type == "legal_next_action_decided":
        print("\n[下一步动作]")
        print(json.dumps(event.data, ensure_ascii=False, indent=2, default=str))
        return

    if event.type == "tool_call":
        print(f"\n[工具调用] {event.data.get('name')}")
        print(json.dumps(event.data.get("arguments"), ensure_ascii=False, indent=2, default=str))
        return

    if event.type == "tool_result":
        result = event.data.get("result")
        print(f"\n[工具结果] {event.data.get('name')}")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    """
    启动法律咨询 Agent 命令行 Demo。
    """

    use_legacy_agent = os.getenv("LEGAL_AGENT_LEGACY", "").strip() in {"1", "true", "yes", "是"}
    if use_legacy_agent:
        session = create_agent_session(system_prompt=SYSTEM_PROMPT)
        print("法律咨询 Agent 已启动（legacy 工具循环模式）。输入 exit 或 quit 退出。")
    else:
        session = create_legal_consultation_session()
        embedding_device = load_embedding_config().device or "auto"
        print("法律咨询 Agent 已启动（多轮案件调研链路）。输入 exit 或 quit 退出。")
        # 启动时直接展示 embedding 设备。原因是 BGE-M3 首次加载可能较慢，用户需要明确知道
        # 当前慢启动来自 CPU 稳定模式还是显式开启的 CUDA 模式。
        print(f"Embedding 设备：{embedding_device}", flush=True)
        if os.getenv("LEGAL_RAG_PRELOAD", "1").strip() not in {"0", "false", "no", "否"}:
            print(
                f"正在预热本地法条 RAG（Embedding 设备：{embedding_device}；BGE-M3、Chroma、关键词索引）...",
                flush=True,
            )
            session.preload_resources()
            print("本地法条 RAG 预热完成。", flush=True)
    if use_legacy_agent:
        print("首次检索会加载本地 BGE-M3 模型，可能需要等待一段时间。")

    while True:
        user_input = input("\n用户：").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("程序已退出。")
            break

        if not user_input:
            print("请输入非空内容。")
            continue

        try:
            print("\n[开始处理] 已收到输入，开始执行法律咨询链路。", flush=True)
            if use_legacy_agent:
                answer, events = session.ask_with_events(user_input)
                for event in events:
                    print_event(event)
            else:
                answer, _ = session.ask_with_events(user_input, on_event=print_event)
        except Exception as error:
            print(f"\n调用失败：{error}", flush=True)
            continue

        print(f"\n助手：{answer}")


if __name__ == "__main__":
    main()
