"""
Agent 对象创建工具。

这个模块负责把 LLM 客户端、本地工具注册表、Runner 和 Session 装配起来。
这样 demo 脚本和未来入口文件只需要调用 create_agent_session()，不需要重复写组装代码。
"""

from agent_system.agent.legal_tools import build_legal_tools
from agent_system.agent.runner import AgentRunner
from agent_system.agent.session import AgentSession
from agent_system.agent.tools import ToolRegistry
from agent_system.agent.web_search_tools import build_web_search_tools
from agent_system.llm.factory import build_llm_client


def create_agent_session(system_prompt: str | None = None) -> AgentSession:
    """
    创建默认 Agent 会话。

    Args:
        system_prompt: 可选系统提示词。传入后会作为 AgentSession 的 system message。

    Returns:
        AgentSession: 已绑定默认 LLM 客户端、demo 工具和法条检索工具的 Agent 会话。
    """

    # Agent 层依赖 LLM 层，而不是让 LLM 层反向依赖 Agent 层。
    # 这样普通 ChatSession 仍然可以独立工作，项目分层更清晰。
    llm = build_llm_client()

    # 法条工具和 Web Search 工具都使用轻量注册：这里只暴露 schema，不主动执行检索或发外部请求。
    # 原因是 Agent 启动阶段应尽量快，把本地 RAG 加载和公网搜索成本都延迟到模型真正调用工具时。
    tools = ToolRegistry(build_legal_tools() + build_web_search_tools())
    runner = AgentRunner(llm=llm, tools=tools)
    return AgentSession(runner=runner, system_prompt=system_prompt)
