"""
Agent 层对外入口。

这里暴露任务执行型 Agent 的核心组件：事件、工具、Runner、Session 和默认工厂。
普通聊天仍然走 agent_system.llm；工具调用和任务执行走 agent_system.agent。
"""

from agent_system.agent.events import AgentEvent
from agent_system.agent.factory import create_agent_session
from agent_system.agent.legal_tools import build_legal_tools
from agent_system.agent.runner import AgentRunOptions, AgentRunner
from agent_system.agent.session import AgentSession
from agent_system.agent.tools import LocalTool, ToolRegistry
from agent_system.agent.web_search_tools import build_web_search_tools

__all__ = [
    "AgentEvent",
    "AgentRunOptions",
    "AgentRunner",
    "AgentSession",
    "LocalTool",
    "ToolRegistry",
    "build_legal_tools",
    "build_web_search_tools",
    "create_agent_session",
]
