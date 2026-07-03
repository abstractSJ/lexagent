"""
LLM 模块对外入口。

常用层级：
1. build_llm_client()：创建默认 LLM 客户端。
2. create_chat_session()：创建带历史管理的聊天会话。
3. OpenAIChatClient：需要更细控制单次请求时直接使用；类名保留兼容，内部使用 Responses API。
"""

from agent_system.llm.factory import build_llm_client, create_chat_session
from agent_system.llm.messages import assistant_message, system_message, user_message
from agent_system.llm.openai_client import ImagePath, Message, OpenAIChatClient
from agent_system.llm.session import ChatSession

__all__ = [
    "Message",
    "ImagePath",
    "OpenAIChatClient",
    "ChatSession",
    "build_llm_client",
    "create_chat_session",
    "system_message",
    "user_message",
    "assistant_message",
]
