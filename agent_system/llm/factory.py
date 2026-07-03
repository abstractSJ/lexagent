"""
LLM 对象创建工具。

这个模块把“读取配置 + 初始化客户端 + 创建会话”封装起来。
这样 main.py、测试脚本和未来 Agent 层都不需要重复写 load_llm_config() 与 OpenAIChatClient(...)。
"""

from agent_system.config import load_llm_config
from agent_system.llm.openai_client import OpenAIChatClient
from agent_system.llm.session import ChatSession


def build_llm_client() -> OpenAIChatClient:
    """
    创建默认 LLM 客户端。

    Returns:
        OpenAIChatClient: 已根据 agent_system/config.py 初始化好的 LLM 客户端。
    """

    # 这里统一读取项目配置。
    # 原因是调用方只应该关心“我要一个能用的 LLM”，不应该每个脚本都重复装配配置对象。
    config = load_llm_config()
    return OpenAIChatClient(config=config)


def create_chat_session(system_prompt: str | None = None) -> ChatSession:
    """
    创建默认聊天会话。

    Args:
        system_prompt: 可选系统提示词。传入后会作为会话第一条 system message。

    Returns:
        ChatSession: 已经绑定默认 LLM 客户端的聊天会话。
    """

    return ChatSession(
        llm=build_llm_client(),
        system_prompt=system_prompt,
    )
