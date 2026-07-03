"""
项目内部 Chat-style message 构造工具。

这个模块只负责生成最常见的 system、user、assistant 消息字典。
这样上层代码不需要到处手写 role 字符串，也不需要关心底层最终使用
Chat Completions 还是 Responses API。
"""

from typing import Any, Dict


Message = Dict[str, Any]


def system_message(content: str) -> Message:
    """
    构造 system 消息。

    Args:
        content: 系统提示词，用来约束模型的整体行为和回答风格。

    Returns:
        Message: 项目内部 Chat-style system 消息。
    """

    return {"role": "system", "content": content}


def user_message(content: str) -> Message:
    """
    构造 user 消息。

    Args:
        content: 用户输入内容。

    Returns:
        Message: 项目内部 Chat-style user 消息。
    """

    return {"role": "user", "content": content}


def assistant_message(content: str) -> Message:
    """
    构造 assistant 消息。

    Args:
        content: 模型已经生成完成的回复内容。

    Returns:
        Message: 项目内部 Chat-style assistant 消息。
    """

    return {"role": "assistant", "content": content}
