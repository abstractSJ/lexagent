"""
本地工具定义与注册表。

工具层只负责两件事：
1. 把本地 Python 函数描述成 Responses API 可识别的 function tool schema。
2. 按模型给出的工具名和参数执行对应函数。

这样 AgentRunner 只需要关心“调用哪个工具”和“把结果回传给模型”，不需要知道每个工具内部怎么实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


ToolHandler = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class LocalTool:
    """
    本地工具定义。

    Args:
        name: 工具名。必须和模型返回的 function_call.name 一致。
        description: 工具说明。模型会根据它判断什么时候调用工具。
        parameters: JSON Schema 参数定义。
        handler: 本地执行函数，接收已解析好的参数字典。
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def to_openai_tool(self) -> dict[str, Any]:
        """
        转换为 Responses API function tool 结构。

        Returns:
            dict[str, Any]: 可传给 client.responses.create(..., tools=[...]) 的工具定义。
        """

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }


class ToolRegistry:
    """
    本地工具注册表。

    Registry 是工具名到 LocalTool 的映射。这样模型返回 name 后，AgentRunner 可以统一查找并执行工具。
    """

    def __init__(self, tools: list[LocalTool] | None = None) -> None:
        """
        初始化工具注册表。

        Args:
            tools: 可选初始工具列表。
        """

        self._tools: dict[str, LocalTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: LocalTool) -> None:
        """
        注册一个本地工具。

        Args:
            tool: 待注册工具。

        Raises:
            ValueError: 工具名为空或重复时抛出。
        """

        if not tool.name.strip():
            raise ValueError("工具名不能为空。")

        if tool.name in self._tools:
            raise ValueError(f"工具名重复：{tool.name}")

        self._tools[tool.name] = tool

    def get(self, name: str) -> LocalTool:
        """
        根据名称获取工具。

        Args:
            name: 工具名。

        Returns:
            LocalTool: 对应工具。

        Raises:
            KeyError: 工具不存在时抛出。
        """

        return self._tools[name]

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """
        返回所有工具的 Responses API schema。

        Returns:
            list[dict[str, Any]]: 工具定义列表。
        """

        return [tool.to_openai_tool() for tool in self._tools.values()]

    def run(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        执行指定工具。

        Args:
            name: 工具名。
            arguments: 已解析好的工具参数。

        Returns:
            Any: 工具执行结果。失败时返回包含 ok=false 的字典，而不是直接中断模型工具循环。
        """

        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"未知工具：{name}"}

        if not isinstance(arguments, dict):
            return {"ok": False, "error": "工具参数必须是 JSON object。"}

        try:
            return tool.handler(arguments)
        except Exception as error:
            # 工具异常要回传给模型，让模型有机会解释或修正，而不是让整个 Agent 直接崩掉。
            return {"ok": False, "error": f"工具 {name} 执行失败：{error}"}
