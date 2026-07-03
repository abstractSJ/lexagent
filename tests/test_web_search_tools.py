"""
博查 Web Search 工具单元测试。

这些测试只验证本地工具封装、参数整理和响应裁剪，不调用真实博查接口，避免测试依赖公网和真实额度。
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agent_system.agent.tools import ToolRegistry
from agent_system.agent.web_search_tools import build_web_search_tools, clamp_count, normalize_web_search_response
from agent_system.config import WebSearchConfig


class WebSearchToolTests(unittest.TestCase):
    """
    测试博查 Web Search LocalTool 封装。
    """

    def build_config(self) -> WebSearchConfig:
        """
        构造测试用 Web Search 配置。

        Returns:
            WebSearchConfig: 不含真实密钥依赖的测试配置。
        """

        return WebSearchConfig(
            api_key="test-key",
            endpoint="https://example.test/v1/web-search",
            timeout=3.0,
            default_count=10,
            max_count=20,
            default_summary=True,
            default_freshness="noLimit",
        )

    def test_tool_schema_is_strict_function_tool(self) -> None:
        """
        web_search 应沿用项目现有 strict function tool schema 风格。
        """

        tool = build_web_search_tools(config=self.build_config())[0]
        schema = tool.to_openai_tool()

        self.assertEqual("function", schema["type"])
        self.assertEqual("web_search", schema["name"])
        self.assertTrue(schema["strict"])
        self.assertEqual("object", schema["parameters"]["type"])
        self.assertFalse(schema["parameters"]["additionalProperties"])
        self.assertEqual(
            ["query", "count", "freshness", "summary", "include", "exclude"],
            schema["parameters"]["required"],
        )

    def test_tool_description_mentions_legal_research_use_cases(self) -> None:
        """
        web_search 描述应覆盖法律实务检索场景，帮助模型正确选择工具。
        """

        tool = build_web_search_tools(config=self.build_config())[0]

        for keyword in ["最高人民法院", "指导性案例", "司法解释", "量刑", "赔偿", "执行"]:
            self.assertIn(keyword, tool.description)

    def test_handler_posts_payload_and_normalizes_response(self) -> None:
        """
        handler 应整理模型参数、限制 count，并返回裁剪后的网页结果。
        """

        fake_response = {
            "code": 200,
            "log_id": "log-1",
            "data": {
                "queryContext": {"originalQuery": "上海最低工资"},
                "webPages": {
                    "webSearchUrl": "https://bochaai.com/search?q=test",
                    "totalEstimatedMatches": 123,
                    "someResultsRemoved": False,
                    "value": [
                        {
                            "name": "上海最低工资标准",
                            "url": "https://example.com/a",
                            "displayUrl": "example.com/a",
                            "snippet": "摘要片段",
                            "summary": "长摘要" * 500,
                            "siteName": "示例站点",
                            "dateLastCrawled": "2026-07-01T00:00:00Z",
                        }
                    ],
                },
            },
        }
        tool = build_web_search_tools(config=self.build_config())[0]

        with patch("agent_system.agent.web_search_tools.post_web_search_request", return_value=fake_response) as mock_post:
            result = tool.handler(
                {
                    "query": "上海最低工资",
                    "count": 999,
                    "freshness": "oneYear",
                    "summary": True,
                }
            )

        self.assertTrue(result["ok"])
        self.assertEqual("上海最低工资", result["query"])
        self.assertEqual(1, len(result["results"]))
        self.assertEqual("上海最低工资标准", result["results"][0]["title"])
        self.assertLessEqual(len(result["results"][0]["summary"]), 1203)
        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["payload"]
        self.assertEqual(
            {"query": "上海最低工资", "count": 20, "summary": True, "freshness": "oneYear"},
            payload,
        )

    def test_empty_query_returns_error_without_request(self) -> None:
        """
        query 为空时不应发外部请求，直接返回可被模型理解的错误。
        """

        tool = build_web_search_tools(config=self.build_config())[0]

        with patch("agent_system.agent.web_search_tools.post_web_search_request") as mock_post:
            result = tool.handler({"query": "", "count": 10, "freshness": "noLimit", "summary": True})

        self.assertFalse(result["ok"])
        self.assertIn("query", result["error"])
        mock_post.assert_not_called()

    def test_include_and_exclude_forwarded_to_payload(self) -> None:
        """
        include/exclude 站点过滤参数应透传给博查接口；为空字符串时不进入 payload。
        """

        fake_response = {"code": 200, "data": {"webPages": {"value": []}}}
        tool = build_web_search_tools(config=self.build_config())[0]

        with patch("agent_system.agent.web_search_tools.post_web_search_request", return_value=fake_response) as mock_post:
            tool.handler(
                {
                    "query": "劳动争议 司法解释",
                    "count": 10,
                    "freshness": "noLimit",
                    "summary": True,
                    "include": "court.gov.cn|chinacourt.org",
                    "exclude": "",
                }
            )

        payload = mock_post.call_args.kwargs["payload"]
        self.assertEqual("court.gov.cn|chinacourt.org", payload["include"])
        self.assertNotIn("exclude", payload)

    def test_registry_can_run_web_search_tool(self) -> None:
        """
        Web Search 工具应能被 ToolRegistry 正常注册和调度。
        """

        registry = ToolRegistry(build_web_search_tools(config=self.build_config()))
        fake_response = {"code": 200, "data": {"webPages": {"value": []}}}

        with patch("agent_system.agent.web_search_tools.post_web_search_request", return_value=fake_response):
            result = registry.run(
                "web_search",
                {"query": "测试", "count": 5, "freshness": "noLimit", "summary": False},
            )

        self.assertTrue(result["ok"])
        self.assertEqual([], result["results"])

    def test_clamp_count_uses_default_and_max(self) -> None:
        """
        count 非法时用默认值，超过上限时截断到最大值。
        """

        config = self.build_config()

        self.assertEqual(10, clamp_count("bad", config))
        self.assertEqual(10, clamp_count(0, config))
        self.assertEqual(20, clamp_count(99, config))
        self.assertEqual(7, clamp_count("7", config))

    def test_non_200_api_code_becomes_error_result(self) -> None:
        """
        博查业务 code 非 200 时，应返回 ok=false 而不是伪装成空结果成功。
        """

        result = normalize_web_search_response({"code": 401, "log_id": "log-2", "msg": "invalid token"})

        self.assertFalse(result["ok"])
        self.assertEqual(401, result["code"])
        self.assertEqual("invalid token", result["error"])

    def test_non_200_api_code_reads_official_message_field(self) -> None:
        """
        博查官方错误响应体字段名是 message；msg 缺失时应读取 message，避免拿不到错误文本。
        """

        result = normalize_web_search_response({"code": 403, "log_id": "log-3", "message": "余额不足"})

        self.assertFalse(result["ok"])
        self.assertEqual("余额不足", result["error"])


if __name__ == "__main__":
    unittest.main()
