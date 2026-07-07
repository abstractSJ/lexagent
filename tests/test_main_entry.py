"""
主入口行为测试。

这些测试只验证 main.py 的入口装配参数，不真正启动 Uvicorn、LLM、BGE-M3 或浏览器。
"""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main


class MainEntryTests(unittest.TestCase):
    """
    验证项目默认入口适合从 IDE 直接运行。
    """

    def parse_args(self, argv: list[str]):
        """
        用指定参数调用真实 parse_args。

        Args:
            argv: 不包含脚本名的参数列表。

        Returns:
            Namespace | None: 解析成功则返回参数对象；解析失败返回 None，便于测试用断言
            表达“当前入口还不支持该参数”。
        """

        with patch.object(sys, "argv", ["main.py", *argv]):
            try:
                return main.parse_args()
            except SystemExit:
                return None

    def test_default_entry_mode_is_web_ui(self) -> None:
        """
        未传命令行参数时，main.py 应默认进入本地 Web UI 启动模式。
        """

        args = self.parse_args([])

        self.assertEqual("web", getattr(args, "mode", None))

    def test_web_entry_accepts_host_port_and_browser_options(self) -> None:
        """
        Web UI 模式应允许在 IDE 运行配置里调整监听地址、端口和是否自动打开浏览器。
        """

        args = self.parse_args(["--host", "0.0.0.0", "--port", "9000", "--no-browser"])

        self.assertIsNotNone(args)
        self.assertEqual("web", getattr(args, "mode", None))
        self.assertEqual("0.0.0.0", getattr(args, "host", None))
        self.assertEqual(9000, getattr(args, "port", None))
        self.assertTrue(getattr(args, "no_browser", False))

    def test_chat_mode_keeps_existing_non_stream_option(self) -> None:
        """
        旧的普通聊天 CLI 仍应可通过 chat 模式使用，并保留 --non-stream 参数。
        """

        args = self.parse_args(["--mode", "chat", "--non-stream"])

        self.assertIsNotNone(args)
        self.assertEqual("chat", getattr(args, "mode", None))
        self.assertTrue(getattr(args, "non_stream", False))

    def test_run_web_server_invokes_uvicorn_factory(self) -> None:
        """
        Web 启动函数应调用 Uvicorn 的应用工厂入口，避免测试或导入 main.py 时提前初始化真实资源。
        """

        run_web_server = getattr(main, "run_web_server", None)
        self.assertTrue(callable(run_web_server))
        if not callable(run_web_server):
            return
        args = SimpleNamespace(host="127.0.0.1", port=8765, reload=False, no_browser=True)

        with (
            patch("uvicorn.run") as run,
            patch("webbrowser.open") as open_browser,
            patch("builtins.print"),
        ):
            run_web_server(args)

        open_browser.assert_not_called()
        run.assert_called_once_with(
            "web_app.server:create_app",
            factory=True,
            host="127.0.0.1",
            port=8765,
            reload=False,
        )

    def test_run_web_server_prints_embedding_device(self) -> None:
        """
        Web UI 启动时应展示当前 embedding 设备，方便用户判断本地 RAG 预热将使用 CPU 还是 CUDA。
        """

        args = SimpleNamespace(host="127.0.0.1", port=8765, reload=False, no_browser=True)

        with (
            patch("uvicorn.run"),
            patch("webbrowser.open"),
            patch("builtins.print") as mocked_print,
        ):
            main.run_web_server(args)

        printed_lines = [" ".join(str(part) for part in call.args) for call in mocked_print.call_args_list]
        self.assertTrue(
            any("Embedding 设备：cpu" in line for line in printed_lines),
            f"Web 启动输出未展示 embedding 设备，实际输出：{printed_lines}",
        )


if __name__ == "__main__":
    unittest.main()
