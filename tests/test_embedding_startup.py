"""
Embedding 启动配置与提示测试。

这些测试只验证默认配置和 CLI 启动文案，不加载真实 BGE-M3、Chroma 或 LLM。
"""

from __future__ import annotations

import importlib
import os
import unittest
from unittest.mock import patch


class EmbeddingStartupTests(unittest.TestCase):
    """
    验证法律咨询链路默认使用稳妥的本地 embedding 设备，并在启动时明确展示。
    """

    def test_default_embedding_device_is_cpu_when_env_is_unset(self) -> None:
        """
        未设置 LEGAL_EMBEDDING_DEVICE 时，应默认使用 CPU，避免 Windows/CUDA 环境不稳定导致启动卡顿或黑屏。
        """

        with patch.dict(os.environ, {}, clear=True):
            import agent_system.config as config

            reloaded_config = importlib.reload(config)

        try:
            self.assertEqual("cpu", reloaded_config.EMBEDDING_CONFIG.device)
        finally:
            # 恢复模块级配置，避免本测试的环境隔离影响后续测试读取真实环境变量。
            importlib.reload(reloaded_config)

    def test_legal_cli_prints_embedding_device_before_preload(self) -> None:
        """
        法律咨询 CLI 启动/预热前应展示当前 embedding 设备，方便用户判断慢启动是否来自本地模型加载。
        """

        class FakeLegalSession:
            """
            只记录预热调用的假法律咨询会话。
            """

            def __init__(self) -> None:
                self.preload_called = False

            def preload_resources(self) -> None:
                """
                模拟预热资源，不访问真实模型或向量库。
                """

                self.preload_called = True

        fake_session = FakeLegalSession()

        with (
            patch("legal_agent_demo.create_legal_consultation_session", return_value=fake_session),
            patch("builtins.input", return_value="exit"),
            patch("builtins.print") as mocked_print,
            patch.dict(os.environ, {"LEGAL_RAG_PRELOAD": "1"}, clear=False),
        ):
            import legal_agent_demo

            legal_agent_demo.main()

        printed_lines = [" ".join(str(part) for part in call.args) for call in mocked_print.call_args_list]
        self.assertTrue(fake_session.preload_called)
        self.assertTrue(
            any("Embedding 设备：cpu" in line for line in printed_lines),
            f"启动输出未展示 embedding 设备，实际输出：{printed_lines}",
        )


if __name__ == "__main__":
    unittest.main()
