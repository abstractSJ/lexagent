"""
补充信息弹窗交互回归测试。

运行前先执行 `npm run build` 生成最新前端静态产物。脚本会启动一个同时托管静态文件和
假后端 API 的本地 HTTP 服务，然后用 Playwright 验证：当后端返回 pause 并保持事件流短暂未结束时，
补充弹窗仍然可以关闭、输入和提交。
"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from playwright.sync_api import Page, expect, sync_playwright


ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT_DIR / "web_app" / "static"
WEB_HOST = "127.0.0.1"
WEB_PORT = 18080
WEB_URL = f"http://{WEB_HOST}:{WEB_PORT}"


class FakeLegalWebHandler(BaseHTTPRequestHandler):
    """
    为前端交互测试提供最小静态服务和后端 API。

    这里故意让 `/api/chat` 在发出 pause 后等待一段时间再发送 done。原因是需要模拟真实网络下
    “补充弹窗已经打开，但上一轮事件流尚未完全结束”的窗口，验证弹窗不会被发送状态误禁用。
    """

    server_version = "FakeLegalWeb/0.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        """
        静默 HTTP 访问日志，避免测试输出被无关请求刷屏。
        """

    def do_GET(self) -> None:
        """
        返回健康检查或静态前端资源。
        """

        if self.path == "/api/health":
            self.send_json({"ok": True, "service": "fake-legal-agent-web", "preloaded": True})
            return
        self.serve_static_file()

    def do_POST(self) -> None:
        """
        返回一个先 pause、后 done 的 NDJSON 聊天流。
        """

        if self.path != "/api/chat":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or "0")
        if length:
            self.rfile.read(length)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.write_ndjson(
            {
                "type": "pause",
                "reason": "请补充关键事实。",
                "message": "请先补充关键事实。",
                "questions": ["买方是否知道买的是毒品？"],
                "evidence_gaps": ["聊天记录"],
                "state_version": 1,
            }
        )
        # 留出足够时间让 Playwright 在 done 前检查弹窗交互状态。
        time.sleep(2)
        self.write_ndjson({"type": "done"})
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def serve_static_file(self) -> None:
        """
        按 FastAPI 的生产路径规则托管构建后的静态资源。
        """

        if self.path in {"/", ""}:
            file_path = STATIC_DIR / "index.html"
        elif self.path.startswith("/static/"):
            relative_path = self.path.removeprefix("/static/").split("?", 1)[0]
            file_path = STATIC_DIR / relative_path
        else:
            self.send_error(404)
            return

        try:
            resolved = file_path.resolve()
            resolved.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self.send_error(403)
            return
        if not resolved.is_file():
            self.send_error(404)
            return

        body = resolved.read_bytes()
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data: dict[str, Any]) -> None:
        """
        发送 JSON 响应。

        Args:
            data: 待序列化的响应对象。
        """

        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_ndjson(self, data: dict[str, Any]) -> None:
        """
        向 NDJSON 响应流写入一行并立即刷新。

        Args:
            data: 待写入的事件对象。
        """

        payload = json.dumps(data, ensure_ascii=False).encode("utf-8") + b"\n"
        self.wfile.write(f"{len(payload):X}\r\n".encode("ascii"))
        self.wfile.write(payload)
        self.wfile.write(b"\r\n")
        self.wfile.flush()


def wait_for_http(url: str, *, timeout: float = 10.0) -> None:
    """
    等待 HTTP 服务可访问。

    Args:
        url: 待探测 URL。
        timeout: 最长等待秒数。

    Raises:
        RuntimeError: 超时仍不可访问时抛出。
    """

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=1):
                return
        except Exception as error:  # noqa: BLE001 - 测试探测需要保留最后一次异常。
            last_error = error
            time.sleep(0.1)
    raise RuntimeError(f"等待服务超时：{url}，最后错误：{last_error}")


def exercise_dialog(page: Page) -> None:
    """
    操作页面并断言补充弹窗的关键交互可用。

    Args:
        page: Playwright 页面对象。
    """

    page.goto(WEB_URL)
    page.wait_for_load_state("networkidle")
    page.get_by_label("案情或追问").fill("有人诱导交易疑似毒品怎么办？")
    page.get_by_role("button", name="发送").click()

    expect(page.get_by_role("heading", name="请先补充关键信息")).to_be_visible(timeout=5000)
    expect(page.get_by_label("关闭补充信息弹窗")).to_be_visible()

    answer_input = page.get_by_placeholder("在这里回答这个问题。").first
    expect(answer_input).to_be_enabled()
    answer_input.fill("买方聊天中明确说这是冰毒。")

    submit_button = page.get_by_role("button", name="提交补充并继续")
    expect(submit_button).to_be_enabled()


def main() -> None:
    """
    执行补充弹窗交互测试。
    """

    if not (STATIC_DIR / "index.html").is_file():
        raise RuntimeError("未找到 web_app/static/index.html，请先运行 npm run build。")

    server = ThreadingHTTPServer((WEB_HOST, WEB_PORT), FakeLegalWebHandler)
    server_thread = threading.Thread(target=server.serve_forever, name="fake-legal-web", daemon=True)
    server_thread.start()
    try:
        wait_for_http(WEB_URL)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                exercise_dialog(page)
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
