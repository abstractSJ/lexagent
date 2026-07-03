"""
项目默认启动入口。

直接在 IDE 中运行本文件时，默认启动法律咨询本地 Web UI，不需要再手动输入
`python -m uvicorn web_app.server:app --reload`。旧的普通命令行聊天入口仍保留在
`--mode chat` 下，方便继续测试最小 ChatSession 链路。
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
import webbrowser

from agent_system.llm import create_chat_session


SYSTEM_PROMPT = "你是一个耐心的 Python Agent 系统学习助手，请用中文回答。"
DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8000
WEB_APP_FACTORY = "web_app.server:create_app"


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    """
    解析项目入口参数。

    Args:
        argv: 可选参数列表；传入 None 时由 argparse 自动读取当前进程命令行参数。

    Returns:
        Namespace: CLI 参数对象，包含启动模式、Web 监听配置和旧聊天模式选项。
    """

    parser = ArgumentParser(description="Agent Learning 项目统一入口。")
    parser.add_argument(
        "--mode",
        choices=("web", "chat"),
        default="web",
        help="启动模式：web 为法律咨询本地 Web UI；chat 为旧的最小命令行聊天程序。",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_WEB_HOST,
        help="Web UI 监听地址；默认只监听本机，避免本地调试服务意外暴露到局域网。",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_WEB_PORT,
        help="Web UI 监听端口。",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="启动 Web UI 后不自动打开浏览器，适合已经手动打开页面或仅想启动后端服务的场景。",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="开启 Uvicorn reload；开发后端代码时可用，普通 IDE 运行不需要开启。",
    )
    parser.add_argument(
        "--non-stream",
        action="store_true",
        help="chat 模式下使用非流式 Responses API 调用，方便测试兼容服务的非流式接口。",
    )
    return parser.parse_args(argv)


def build_web_url(host: str, port: int) -> str:
    """
    构造浏览器访问地址。

    Args:
        host: Uvicorn 监听地址。
        port: Uvicorn 监听端口。

    Returns:
        str: 可在本机浏览器直接打开的 URL。
    """

    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{browser_host}:{port}/"


def run_web_server(args: Namespace) -> None:
    """
    启动法律咨询 Web UI。

    Args:
        args: 由 parse_args 生成的参数对象。
    """

    import uvicorn

    url = build_web_url(args.host, args.port)
    print("法律咨询 Web UI 正在启动。")
    print(f"访问地址：{url}")
    print("关闭窗口或按 Ctrl+C 可停止服务。")

    if not args.no_browser:
        # 先打开浏览器再进入阻塞式服务循环。这样在 IDE 中点击运行后，用户可以直接看到页面。
        webbrowser.open(url)

    # 使用应用工厂字符串而不是提前 create_app()。这样做可以避免导入 main.py 或运行单测时初始化
    # 真实 LLM/RAG 资源，也兼容 Uvicorn reload 对 import string 的要求。
    uvicorn.run(WEB_APP_FACTORY, factory=True, host=args.host, port=args.port, reload=args.reload)


def run_chat_cli(args: Namespace) -> None:
    """
    启动旧的最小命令行聊天程序。

    Args:
        args: 由 parse_args 生成的参数对象；仅使用其中的 non_stream 选项。
    """

    session = create_chat_session(system_prompt=SYSTEM_PROMPT)

    if args.non_stream:
        print("最小非流式 LLM 聊天程序已启动。输入 exit 或 quit 退出。")
    else:
        print("最小流式 LLM 聊天程序已启动。输入 exit 或 quit 退出。")

    while True:
        user_input = input("\n用户：").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("程序已退出。")
            break

        if not user_input:
            print("请输入非空内容。")
            continue

        try:
            if args.non_stream:
                answer = session.ask_non_stream(user_input)
                print(f"\n助手：{answer}")
            else:
                print("\n助手：", end="", flush=True)
                for text in session.stream_ask(user_input):
                    print(text, end="", flush=True)
                print()
        except Exception as error:
            # ChatSession 内部已经完成失败回滚；这里保留异常提示，方便 CLI 使用者知道本轮失败。
            print(f"\n调用失败：{error}")


def main(argv: Sequence[str] | None = None) -> None:
    """
    根据入口参数启动项目。

    Args:
        argv: 可选参数列表；IDE 直接运行时保持 None 即可使用默认 Web UI 模式。
    """

    args = parse_args(argv)
    if args.mode == "chat":
        run_chat_cli(args)
        return
    run_web_server(args)


if __name__ == "__main__":
    main()
