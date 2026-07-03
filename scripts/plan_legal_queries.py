"""
把原始案情拆解为法律检索 query 计划。

运行方式：
    python scripts/plan_legal_queries.py --case-text "公司一直没签劳动合同，也没交社保"
    python scripts/plan_legal_queries.py --case-file case.txt --show-raw

这个脚本只负责调用一次性法律 query planner，输出结构化检索计划；
它不会直接执行 Chroma 检索，也不会生成最终法律意见。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    # 允许用户直接通过 `python scripts/plan_legal_queries.py` 运行脚本。
    # 原因是脚本位于 scripts 子目录，直接运行时 Python 默认不会把项目根目录加入模块搜索路径。
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_system.planning import (  # noqa: E402
    DEFAULT_MAX_REPAIR_ATTEMPTS,
    LegalQueryPlanError,
    plan_legal_queries,
    plan_to_dict,
)


def main() -> None:
    """
    解析命令行参数并执行一次性法律 query 规划。
    """

    args = parse_args()
    case_text = load_case_text(args)

    try:
        plan = plan_legal_queries(
            case_text,
            max_repair_attempts=args.repair_attempts,
        )
    except LegalQueryPlanError as error:
        print("法律 query planner 执行失败：", file=sys.stderr)
        print(f"- {error}", file=sys.stderr)
        for item in error.errors:
            print(f"- {item}", file=sys.stderr)
        if args.show_raw and error.raw_response:
            print("\n===== LLM 原始输出 =====", file=sys.stderr)
            print(error.raw_response, file=sys.stderr)
        raise SystemExit(1) from error
    except Exception as error:
        print(f"调用失败：{error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(
        json.dumps(
            plan_to_dict(plan, include_raw_response=args.show_raw),
            ensure_ascii=False,
            indent=None if args.compact else 2,
        )
    )


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    """

    parser = argparse.ArgumentParser(
        description="把一段案情拆解为结构化法律检索 query 计划。"
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--case-text",
        default="",
        help="直接传入案情文本。",
    )
    source_group.add_argument(
        "--case-file",
        default="",
        help="从 UTF-8 文本文件读取案情。",
    )

    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=DEFAULT_MAX_REPAIR_ATTEMPTS,
        help=f"JSON 解析或校验失败后的修复重试次数，默认 {DEFAULT_MAX_REPAIR_ATTEMPTS}。",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="在最终 JSON 中附带 LLM 原始输出，便于调试 prompt。",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="使用紧凑 JSON 输出，适合管道处理。",
    )
    return parser.parse_args()


def load_case_text(args: argparse.Namespace) -> str:
    """
    从命令行参数中读取案情文本。

    Args:
        args: 命令行参数对象。

    Returns:
        str: 原始案情文本。

    Raises:
        RuntimeError: 文件读取失败或案情为空时抛出。
    """

    if args.case_text:
        case_text = args.case_text.strip()
    else:
        file_path = Path(args.case_file)
        try:
            case_text = file_path.read_text(encoding="utf-8").strip()
        except Exception as error:
            raise RuntimeError(f"读取案情文件失败：{file_path}") from error

    if not case_text:
        raise RuntimeError("案情文本不能为空。")
    return case_text


if __name__ == "__main__":
    main()
