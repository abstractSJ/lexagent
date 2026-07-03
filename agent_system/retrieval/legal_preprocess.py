"""
法条数据预处理模块。

该模块只处理原始 JSON 数据到“可入库文档”的转换，不依赖 Chroma 或 embedding 模型。
这样做的原因是：清洗、解析、建库和查询属于不同阶段，把预处理逻辑独立出来，
后续无论换向量库还是换 embedding 模型，都可以复用同一套法条解析规则。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re
from typing import Any, Iterable


_TITLE_DATE_RE = re.compile(r"^(?P<name>.*?)(?P<date>\d{4}-\d{2}-\d{2})$")
_ARTICLE_NO_RE = re.compile(
    r"^(?P<article>第[一二三四五六七八九十百千万亿零〇两\d]+条(?:之[一二三四五六七八九十百千万亿零〇两\d]+)?)\s*"
)
_URL_RE = re.compile(r'https?://[^\s，。；、）)\]\}"]+')
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class LegalArticleDocument:
    """
    一条准备写入 Chroma 的法条文档。

    Attributes:
        id: 稳定文档 ID。重复建库时同一条法条会得到同一个 ID，便于覆盖写入。
        document: 写入向量库并参与语义检索的文本，包含法律名称、条号和正文。
        metadata: Chroma metadata，只保存字符串、数字和布尔值等标量字段。
    """

    id: str
    document: str
    metadata: dict[str, str | int | bool]


def load_raw_legal_rows(data_path: str | Path) -> list[list[str]]:
    """
    读取原始法条 JSON 数据。

    Args:
        data_path: 原始法条 JSON 文件路径。

    Returns:
        list[list[str]]: 原始三元组列表，每条记录为 [分类, 标题, 正文]。

    Raises:
        ValueError: JSON 顶层结构不是 list，或记录不是三字段结构时抛出。
    """

    path = Path(data_path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError(f"法条数据顶层必须是 list：{path}")

    rows: list[list[str]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"第 {index} 条记录必须是长度为 3 的 list。")

        category, title, text = row
        rows.append([str(category), str(title), str(text)])

    return rows


def build_legal_documents(
    data_path: str | Path,
    *,
    include_source_types: Iterable[str] | None = ("law_article",),
) -> list[LegalArticleDocument]:
    """
    从原始 JSON 构造可写入 Chroma 的法条文档。

    Args:
        data_path: 原始法条 JSON 文件路径。
        include_source_types: 需要保留的内容类型。默认只保留正式法条；传入 None 表示全部保留。

    Returns:
        list[LegalArticleDocument]: 清洗、解析和补充 metadata 后的文档列表。
    """

    rows = load_raw_legal_rows(data_path)
    include_set = set(include_source_types) if include_source_types is not None else None

    title_to_record_indexes: dict[str, list[int]] = defaultdict(list)
    for record_index, row in enumerate(rows):
        title_to_record_indexes[row[1]].append(record_index)

    title_positions: dict[int, int] = {}
    title_prev_next: dict[int, tuple[int, int]] = {}
    for indexes in title_to_record_indexes.values():
        for position, record_index in enumerate(indexes):
            title_positions[record_index] = position
            prev_record_index = indexes[position - 1] if position > 0 else -1
            next_record_index = indexes[position + 1] if position + 1 < len(indexes) else -1
            title_prev_next[record_index] = (prev_record_index, next_record_index)

    documents: list[LegalArticleDocument] = []
    source_path = str(Path(data_path))

    for record_index, (category, raw_title, raw_text) in enumerate(rows):
        text_clean = clean_text(raw_text)
        has_format_artifact = detect_format_artifact(raw_text)
        article_no = extract_article_no(text_clean)
        is_article = bool(article_no)
        legal_name, version_date, part_title = parse_legal_title(category, raw_title)
        source_type = classify_source_type(
            raw_title=raw_title,
            text_clean=text_clean,
            article_no=article_no,
            has_format_artifact=has_format_artifact,
        )

        if include_set is not None and source_type not in include_set:
            continue

        if not text_clean:
            continue

        text_hash = make_text_hash(text_clean)
        prev_record_index, next_record_index = title_prev_next.get(record_index, (-1, -1))
        document_text = build_document_text(
            category=category,
            legal_name=legal_name,
            raw_title=raw_title,
            version_date=version_date,
            article_no=article_no,
            part_title=part_title,
            text_clean=text_clean,
        )

        metadata: dict[str, str | int | bool] = {
            "source_path": source_path,
            "record_index": record_index,
            "category": category,
            "title": raw_title,
            "legal_name": legal_name,
            "version_date": version_date,
            "article_no": article_no,
            "part_title": part_title,
            "chunk_index_in_title": title_positions.get(record_index, 0),
            "prev_record_index": prev_record_index,
            "next_record_index": next_record_index,
            "source_type": source_type,
            "char_len": len(text_clean),
            "is_article": is_article,
            "has_format_artifact": has_format_artifact,
            "text_hash": text_hash,
            "text": text_clean,
            "citation": build_citation(legal_name, article_no),
        }

        documents.append(
            LegalArticleDocument(
                id=f"law:{record_index}:{text_hash[:8]}",
                document=document_text,
                metadata=metadata,
            )
        )

    return documents


def parse_legal_title(category: str, raw_title: str) -> tuple[str, str, str]:
    """
    解析法律名称、版本日期和民法典分编标题。

    Args:
        category: 原始数据第一列分类。
        raw_title: 原始数据第二列标题。

    Returns:
        tuple[str, str, str]: legal_name、version_date、part_title。没有的字段返回空字符串。
    """

    title = raw_title.strip()

    if category == "民法典":
        # 民法典数据的第二列是“合同编/物权编/总则”等分编，不是独立法律名称。
        # 检索文本中补充“民法典”能够避免模型把“合同编”误解为普通章节标题。
        return "民法典", "", title

    match = _TITLE_DATE_RE.match(title)
    if match:
        return match.group("name").strip(), match.group("date"), ""

    return title, "", ""


def clean_text(raw_text: str) -> str:
    """
    对正文做轻量清洗。

    Args:
        raw_text: 原始正文。

    Returns:
        str: 清洗后的正文。
    """

    text = raw_text.strip()

    # 少量攻略/标准数据混入了类似 JSON list 的残留。优先尝试按 JSON 解析，
    # 解析成功时把列表片段合并为自然文本；解析失败时只做保守的外层符号清理。
    if text.startswith('["'):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, list):
            text = "，".join(str(item).strip() for item in parsed if str(item).strip())
        else:
            text = text.strip("[]")
            text = text.replace('","', "，").replace("\"", "")

    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    elif text.startswith('"'):
        text = text[1:]
    elif text.endswith('"'):
        text = text[:-1]

    text = _URL_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip(" ，,;；")


def detect_format_artifact(raw_text: str) -> bool:
    """
    判断正文是否带有明显格式残留。

    Args:
        raw_text: 原始正文。

    Returns:
        bool: 存在明显格式残留时返回 True。
    """

    text = raw_text.strip()
    if text == "[":
        return True
    if text.startswith('["'):
        return True
    if '","' in text:
        return True
    if text.startswith('"') or text.endswith('"'):
        return True
    if _URL_RE.search(text):
        return True
    return False


def extract_article_no(text_clean: str) -> str:
    """
    从正文开头提取条号。

    Args:
        text_clean: 清洗后的正文。

    Returns:
        str: 条号，例如“第八十二条”；没有条号时返回空字符串。
    """

    match = _ARTICLE_NO_RE.match(text_clean)
    if not match:
        return ""
    return match.group("article")


def classify_source_type(
    *,
    raw_title: str,
    text_clean: str,
    article_no: str,
    has_format_artifact: bool,
) -> str:
    """
    将原始记录归类为正式法条、攻略、标准、前言或格式残留。

    Args:
        raw_title: 原始标题。
        text_clean: 清洗后的正文。
        article_no: 已提取的条号。
        has_format_artifact: 是否存在格式残留。

    Returns:
        str: source_type 字段值。
    """

    if not text_clean or text_clean == "[":
        return "artifact"

    if raw_title == "劳动仲裁和劳动诉讼的攻略":
        return "guide"

    if "鉴定标准" in raw_title:
        return "standard"

    if article_no:
        return "law_article"

    # 只有在不是正式条文时，格式残留才作为 artifact 处理。
    # 原因是少量正式条文内部可能引用带引号或括号的文件名，不能因为符号就丢弃法条。
    if has_format_artifact:
        return "artifact"

    return "preface_or_misc"


def build_document_text(
    *,
    category: str,
    legal_name: str,
    raw_title: str,
    version_date: str,
    article_no: str,
    part_title: str,
    text_clean: str,
) -> str:
    """
    构造写入向量库的 document 文本。

    Args:
        category: 原始分类。
        legal_name: 法律名称。
        raw_title: 原始标题。
        version_date: 版本日期，没有则为空字符串。
        article_no: 条号，没有则为空字符串。
        part_title: 民法典分编标题，没有则为空字符串。
        text_clean: 清洗后的正文。

    Returns:
        str: 带上下文的检索文本。
    """

    display_name = f"{legal_name}·{part_title}" if part_title else legal_name
    lines = [
        f"分类：{category}",
        f"法律名称：{display_name}",
        f"标题：{raw_title}",
    ]

    if version_date:
        lines.append(f"版本日期：{version_date}")
    if article_no:
        lines.append(f"条文：{article_no}")

    lines.append("")
    lines.append(text_clean)
    return "\n".join(lines)


def build_citation(legal_name: str, article_no: str) -> str:
    """
    构造法条引用文本。

    Args:
        legal_name: 法律名称。
        article_no: 条号。

    Returns:
        str: 例如“《劳动合同法》第八十二条”。
    """

    if article_no:
        return f"《{legal_name}》{article_no}"
    return f"《{legal_name}》相关条款"


def make_text_hash(text: str) -> str:
    """
    计算正文哈希。

    Args:
        text: 清洗后的正文。

    Returns:
        str: SHA-256 十六进制摘要。
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def calculate_file_sha256(data_path: str | Path) -> str:
    """
    计算文件 SHA-256，用于记录 Chroma collection 的数据来源版本。

    Args:
        data_path: 文件路径。

    Returns:
        str: 文件 SHA-256 十六进制摘要。
    """

    digest = hashlib.sha256()
    with Path(data_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
