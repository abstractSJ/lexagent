"""
本地会话持久化存储层。

当前实现服务于法律咨询 Web 链路，使用轻量文件存储：

```text
data/sessions/<session_id>/
  meta.json       会话元数据（标题、时间、轮次），低频更新
  snapshot.json   可恢复主状态，一轮成功后原子覆盖
  events.jsonl    append-only 过程事件日志
```

设计要点：

1. snapshot 必须原子写（先写 tmp 再 rename），崩溃时不会留下半个 JSON。
2. events.jsonl 只追加不改写，每行一个带 seq/ts 的 JSON 对象。
3. 存储层不理解业务 schema：快照内容由调用方组装，这里只负责文件可靠读写。
4. session_id 必须通过白名单校验后才拼接路径，避免 `../` 路径穿越。
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

META_SCHEMA_VERSION = "legal_session_meta.v1"
SNAPSHOT_SCHEMA_VERSION = "legal_session_snapshot.v1"
EVENT_SCHEMA_VERSION = "legal_session_event.v1"

META_FILENAME = "meta.json"
SNAPSHOT_FILENAME = "snapshot.json"
EVENTS_FILENAME = "events.jsonl"

# session_id 白名单格式：sess_日期_时间_短随机后缀。
# 目录名同时承担“按时间自然排序”和“肉眼可读”两个职责；正则校验是路径安全的第一道防线。
SESSION_ID_PATTERN = re.compile(r"^sess_\d{8}_\d{6}_[0-9a-f]{4}$")


class SessionStoreError(Exception):
    """
    会话存储层错误。

    统一异常类型的原因是调用方（Web 层）只需要区分“存储操作失败”，并把消息透传给
    前端或日志，不需要分别捕获 OSError/JSONDecodeError 等底层异常。
    """


class SessionStore:
    """
    基于本地文件的会话存储。

    Args:
        base_dir: 会话根目录，例如 `data/sessions`；不存在时自动创建。
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # 事件 seq 按 session 缓存在内存中。首次追加时从现有文件行数初始化，
        # 后续递增；当前 Web 层用全局锁串行处理请求，这里再加一把锁做双保险。
        self._event_seq_cache: dict[str, int] = {}
        self._lock = threading.Lock()

    def create_session(self, *, title: str = "") -> str:
        """
        创建新会话目录和初始 meta.json。

        Args:
            title: 可选初始标题；通常首轮成功后再由调用方更新。

        Returns:
            str: 新分配的 session_id。
        """

        with self._lock:
            for _ in range(8):
                session_id = self._generate_session_id()
                session_dir = self.base_dir / session_id
                if session_dir.exists():
                    # 同秒并发创建可能撞名；短随机后缀 + 重试足以解决本地单用户场景。
                    continue
                session_dir.mkdir(parents=True)
                now = utc_now_text()
                meta = {
                    "schema_version": META_SCHEMA_VERSION,
                    "session_id": session_id,
                    "title": str(title or ""),
                    "created_at": now,
                    "updated_at": now,
                    "turn_count": 0,
                }
                atomic_write_json(session_dir / META_FILENAME, meta)
                self._event_seq_cache[session_id] = 0
                self._append_event_unlocked(session_id, "session_created", {"title": meta["title"]})
                return session_id
        raise SessionStoreError("无法分配新的 session_id，请重试。")

    def session_exists(self, session_id: str) -> bool:
        """
        判断会话是否存在（以 meta.json 为准）。
        """

        try:
            session_dir = self._session_dir(session_id)
        except SessionStoreError:
            return False
        return (session_dir / META_FILENAME).is_file()

    def save_snapshot(
        self,
        session_id: str,
        snapshot: dict[str, Any],
        *,
        title: str | None = None,
        turn_count: int | None = None,
    ) -> None:
        """
        原子写入会话快照，并同步更新 meta 的时间戳、标题和轮次。

        Args:
            session_id: 会话 ID。
            snapshot: 业务层组装的可恢复状态；存储层只附加 schema/时间戳字段。
            title: 可选新标题；None 表示保持 meta 现状。
            turn_count: 可选轮次计数；None 表示保持 meta 现状。
        """

        session_dir = self._require_session_dir(session_id)
        now = utc_now_text()
        payload = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "session_id": session_id,
            **snapshot,
            "updated_at": now,
        }
        atomic_write_json(session_dir / SNAPSHOT_FILENAME, payload)

        meta = self.load_meta(session_id)
        meta["updated_at"] = now
        if title is not None:
            meta["title"] = str(title)
        if turn_count is not None:
            meta["turn_count"] = int(turn_count)
        atomic_write_json(session_dir / META_FILENAME, meta)

    def load_snapshot(self, session_id: str) -> dict[str, Any] | None:
        """
        读取会话快照。

        Returns:
            dict | None: 快照内容；会话存在但从未成功保存快照时返回 None。

        Raises:
            SessionStoreError: 会话不存在、快照损坏或 schema 大版本不兼容时抛出。
        """

        session_dir = self._require_session_dir(session_id)
        snapshot_path = session_dir / SNAPSHOT_FILENAME
        if not snapshot_path.is_file():
            return None
        data = read_json_file(snapshot_path)
        validate_schema_version(data, SNAPSHOT_SCHEMA_VERSION, snapshot_path)
        return data

    def load_meta(self, session_id: str) -> dict[str, Any]:
        """
        读取会话元数据。

        Raises:
            SessionStoreError: 会话不存在或 meta 损坏时抛出。
        """

        session_dir = self._require_session_dir(session_id)
        meta_path = session_dir / META_FILENAME
        if not meta_path.is_file():
            raise SessionStoreError(f"会话 {session_id} 缺少 meta.json。")
        data = read_json_file(meta_path)
        validate_schema_version(data, META_SCHEMA_VERSION, meta_path)
        return data

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        扫描根目录返回会话元数据列表，按 updated_at 倒序。

        单条 meta 损坏时跳过该会话而不是整体失败，避免一个坏文件让历史列表不可用。
        """

        sessions: list[dict[str, Any]] = []
        for entry in self.base_dir.iterdir():
            if not entry.is_dir() or not SESSION_ID_PATTERN.match(entry.name):
                continue
            try:
                meta = self.load_meta(entry.name)
            except SessionStoreError:
                continue
            sessions.append(meta)
        sessions.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> None:
        """
        删除整个会话目录。

        Raises:
            SessionStoreError: 会话不存在或删除失败时抛出。
        """

        session_dir = self._require_session_dir(session_id)
        try:
            shutil.rmtree(session_dir)
        except OSError as error:
            raise SessionStoreError(f"删除会话 {session_id} 失败：{error}") from error
        with self._lock:
            self._event_seq_cache.pop(session_id, None)

    def append_event(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any] | None = None,
        *,
        turn_id: int | None = None,
    ) -> None:
        """
        向 events.jsonl 追加一条过程事件。

        Args:
            session_id: 会话 ID。
            event_type: 事件类型，例如 turn_committed、turn_failed。
            data: 事件负载。
            turn_id: 可选轮次编号；会话级事件可以不带。
        """

        with self._lock:
            self._append_event_unlocked(session_id, event_type, data, turn_id=turn_id)

    def read_events(self, session_id: str) -> list[dict[str, Any]]:
        """
        按写入顺序读取会话的全部过程事件。

        Args:
            session_id: 会话 ID。

        Returns:
            list[dict[str, Any]]: 事件对象列表；events.jsonl 尚不存在时返回空列表。
            单行 JSON 损坏时跳过该行而不是整体失败：事件日志是观测与审计数据，
            指标聚合宁可少统计一行，也不能因一个坏行让整个接口不可用。

        Raises:
            SessionStoreError: 会话不存在或事件文件不可读时抛出。
        """

        session_dir = self._require_session_dir(session_id)
        events_path = session_dir / EVENTS_FILENAME
        if not events_path.is_file():
            return []
        events: list[dict[str, Any]] = []
        try:
            with open(events_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        item = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        events.append(item)
        except OSError as error:
            raise SessionStoreError(f"读取会话事件失败：{error}") from error
        return events

    def _append_event_unlocked(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any] | None = None,
        *,
        turn_id: int | None = None,
    ) -> None:
        """
        无锁版事件追加，调用方必须已持有 self._lock。
        """

        session_dir = self._require_session_dir(session_id)
        seq = self._next_event_seq(session_id, session_dir)
        event: dict[str, Any] = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "seq": seq,
            "ts": utc_now_text(),
            "type": str(event_type),
            "session_id": session_id,
        }
        if turn_id is not None:
            event["turn_id"] = int(turn_id)
        event["data"] = data or {}

        events_path = session_dir / EVENTS_FILENAME
        line = json.dumps(event, ensure_ascii=False, default=str)
        try:
            # 逐行 append + flush + fsync。事件日志是崩溃排查的最后依据，这里宁可慢一点
            # 也要保证掉电前已写入的行不丢；本地单用户的事件频率很低，性能完全够用。
            with open(events_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            raise SessionStoreError(f"写入会话事件失败：{error}") from error

    def _next_event_seq(self, session_id: str, session_dir: Path) -> int:
        """
        获取下一个事件 seq。首次访问时从现有文件行数初始化。
        """

        if session_id not in self._event_seq_cache:
            events_path = session_dir / EVENTS_FILENAME
            count = 0
            if events_path.is_file():
                try:
                    with open(events_path, "r", encoding="utf-8") as handle:
                        count = sum(1 for line in handle if line.strip())
                except OSError:
                    count = 0
            self._event_seq_cache[session_id] = count
        self._event_seq_cache[session_id] += 1
        return self._event_seq_cache[session_id]

    def _session_dir(self, session_id: str) -> Path:
        """
        校验 session_id 格式后返回会话目录路径。

        Raises:
            SessionStoreError: session_id 非法时抛出，避免拼接出可穿越路径。
        """

        if not SESSION_ID_PATTERN.match(str(session_id or "")):
            raise SessionStoreError(f"非法 session_id：{session_id!r}")
        return self.base_dir / session_id

    def _require_session_dir(self, session_id: str) -> Path:
        """
        返回已存在的会话目录，不存在时抛出统一错误。
        """

        session_dir = self._session_dir(session_id)
        if not session_dir.is_dir():
            raise SessionStoreError(f"会话 {session_id} 不存在。")
        return session_dir

    def _generate_session_id(self) -> str:
        """
        生成新的 session_id：`sess_YYYYMMDD_HHMMSS_<4位十六进制>`。
        """

        timestamp = datetime.now()
        return f"sess_{timestamp:%Y%m%d}_{timestamp:%H%M%S}_{secrets.token_hex(2)}"


def utc_now_text() -> str:
    """
    返回 UTC ISO-8601 时间字符串。
    """

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """
    原子写入 JSON 文件：先写同目录 tmp 文件并 fsync，再 rename 覆盖目标。

    Args:
        path: 目标文件路径。
        payload: 可 JSON 序列化的字典。

    Raises:
        SessionStoreError: 序列化或写盘失败时抛出。
    """

    tmp_path = path.with_name(path.name + ".tmp")
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        # os.replace 在 Windows 上也支持原子覆盖已存在的目标文件。
        os.replace(tmp_path, path)
    except (OSError, TypeError, ValueError) as error:
        raise SessionStoreError(f"写入 {path.name} 失败：{error}") from error


def read_json_file(path: Path) -> dict[str, Any]:
    """
    读取 JSON 文件并要求顶层是对象。

    Raises:
        SessionStoreError: 文件不可读、JSON 损坏或顶层不是对象时抛出。
    """

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SessionStoreError(f"读取 {path.name} 失败：{error}") from error
    if not isinstance(data, dict):
        raise SessionStoreError(f"{path.name} 内容不是 JSON 对象。")
    return data


def validate_schema_version(data: dict[str, Any], expected: str, path: Path) -> None:
    """
    校验 schema_version 的大版本兼容性。

    只比较 `.v` 前的名字部分和主版本号；缺失可选字段由调用方用默认值兜底，
    未知大版本直接报错，避免按错误结构静默恢复历史会话。
    """

    version = str(data.get("schema_version") or "")
    if version != expected:
        raise SessionStoreError(f"{path.name} schema 版本不兼容：{version or '缺失'}（期望 {expected}）。")


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "META_SCHEMA_VERSION",
    "SNAPSHOT_SCHEMA_VERSION",
    "SessionStore",
    "SessionStoreError",
]
