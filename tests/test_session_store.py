"""
会话持久化存储层单元测试。

只测试文件读写、原子落盘、列表和删除；不涉及任何 LLM 或业务链路。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_system.storage import SessionStore, SessionStoreError


class SessionStoreTests(unittest.TestCase):
    """
    SessionStore 的文件行为测试。
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base_dir = Path(self._tmp.name) / "sessions"
        self.store = SessionStore(self.base_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_create_session_writes_meta_and_created_event(self) -> None:
        """
        创建会话应生成合法 ID、meta.json 和 session_created 事件。
        """

        session_id = self.store.create_session(title="测试会话")

        self.assertRegex(session_id, r"^sess_\d{8}_\d{6}_[0-9a-f]{4}$")
        meta = self.store.load_meta(session_id)
        self.assertEqual("测试会话", meta["title"])
        self.assertEqual(0, meta["turn_count"])
        self.assertEqual("legal_session_meta.v1", meta["schema_version"])

        events_path = self.base_dir / session_id / "events.jsonl"
        lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(1, len(lines))
        self.assertEqual("session_created", lines[0]["type"])
        self.assertEqual(1, lines[0]["seq"])

    def test_save_and_load_snapshot_roundtrip(self) -> None:
        """
        快照保存后应能原样读回，并同步更新 meta 的标题和轮次。
        """

        session_id = self.store.create_session()
        snapshot = {
            "messages": [
                {"role": "system", "content": "系统提示"},
                {"role": "user", "content": "公司不签劳动合同怎么办"},
                {"role": "assistant", "content": "可以主张二倍工资。"},
            ],
            "case_state": {"summary": "未签劳动合同", "version": 1},
            "materials": {"laws": [], "web": [], "warnings": []},
            "pending_supplement": None,
        }

        self.store.save_snapshot(session_id, snapshot, title="公司不签劳动合同怎么办", turn_count=1)

        loaded = self.store.load_snapshot(session_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(snapshot["messages"], loaded["messages"])
        self.assertEqual(snapshot["case_state"], loaded["case_state"])
        self.assertEqual("legal_session_snapshot.v1", loaded["schema_version"])

        meta = self.store.load_meta(session_id)
        self.assertEqual("公司不签劳动合同怎么办", meta["title"])
        self.assertEqual(1, meta["turn_count"])

    def test_load_snapshot_returns_none_before_first_save(self) -> None:
        """
        会话存在但从未保存快照时应返回 None，而不是抛错。
        """

        session_id = self.store.create_session()
        self.assertIsNone(self.store.load_snapshot(session_id))

    def test_list_sessions_sorted_by_updated_at_desc(self) -> None:
        """
        列表应按 updated_at 倒序，最近更新的会话排最前。
        """

        first = self.store.create_session()
        second = self.store.create_session()
        # 手工改写 updated_at，避免测试依赖真实时间先后。
        self.store.save_snapshot(first, {"messages": []}, turn_count=1)
        self.store.save_snapshot(second, {"messages": []}, turn_count=2)
        meta_path = self.base_dir / first / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["updated_at"] = "2099-01-01T00:00:00+00:00"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        sessions = self.store.list_sessions()

        self.assertEqual([first, second], [item["session_id"] for item in sessions])

    def test_list_sessions_skips_corrupted_meta(self) -> None:
        """
        单个会话 meta 损坏时应跳过该会话，不影响其他会话列出。
        """

        good = self.store.create_session()
        bad = self.store.create_session()
        (self.base_dir / bad / "meta.json").write_text("{不是 JSON", encoding="utf-8")

        sessions = self.store.list_sessions()

        self.assertEqual([good], [item["session_id"] for item in sessions])

    def test_delete_session_removes_directory(self) -> None:
        """
        删除会话后目录应消失，再次访问应报会话不存在。
        """

        session_id = self.store.create_session()
        self.store.delete_session(session_id)

        self.assertFalse((self.base_dir / session_id).exists())
        self.assertFalse(self.store.session_exists(session_id))
        with self.assertRaises(SessionStoreError):
            self.store.load_meta(session_id)

    def test_invalid_session_id_rejected(self) -> None:
        """
        非法 session_id（含路径穿越）应直接拒绝，不触碰文件系统。
        """

        for bad_id in ["../escape", "sess_2026..", "", "sess_20260703_120000_zzzz/../x"]:
            with self.assertRaises(SessionStoreError):
                self.store.load_meta(bad_id)
        self.assertFalse(self.store.session_exists("../escape"))

    def test_corrupted_snapshot_raises_clear_error(self) -> None:
        """
        快照损坏时应抛出带文件名的清晰错误。
        """

        session_id = self.store.create_session()
        (self.base_dir / session_id / "snapshot.json").write_text("{半个 JSON", encoding="utf-8")

        with self.assertRaises(SessionStoreError) as context:
            self.store.load_snapshot(session_id)
        self.assertIn("snapshot.json", str(context.exception))

    def test_snapshot_schema_version_mismatch_raises(self) -> None:
        """
        未知 schema 版本应报错，避免按错误结构恢复历史。
        """

        session_id = self.store.create_session()
        payload = {"schema_version": "legal_session_snapshot.v999", "messages": []}
        (self.base_dir / session_id / "snapshot.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

        with self.assertRaises(SessionStoreError):
            self.store.load_snapshot(session_id)

    def test_append_event_increments_seq_across_instances(self) -> None:
        """
        事件 seq 应跨 store 实例连续：新实例从现有文件行数继续递增。
        """

        session_id = self.store.create_session()
        self.store.append_event(session_id, "turn_committed", {"turn_count": 1}, turn_id=1)

        reopened = SessionStore(self.base_dir)
        reopened.append_event(session_id, "turn_committed", {"turn_count": 2}, turn_id=2)

        events_path = self.base_dir / session_id / "events.jsonl"
        lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual([1, 2, 3], [item["seq"] for item in lines])
        self.assertEqual(["session_created", "turn_committed", "turn_committed"], [item["type"] for item in lines])

    def test_read_events_returns_events_in_order(self) -> None:
        """
        read_events 应按写入顺序返回全部事件，负载原样读回。
        """

        session_id = self.store.create_session()
        self.store.append_event(
            session_id,
            "turn_committed",
            {"turn_count": 1, "metrics": {"total_duration_ms": 1500}},
            turn_id=1,
        )
        self.store.append_event(session_id, "turn_failed", {"error": "boom"})

        events = self.store.read_events(session_id)

        self.assertEqual(["session_created", "turn_committed", "turn_failed"], [item["type"] for item in events])
        self.assertEqual(1500, events[1]["data"]["metrics"]["total_duration_ms"])

    def test_read_events_skips_corrupted_lines(self) -> None:
        """
        events.jsonl 中间混入坏行时应跳过该行，其余事件正常返回。
        """

        session_id = self.store.create_session()
        self.store.append_event(session_id, "turn_committed", {"turn_count": 1}, turn_id=1)
        events_path = self.base_dir / session_id / "events.jsonl"
        with open(events_path, "a", encoding="utf-8") as handle:
            handle.write("{坏行不是 JSON\n")
        self.store.append_event(session_id, "turn_failed", {"error": "boom"})

        events = self.store.read_events(session_id)

        self.assertEqual(["session_created", "turn_committed", "turn_failed"], [item["type"] for item in events])

    def test_read_events_missing_session_raises(self) -> None:
        """
        读取不存在会话的事件应抛出统一存储错误。
        """

        with self.assertRaises(SessionStoreError):
            self.store.read_events("sess_20260101_090000_aaaa")

    def test_atomic_write_leaves_no_tmp_file(self) -> None:
        """
        快照写入成功后不应残留 tmp 文件。
        """

        session_id = self.store.create_session()
        self.store.save_snapshot(session_id, {"messages": []})

        leftovers = list((self.base_dir / session_id).glob("*.tmp"))
        self.assertEqual([], leftovers)


if __name__ == "__main__":
    unittest.main()
