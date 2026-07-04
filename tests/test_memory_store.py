"""
跨会话案件记忆存储的单元测试。

覆盖三块行为：
1. 从会话快照确定性构建记忆条目（不发起任何 LLM 调用）。
2. MemoryStore 的文件读写：原子 upsert、损坏文件跳过、删除幂等和 session_id 校验。
3. 关键词记忆检索：命中长度加权打分、排除当前会话、top_k 截断。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_system.memory import (
    LegalCaseMemory,
    MemoryStore,
    MemoryStoreError,
    build_case_memory_from_snapshot,
    recalled_memories_to_prompt_payload,
)


def build_snapshot(**case_state_overrides) -> dict[str, object]:
    """
    构造与 LegalConsultationSession.export_snapshot() 结构一致的最小快照。
    """

    case_state: dict[str, object] = {
        "summary": "用户与公司存在未签书面劳动合同的争议。",
        "parties": ["劳动者", "公司"],
        "confirmed_facts": ["公司未签书面劳动合同", "工作满两年"],
        "user_goals": ["要求二倍工资"],
        "legal_concepts": ["可能涉及二倍工资", "涉嫌违法解除劳动合同"],
        "version": 2,
    }
    case_state.update(case_state_overrides)
    return {
        "messages": [
            {"role": "system", "content": "系统提示"},
            {"role": "user", "content": "公司没签劳动合同怎么办？"},
            {"role": "assistant", "content": "阶段性答复。"},
        ],
        "case_state": case_state,
    }


class BuildCaseMemoryTests(unittest.TestCase):
    """
    测试从会话快照构建记忆条目的确定性提取逻辑。
    """

    def test_build_memory_extracts_fields_and_strips_concept_prefix(self) -> None:
        """
        记忆应提取摘要、事实、诉求和法律概念；检索关键词需剥掉“可能涉及”类前缀。
        """

        memory = build_case_memory_from_snapshot(
            "sess_20260101_090000_aaaa",
            build_snapshot(),
            title="公司没签劳动合同怎么办？",
            turn_count=1,
        )

        self.assertIsNotNone(memory)
        self.assertEqual("sess_20260101_090000_aaaa", memory.session_id)
        self.assertEqual("公司没签劳动合同怎么办？", memory.title)
        self.assertIn("未签书面劳动合同", memory.summary)
        self.assertIn("公司未签书面劳动合同", memory.key_facts)
        self.assertIn("要求二倍工资", memory.user_goals)
        # 概念进入记忆时保留原文，关键词里则必须是剥掉推测性前缀后的短词。
        self.assertIn("二倍工资", memory.keywords)
        self.assertIn("违法解除劳动合同", memory.keywords)
        self.assertNotIn("可能涉及二倍工资", memory.keywords)
        self.assertIn("劳动者", memory.keywords)
        self.assertEqual(1, memory.turn_count)

    def test_build_memory_returns_none_for_empty_case_state(self) -> None:
        """
        案件状态没有可沉淀内容（无摘要/事实/概念）时不应生成记忆条目。
        """

        snapshot = build_snapshot(summary="", confirmed_facts=[], legal_concepts=[])
        memory = build_case_memory_from_snapshot("sess_20260101_090000_aaaa", snapshot, title="t", turn_count=1)
        self.assertIsNone(memory)

    def test_build_memory_returns_none_without_case_state(self) -> None:
        """
        快照缺少 case_state 字段（如旧版 fake 会话）时返回 None，不抛异常。
        """

        memory = build_case_memory_from_snapshot(
            "sess_20260101_090000_aaaa",
            {"messages": []},
            title="t",
            turn_count=1,
        )
        self.assertIsNone(memory)

    def test_build_memory_uses_summary_as_title_fallback(self) -> None:
        """
        没有显式标题时用案情摘要截断兜底，保证记忆在展示时可读。
        """

        memory = build_case_memory_from_snapshot("sess_20260101_090000_aaaa", build_snapshot(), title="", turn_count=1)
        self.assertIsNotNone(memory)
        self.assertTrue(memory.title)
        self.assertIn("未签", memory.title)


class MemoryStoreTests(unittest.TestCase):
    """
    测试记忆文件存储的读写、upsert、删除和损坏兜底。
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base_dir = Path(self._tmp.name) / "memory"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def build_memory(self, session_id: str, **overrides) -> LegalCaseMemory:
        """
        构造一条可直接保存的记忆。
        """

        kwargs: dict[str, object] = {
            "session_id": session_id,
            "title": "劳动合同咨询",
            "summary": "未签书面劳动合同争议。",
            "key_facts": ["公司未签书面劳动合同"],
            "user_goals": ["要求二倍工资"],
            "legal_concepts": ["可能涉及二倍工资"],
            "keywords": ["二倍工资", "劳动合同"],
            "turn_count": 1,
        }
        kwargs.update(overrides)
        return LegalCaseMemory(**kwargs)

    def test_save_and_load_roundtrip(self) -> None:
        """
        保存后按 session_id 读取应还原全部字段，并自动补上时间戳。
        """

        store = MemoryStore(self.base_dir)
        saved = store.save(self.build_memory("sess_20260101_090000_aaaa"))

        loaded = store.load("sess_20260101_090000_aaaa")
        self.assertIsNotNone(loaded)
        self.assertEqual(saved.title, loaded.title)
        self.assertEqual(["二倍工资", "劳动合同"], loaded.keywords)
        self.assertTrue(loaded.created_at)
        self.assertTrue(loaded.updated_at)

    def test_save_upserts_and_preserves_created_at(self) -> None:
        """
        同一会话重复保存应覆盖旧内容：created_at 保留首次值，updated_at 使用最新时间。
        """

        times = iter(["2026-01-01T09:00:00+00:00", "2026-01-02T09:00:00+00:00"])
        store = MemoryStore(self.base_dir, now_provider=lambda: next(times))

        store.save(self.build_memory("sess_20260101_090000_aaaa", turn_count=1))
        store.save(self.build_memory("sess_20260101_090000_aaaa", turn_count=2, title="更新后的标题"))

        loaded = store.load("sess_20260101_090000_aaaa")
        self.assertEqual("更新后的标题", loaded.title)
        self.assertEqual(2, loaded.turn_count)
        self.assertEqual("2026-01-01T09:00:00+00:00", loaded.created_at)
        self.assertEqual("2026-01-02T09:00:00+00:00", loaded.updated_at)
        # upsert 语义：一个会话只有一个记忆文件。
        self.assertEqual(1, len(list(self.base_dir.glob("sess_*.json"))))

    def test_load_missing_returns_none(self) -> None:
        """
        不存在的会话记忆返回 None 而不是抛错，调用方无需先探测文件。
        """

        store = MemoryStore(self.base_dir)
        self.assertIsNone(store.load("sess_20260101_090000_aaaa"))

    def test_load_all_skips_corrupt_file(self) -> None:
        """
        单个记忆文件损坏时 load_all 应跳过它，不能让整个记忆库不可用。
        """

        store = MemoryStore(self.base_dir)
        store.save(self.build_memory("sess_20260101_090000_aaaa"))
        (self.base_dir / "sess_20260102_090000_bbbb.json").write_text("{broken", encoding="utf-8")

        memories = store.load_all()
        self.assertEqual(1, len(memories))
        self.assertEqual("sess_20260101_090000_aaaa", memories[0].session_id)

    def test_delete_removes_file_and_is_idempotent(self) -> None:
        """
        删除应移除记忆文件；重复删除不存在的记忆不应报错。
        """

        store = MemoryStore(self.base_dir)
        store.save(self.build_memory("sess_20260101_090000_aaaa"))

        store.delete("sess_20260101_090000_aaaa")
        self.assertIsNone(store.load("sess_20260101_090000_aaaa"))
        store.delete("sess_20260101_090000_aaaa")

    def test_invalid_session_id_rejected(self) -> None:
        """
        非法 session_id 必须在拼接路径前被拒绝，防止路径穿越。
        """

        store = MemoryStore(self.base_dir)
        with self.assertRaises(MemoryStoreError):
            store.save(self.build_memory("../evil"))
        with self.assertRaises(MemoryStoreError):
            store.load("../evil")


class MemorySearchTests(unittest.TestCase):
    """
    测试跨会话记忆的关键词检索打分。
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmp.name) / "memory")
        self.store.save(
            LegalCaseMemory(
                session_id="sess_20260101_090000_aaaa",
                title="劳动合同咨询",
                summary="未签书面劳动合同争议。",
                keywords=["二倍工资", "劳动合同", "劳动者"],
                turn_count=1,
            )
        )
        self.store.save(
            LegalCaseMemory(
                session_id="sess_20260102_090000_bbbb",
                title="民间借贷咨询",
                summary="借条与利息争议。",
                keywords=["民间借贷", "借条", "利息"],
                turn_count=1,
            )
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_search_scores_by_keyword_hits(self) -> None:
        """
        只应召回关键词命中的记忆，并返回命中的关键词列表。
        """

        recalls = self.store.search("公司一直没签劳动合同，能要二倍工资吗")

        self.assertEqual(1, len(recalls))
        self.assertEqual("sess_20260101_090000_aaaa", recalls[0].memory.session_id)
        self.assertIn("劳动合同", recalls[0].matched_keywords)
        self.assertIn("二倍工资", recalls[0].matched_keywords)
        self.assertGreater(recalls[0].score, 0)

    def test_search_orders_by_score(self) -> None:
        """
        命中更多、更长关键词的记忆应排在前面。
        """

        recalls = self.store.search("借条丢了，公司还没签劳动合同，二倍工资和利息都想主张")

        self.assertEqual(2, len(recalls))
        self.assertEqual("sess_20260101_090000_aaaa", recalls[0].memory.session_id)
        self.assertEqual("sess_20260102_090000_bbbb", recalls[1].memory.session_id)

    def test_search_excludes_current_session(self) -> None:
        """
        续聊已有会话时必须排除该会话自己的记忆：它的知识已在 case_state 里，重复注入只会浪费上下文。
        """

        recalls = self.store.search(
            "劳动合同二倍工资",
            exclude_session_id="sess_20260101_090000_aaaa",
        )
        self.assertEqual([], [item.memory.session_id for item in recalls if item.memory.session_id == "sess_20260101_090000_aaaa"])

    def test_search_returns_empty_without_hits_or_query(self) -> None:
        """
        无命中或空输入时返回空列表，不做兜底放水。
        """

        self.assertEqual([], self.store.search("邻居装修噪音扰民怎么投诉"))
        self.assertEqual([], self.store.search("   "))

    def test_search_respects_top_k(self) -> None:
        """
        top_k 限制返回条数，避免把过多历史记忆塞进 prompt。
        """

        self.store.save(
            LegalCaseMemory(
                session_id="sess_20260103_090000_cccc",
                title="工伤赔偿咨询",
                summary="工伤认定与赔偿。",
                keywords=["劳动合同", "工伤"],
                turn_count=1,
            )
        )
        recalls = self.store.search("劳动合同相关问题", top_k=1)
        self.assertEqual(1, len(recalls))

    def test_recalled_memories_to_prompt_payload_whitelists_fields(self) -> None:
        """
        注入 prompt 的负载只包含白名单展示字段，不透出 session_id 等内部标识。
        """

        recalls = self.store.search("劳动合同二倍工资")
        payload = recalled_memories_to_prompt_payload(recalls)

        self.assertEqual(1, len(payload))
        item = payload[0]
        self.assertEqual({"title", "summary", "key_facts", "user_goals", "legal_concepts", "updated_at"}, set(item.keys()))
        self.assertEqual("劳动合同咨询", item["title"])


if __name__ == "__main__":
    unittest.main()
