"""
本地会话持久化存储。

该包只负责会话目录、meta/snapshot/events 文件的读写和原子落盘；快照内容的业务 schema
由调用方（当前是 Web 层）组装，存储层不理解法律业务字段。
"""

from agent_system.storage.session_store import SessionStore, SessionStoreError

__all__ = ["SessionStore", "SessionStoreError"]
