# Agent 历史会话本地永久保存方案

## 1. 目标

为当前项目增加本地持久化能力，让 `AgentSession` 和普通 `ChatSession` 的历史会话在进程退出、程序崩溃、电脑重启后仍可恢复。

本方案优先满足：

1. **可续聊**：恢复后继续把历史上下文传给模型。
2. **可追踪**：保存工具调用、工具结果、模型回复、失败事件、token usage。
3. **可崩溃恢复**：中途失败不污染正式上下文。
4. **可演进**：后续可平滑迁移到 SQLite、Web UI、检索索引或多用户会话系统。
5. **贴合当前代码结构**：不强行把工具调用塞进 `messages`，避免破坏现有 Responses API 转换逻辑。

## 2. 非目标

第一版不做：

1. 多用户权限隔离。
2. 远程同步。
3. 全文搜索索引。
4. SQLite 数据库。
5. 云端会话存储。
6. provider 原始响应的完整重放。

这些可以在 V2/V3 里加。

## 3. 当前代码现状

### 3.1 主会话状态在内存 `messages` 里

当前 `AgentSession` 只维护 Chat-style messages：

- [agent_system/agent/session.py:33-37](agent_system/agent/session.py#L33-L37)
- [agent_system/agent/session.py:74-87](agent_system/agent/session.py#L74-L87)

普通 `ChatSession` 也是同样模式：

- [agent_system/llm/session.py:36-40](agent_system/llm/session.py#L36-L40)
- [agent_system/llm/session.py:96-125](agent_system/llm/session.py#L96-L125)

因此，**能恢复续聊的最小状态就是 `messages`**。

### 3.2 工具调用不在 `messages` 里

Agent 工具调用过程通过 `AgentEvent` 暴露：

- [agent_system/agent/events.py:12-31](agent_system/agent/events.py#L12-L31)
- [agent_system/agent/runner.py:199-231](agent_system/agent/runner.py#L199-L231)

`AgentSession` 最终只把用户输入和最终 assistant 回复写进历史，不保存中间工具 item。

这是当前代码的设计选择。持久化层应该尊重这个边界。

### 3.3 当前 `tool` role 不适合直接写回 history

`OpenAIChatClient` 目前把项目内部 messages 转为 Responses API input。它显式拒绝 `tool` role：

- [agent_system/llm/openai_client.py:350-356](agent_system/llm/openai_client.py#L350-L356)

所以第一版不要设计成：

```json
{"role": "tool", "content": "..."}
```

否则恢复后再请求模型会触发转换错误。

### 3.4 图片当前只是临时请求输入

图片不会永久写入历史，而是在请求前临时追加：

- [agent_system/llm/session.py:88-90](agent_system/llm/session.py#L88-L90)
- [agent_system/llm/openai_client.py:571-598](agent_system/llm/openai_client.py#L571-L598)

因此第一版不要把 base64 图片写进 `snapshot.json`。

## 4. 核心设计决策

### 4.1 采用文件存储，不先上 SQLite

推荐第一版采用：

```text
snapshot.json + events.jsonl + meta.json
```

不用单个大 JSON，也不用 SQLite。

原因：

1. 当前项目是学习型 Python 项目，文件方案更直接。
2. `events.jsonl` append-only，崩溃时更安全。
3. `snapshot.json` 可快速恢复，不需要每次重放全部事件。
4. 未来可把 `events.jsonl` 导入 SQLite，不锁死路线。
5. 人可以直接打开文件观察，会话调试更方便。

### 4.2 主真相使用项目内部 Chat-style messages

主状态不存 OpenAI Responses 原始结构，而存当前项目已经使用的内部格式：

```json
{"role": "user", "content": "你好"}
{"role": "assistant", "content": "你好，有什么可以帮你？"}
```

原因：

1. 当前 `ChatSession` / `AgentSession` 已经使用这个格式。
2. `OpenAIChatClient` 已经负责 provider 格式转换。
3. 后续换模型服务或适配别的 API 时，历史存储不需要大改。

### 4.3 工具调用作为事件保存，不塞进 `messages`

工具调用、工具结果、usage、错误都进入 `events.jsonl`。

正式上下文仍只存：

1. `system`
2. `user`
3. `assistant`

这样恢复续聊最稳。

## 5. 推荐目录结构

默认存储目录：

```text
data/
  sessions/
    <session_id>/
      meta.json
      snapshot.json
      events.jsonl
      turns/
        000001.json
        000002.json
      attachments/
```

第一版必须文件：

```text
meta.json
snapshot.json
events.jsonl
```

可选文件夹：

```text
turns/
attachments/
```

### 5.1 session_id 命名建议

建议格式：

```text
sess_YYYYMMDD_HHMMSS_<short_random>
```

示例：

```text
sess_20260701_101530_a8f3
```

原因：

1. 文件夹自然按时间排序。
2. 短随机后缀避免同秒冲突。
3. 肉眼可读。

## 6. 文件职责

### 6.1 `meta.json`

`meta.json` 保存会话元数据，低频更新。

示例：

```json
{
  "schema_version": "session_meta.v1",
  "session_id": "sess_20260701_101530_a8f3",
  "session_type": "agent",
  "title": "法律咨询会话",
  "created_at": "2026-07-01T10:15:30Z",
  "updated_at": "2026-07-01T10:20:10Z",
  "model": "gpt-5.5",
  "base_url": "https://your-openai-compatible-endpoint.example/v1",
  "tool_names": [
    "get_current_time",
    "calculate",
    "remember_note",
    "read_note",
    "search_legal_articles"
  ],
  "tags": ["legal", "agent"],
  "archived": false
}
```

注意：

1. **不要保存 API Key**。
2. `base_url` 可选；如果担心暴露内部地址，可以不存。
3. `tool_names` 只用于调试和兼容检查，不作为工具真实定义。

### 6.2 `snapshot.json`

`snapshot.json` 保存可恢复的当前状态。

示例：

```json
{
  "schema_version": "session_snapshot.v1",
  "session_id": "sess_20260701_101530_a8f3",
  "last_event_seq": 42,
  "last_turn_id": 7,
  "messages": [
    {"role": "system", "content": "你是一个严谨的法律助手。"},
    {"role": "user", "content": "试用期最长多久？"},
    {"role": "assistant", "content": "一般最长不超过六个月，具体取决于劳动合同期限。"}
  ],
  "tool_state": {
    "notes": {
      "user_name": "张三"
    }
  },
  "stats": {
    "turn_count": 7,
    "message_count": 15,
    "total_input_tokens": 12345,
    "total_output_tokens": 2345,
    "total_tokens": 14690
  },
  "updated_at": "2026-07-01T10:20:10Z"
}
```

`messages` 是恢复续聊的主字段。

`tool_state` 用于保存工具自己的状态。例如当前 demo 工具里的 `remember_note/read_note` 其实是进程内 dict，未来应移动到这里。

### 6.3 `events.jsonl`

`events.jsonl` 是 append-only 事件日志。每行一个 JSON 对象。

示例：

```jsonl
{"seq":1,"ts":"2026-07-01T10:15:30Z","type":"session_created","session_id":"sess_20260701_101530_a8f3","data":{"session_type":"agent"}}
{"seq":2,"ts":"2026-07-01T10:16:01Z","type":"turn_started","session_id":"sess_20260701_101530_a8f3","turn_id":1,"data":{"user_text":"试用期最长多久？"}}
{"seq":3,"ts":"2026-07-01T10:16:02Z","type":"tool_call","session_id":"sess_20260701_101530_a8f3","turn_id":1,"data":{"name":"search_legal_articles","call_id":"call_abc","arguments":{"query":"劳动合同 试用期 最长"},"raw_arguments":"{\"query\":\"劳动合同 试用期 最长\"}"}}
{"seq":4,"ts":"2026-07-01T10:16:03Z","type":"tool_result","session_id":"sess_20260701_101530_a8f3","turn_id":1,"data":{"name":"search_legal_articles","call_id":"call_abc","result":{"ok":true}}}
{"seq":5,"ts":"2026-07-01T10:16:04Z","type":"llm_usage","session_id":"sess_20260701_101530_a8f3","turn_id":1,"data":{"phase":"tool_result","response_id":"resp_123","model":"gpt-5.5","usage":{"input_tokens":800,"output_tokens":120,"total_tokens":920},"latency_ms":1820}}
{"seq":6,"ts":"2026-07-01T10:16:05Z","type":"turn_committed","session_id":"sess_20260701_101530_a8f3","turn_id":1,"data":{"assistant_text":"一般最长不超过六个月，具体取决于劳动合同期限。"}}
```

设计原因：

1. append 写入比频繁重写大 JSON 更安全。
2. 崩溃时能看到最后执行到哪一步。
3. 后续可以重放、统计、导入 SQLite。
4. 工具调用过程不会污染模型上下文。

### 6.4 `turns/000001.json`

可选调试包。每轮一个文件。

示例：

```json
{
  "schema_version": "session_turn.v1",
  "session_id": "sess_20260701_101530_a8f3",
  "turn_id": 1,
  "started_at": "2026-07-01T10:16:01Z",
  "completed_at": "2026-07-01T10:16:05Z",
  "user_text": "试用期最长多久？",
  "assistant_text": "一般最长不超过六个月，具体取决于劳动合同期限。",
  "events": [
    {"type": "tool_call", "data": {"name": "search_legal_articles"}},
    {"type": "tool_result", "data": {"name": "search_legal_articles", "result": {"ok": true}}}
  ],
  "usage_summary": {
    "input_tokens": 800,
    "output_tokens": 120,
    "total_tokens": 920
  }
}
```

第一版可以不做。等调试需要增强时再加。

### 6.5 `attachments/`

用于保存图片、PDF、上传文件等会话附件。

规则：

1. 不把 base64 写进 `snapshot.json`。
2. 附件文件存磁盘。
3. messages 或 events 里只存相对路径、hash、mime、原始文件名。

示例附件引用：

```json
{
  "type": "attachment_ref",
  "attachment_id": "att_0001",
  "path": "attachments/att_0001.png",
  "mime_type": "image/png",
  "sha256": "..."
}
```

## 7. 事件类型设计

第一版建议支持这些事件：

| 事件类型 | 触发时机 | 主要用途 |
|---|---|---|
| `session_created` | 创建会话 | 初始化审计 |
| `session_loaded` | 从磁盘恢复 | 调试恢复行为 |
| `turn_started` | 用户输入进入会话 | 记录输入 |
| `tool_call` | 模型请求调用工具 | 调试工具选择 |
| `tool_result` | 本地工具执行完成 | 调试工具输出 |
| `llm_usage` | 每次模型响应完成 | token 和成本统计 |
| `message_done` | runner 得到最终文本 | 对齐现有 AgentEvent |
| `turn_committed` | user + assistant 正式写入 snapshot | 标记一轮成功完成 |
| `turn_failed` | 模型调用或工具循环失败 | 崩溃恢复、错误排查 |
| `session_cleared` | 清空历史 | 记录破坏性状态变更 |
| `snapshot_written` | 快照写入成功 | 调试持久化层 |

所有事件必须有：

```json
{
  "seq": 1,
  "ts": "2026-07-01T10:16:01Z",
  "type": "turn_started",
  "session_id": "sess_...",
  "turn_id": 1,
  "data": {}
}
```

字段说明：

1. `seq`：单 session 内单调递增。
2. `ts`：UTC ISO-8601 时间。
3. `type`：事件类型。
4. `session_id`：会话 ID。
5. `turn_id`：可选；会话级事件可以没有。
6. `data`：事件负载。

## 8. 写入策略

### 8.1 事件先写，快照后写

一轮请求流程：

```text
append turn_started
run LLM / tools
append tool_call / tool_result / llm_usage
append turn_committed
atomic write snapshot.json
```

失败流程：

```text
append turn_started
run LLM / tools
发生异常
append turn_failed
不写入 snapshot.messages
```

这样和当前代码失败回滚语义一致。

### 8.2 `snapshot.json` 必须原子写

不要直接覆盖 `snapshot.json`。

正确流程：

```text
write snapshot.json.tmp
flush
fsync
rename snapshot.json.tmp -> snapshot.json
```

原因：

1. 程序崩溃时不会留下半个 JSON。
2. Windows 下 rename 替换也比直接覆盖安全。
3. 恢复时只读完整快照。

### 8.3 `events.jsonl` 每行写完要 flush

第一版可以每个事件：

1. append 一行。
2. flush。
3. 必要时 fsync。

如果后续性能压力大，再改批量 flush。

## 9. 恢复策略

### 9.1 正常恢复

启动时：

1. 读取 `meta.json`。
2. 读取 `snapshot.json`。
3. 校验 `schema_version`。
4. 创建 `AgentSession` 或 `ChatSession`。
5. 把 `snapshot.messages` 赋给 `session.messages`。
6. 把 `snapshot.tool_state` 注入工具状态。
7. 从 `snapshot.last_turn_id` 继续分配 turn id。
8. 从 `snapshot.last_event_seq` 继续分配 seq。

### 9.2 崩溃恢复

如果 `events.jsonl` 最后一轮存在 `turn_started`，但不存在 `turn_committed`，说明上一轮没有完成。

处理方式：

1. 不把未完成轮合入 `messages`。
2. 在 UI 或 CLI 中标记：上一轮异常中断。
3. 可以追加一个 `turn_aborted_detected` 事件。
4. 用户后续可以重新提问。

不要自动把半截 user 输入恢复为正式上下文。

原因：当前 `AgentSession` 和 `ChatSession` 本来就是失败回滚设计。

### 9.3 快照损坏恢复

如果 `snapshot.json` 损坏：

1. 尝试读取 `snapshot.json.bak`，如果未来实现备份。
2. 或从 `events.jsonl` 重放到最后一个 `turn_committed`。
3. 如果 JSONL 中某一行损坏，跳过损坏行后面的事件，并提示人工处理。

第一版可以先只支持：快照损坏时报错。

## 10. SessionStore 接口建议

后续实现时建议新增模块：

```text
agent_system/storage/
  __init__.py
  session_store.py
  schemas.py
```

核心类：

```python
class SessionStore:
    """
    本地会话持久化存储。

    负责写入 meta、追加事件、原子写入 snapshot，以及从磁盘恢复会话状态。
    业务层不直接操作 JSON 文件，避免路径、schema、原子写逻辑散落到多个模块。
    """

    def create_session(self, session_type: str, title: str | None = None) -> str:
        """创建会话目录和 meta.json，返回 session_id。"""

    def append_event(self, event_type: str, data: dict, turn_id: int | None = None) -> None:
        """向 events.jsonl 追加一条事件。"""

    def save_snapshot(self, messages: list[dict], tool_state: dict | None = None) -> None:
        """原子写入 snapshot.json。"""

    def load_snapshot(self, session_id: str) -> dict:
        """读取 snapshot.json，用于恢复会话。"""

    def list_sessions(self) -> list[dict]:
        """扫描 data/sessions，返回会话列表摘要。"""
```

注意：这里是设计草案，不是要求一次实现全部方法。

## 11. 集成点

### 11.1 `AgentSession`

建议改造位置：

- [agent_system/agent/session.py:24-38](agent_system/agent/session.py#L24-L38)
- [agent_system/agent/session.py:74-87](agent_system/agent/session.py#L74-L87)
- [agent_system/agent/session.py:99-115](agent_system/agent/session.py#L99-L115)

建议新增可选参数：

```python
store: SessionStore | None = None
session_id: str | None = None
```

行为：

1. `ask_with_events()` 开始时写 `turn_started`。
2. runner 返回后写 runner events。
3. 成功 append assistant 后写 `turn_committed`。
4. 然后写 `snapshot.json`。
5. 异常时写 `turn_failed`，不写 snapshot。
6. `clear()` 成功后写 `session_cleared` 和新 snapshot。

### 11.2 `ChatSession`

建议改造位置：

- [agent_system/llm/session.py:27-41](agent_system/llm/session.py#L27-L41)
- [agent_system/llm/session.py:96-125](agent_system/llm/session.py#L96-L125)
- [agent_system/llm/session.py:136-151](agent_system/llm/session.py#L136-L151)

行为和 `AgentSession` 类似，但没有工具事件。

### 11.3 `AgentRunner`

建议改造位置：

- [agent_system/agent/runner.py:100-141](agent_system/agent/runner.py#L100-L141)
- [agent_system/agent/runner.py:199-231](agent_system/agent/runner.py#L199-L231)

第一版不一定让 `Runner` 直接依赖 `SessionStore`。

更低耦合方式：

1. `Runner` 继续返回 `events`。
2. `AgentSession` 收到 events 后写入 store。

如果后续要存每次 LLM 调用 usage 和 response_id，就需要让 `Runner` 在每次 `create_response()` 后追加 `llm_usage` event。

### 11.4 factory 层

建议改造位置：

- [agent_system/agent/factory.py:15-34](agent_system/agent/factory.py#L15-L34)
- [agent_system/llm/factory.py:27-41](agent_system/llm/factory.py#L27-L41)

建议新增：

```python
def create_agent_session(system_prompt: str | None = None, session_id: str | None = None) -> AgentSession:
    ...
```

行为：

1. `session_id is None`：创建新会话。
2. `session_id exists`：从磁盘恢复。
3. 如果恢复的 session_type 和 factory 不匹配，抛错。

## 12. 工具状态设计

当前 demo 工具里的记忆：

- [agent_system/agent/tools.py:286-337](agent_system/agent/tools.py#L286-L337)

是函数闭包中的 `notes` 字典。这个状态进程结束就消失。

如果要让 `remember_note/read_note` 真正永久保存，建议改为：

1. 增加 `ToolState` 或让 `ToolRegistry` 持有 `state`。
2. `remember_note` 写入 `state["notes"]`。
3. `read_note` 从 `state["notes"]` 读取。
4. `SessionStore.save_snapshot()` 保存 `tool_state`。
5. 恢复时把 `tool_state` 注入工具层。

第一版可以先不改工具状态，但文档和实现要明确：

> 恢复会话历史不等于恢复工具内部记忆。

## 13. usage 设计

当前 `OpenAIChatClient.get_usage()` 可以拿 usage：

- [agent_system/llm/openai_client.py:152-198](agent_system/llm/openai_client.py#L152-L198)
- [agent_system/llm/openai_client.py:520-551](agent_system/llm/openai_client.py#L520-L551)

但 Agent Runner 的 `create_response()` 调用后，目前没有把 usage 写成事件。

后续建议：

1. 每次 `create_response()` 后读取 `response.usage`。
2. 标准化为 `llm_usage` event。
3. 每轮结束汇总到 `snapshot.stats`。

一轮 Agent 可能有多次模型调用，所以 usage 不能只存最终一条。

建议字段：

```json
{
  "phase": "initial",
  "response_id": "resp_123",
  "model": "gpt-5.5",
  "usage": {
    "input_tokens": 800,
    "output_tokens": 120,
    "total_tokens": 920
  },
  "latency_ms": 1820
}
```

`phase` 可选值：

1. `initial`
2. `tool_result`
3. `final`
4. `usage_probe`

## 14. schema 版本与兼容

所有文件都必须包含 `schema_version`。

推荐版本：

```text
session_meta.v1
session_snapshot.v1
session_event.v1
session_turn.v1
```

加载时策略：

1. 支持当前版本。
2. 遇到未知大版本直接报错。
3. 遇到缺失的可选字段使用默认值。
4. 永远保留未知字段，不要读写时删除，方便未来扩展。

## 15. 安全与隐私

### 15.1 不保存敏感配置

不要保存：

1. API Key。
2. access token。
3. 密码。
4. cookie。
5. 私有密钥。

当前配置里有 API Key 字段，持久化时必须跳过。

### 15.2 工具结果可能含敏感信息

`tool_result` 可能包含检索内容、文件内容、用户输入。第一版可以原样保存，但要明确：

1. `data/sessions/` 不应提交到 Git。
2. 后续如做用户删除，需要支持按 session 删除目录。
3. 如果工具会读本地敏感文件，需要增加脱敏或确认机制。

### 15.3 附件路径要限制在 session 目录内

保存附件时，路径必须 canonicalize，避免 `../` 路径穿越。

## 16. 清理策略

第一版可手动清理。

后续可加：

1. `archived` 标记。
2. 最近 N 个会话保留。
3. 超过 N 天自动归档。
4. 单 session 最大体积限制。
5. 大工具结果截断或外置文件保存。

## 17. 未来升级到 SQLite 的条件

出现以下需求时再迁移 SQLite：

1. 会话数量很多，目录扫描慢。
2. 需要按标题、时间、标签、模型、工具名快速筛选。
3. 需要统计所有会话 token 消耗。
4. 需要分页 UI。
5. 需要多进程并发写。
6. 需要事务级一致性。

迁移方式：

1. 保留 `events.jsonl` 作为原始审计日志。
2. 新增 SQLite 作为索引和查询层。
3. 写入事件时同步写 SQLite。
4. 或定期从 JSONL 导入 SQLite。

不要一开始就把 JSONL 删除。

## 18. 实施阶段

### 阶段 1：最小可用持久化

目标：退出后能恢复继续聊天。

任务：

1. 新增 `agent_system/storage/session_store.py`。
2. 实现 `meta.json`、`snapshot.json`、`events.jsonl` 写入。
3. `AgentSession` 成功一轮后保存 snapshot。
4. `ChatSession` 成功一轮后保存 snapshot。
5. factory 支持 `session_id` 恢复。

验收：

1. 创建 agent 会话。
2. 问一轮。
3. 退出程序。
4. 用相同 `session_id` 恢复。
5. 再问“我刚才问了什么”。
6. 模型能基于历史回答。

### 阶段 2：工具事件持久化

目标：能看到每轮工具调用轨迹。

任务：

1. 把 `AgentEvent` 写入 `events.jsonl`。
2. 为每轮分配 `turn_id`。
3. 记录 `tool_call/tool_result/message_done/turn_failed`。
4. 可选生成 `turns/000001.json`。

验收：

1. 触发 `search_legal_articles`。
2. `events.jsonl` 里能看到 query、call_id、结果摘要。
3. 模型失败时能看到 `turn_failed`。

### 阶段 3：usage 和统计

目标：统计 token 和成本基础数据。

任务：

1. `Runner` 每次 `create_response()` 后提取 usage。
2. 写 `llm_usage` event。
3. `snapshot.stats` 维护累计值。
4. 列表页或 CLI 能展示每个 session token 总量。

验收：

1. 多工具调用一轮产生多条 usage。
2. `snapshot.stats.total_tokens` 等于历史 usage 累计。

### 阶段 4：工具状态持久化

目标：`remember_note/read_note` 跨进程可用。

任务：

1. 把 demo notes 从闭包 dict 改成 session tool_state。
2. `snapshot.tool_state` 保存 notes。
3. 恢复会话时注入 notes。

验收：

1. 用户要求记住一条信息。
2. 程序退出。
3. 恢复同一个 session。
4. 用户询问之前保存的信息。
5. 工具能读出。

### 阶段 5：附件持久化

目标：支持图片/PDF 等多模态上下文恢复。

任务：

1. 新增 `attachments/`。
2. 图片请求时复制文件或记录安全引用。
3. messages/events 中保存 attachment ref。
4. 恢复时可重新注入附件。

验收：

1. 发送图片问题。
2. 程序退出恢复。
3. 后续能引用该图片上下文或提示附件仍可用。

## 19. 测试清单

### 19.1 正常路径

- [ ] 新建 chat session 后生成 `meta.json`。
- [ ] 成功一轮后生成 `snapshot.json`。
- [ ] 成功一轮后追加 `turn_committed`。
- [ ] 恢复后 `messages` 数量一致。

### 19.2 Agent 工具路径

- [ ] 工具调用写入 `tool_call`。
- [ ] 工具结果写入 `tool_result`。
- [ ] 最终回答写入 `turn_committed`。
- [ ] `snapshot.messages` 不包含 `tool` role。

### 19.3 失败路径

- [ ] LLM 调用异常时写 `turn_failed`。
- [ ] 失败轮 user 不进入 `snapshot.messages`。
- [ ] 中途异常后恢复不会出现半截历史。

### 19.4 清空历史

- [ ] `clear(keep_system=True)` 后只保留 system。
- [ ] `clear(keep_system=False)` 后清空 messages。
- [ ] 写入 `session_cleared` 事件。

### 19.5 文件可靠性

- [ ] `snapshot.json` 原子写。
- [ ] `events.jsonl` 每行合法 JSON。
- [ ] 缺失 `meta.json` 时有清晰错误。
- [ ] schema version 不匹配时有清晰错误。

## 20. 最终推荐

本项目后续实现 agent 历史会话本地永久保存时，采用：

```text
meta.json       # 会话元数据
snapshot.json   # 快速恢复主状态
events.jsonl    # append-only 过程日志
```

不要第一版上 SQLite。

不要把工具调用塞进 `messages`。

不要把 base64 图片塞进 `snapshot.json`。

主状态保存项目内部 Chat-style messages。

工具调用、usage、错误作为事件保存。

一轮完成后才写 snapshot，失败只写 failed event。

这套方案最贴合当前项目结构，改动小，恢复稳，后续演进空间足够。
