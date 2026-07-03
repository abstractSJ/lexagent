# 法律咨询 Agent 前端方案

## 目标

为当前法律咨询 Agent 增加一个本地 Web 前端，替代纯命令行交互。

第一版目标不是做复杂产品，而是做一个可用的 Agent 调试/咨询控制台：

- 用户可以在浏览器里输入案情。
- 页面能实时看到 Agent 当前执行到哪一步。
- 页面能显示 RAG 检索 query、错误、最终回答。
- 后端直接复用现有 `LegalConsultationSession`。
- 不引入复杂前端工程，保持小项目简单。

## 推荐技术选型

第一版推荐：

```text
FastAPI + Uvicorn + 原生 HTML/CSS/JavaScript
```

暂时不推荐 React/Vue。

原因：

- 当前重点是验证 Agent 链路，不是做复杂 UI 工程。
- Python 后端可以直接调用现有 Agent 代码。
- 原生 JS 足够实现聊天窗口和事件流。
- 项目结构简单，后续要升级 React/Vue 也容易。

## 目录结构

新增目录：

```text
web_app/
  server.py              # FastAPI 服务入口
  static/
    index.html           # 单页前端
    app.js               # 调接口、读事件流、更新页面
    style.css            # 简单样式
```

不建议放进 `agent_system/`。

原因：

- `agent_system/` 是核心 Agent 库。
- `web_app/` 是一个应用入口，类似 `main.py` 和 `legal_agent_demo.py`。
- 这样分层更清楚。

## 总体架构

```text
浏览器页面
  ↓ HTTP / NDJSON stream
FastAPI server
  ↓
LegalConsultationSession
  ↓
状态更新 / 案情拆解 + 多 query RAG / 风险识别 / 下一步动作 / 最终回答
```

核心复用点：

```python
session.ask_with_events(user_input, on_event=callback)
```

CLI 里现在是：

```python
on_event=print_event
```

Web 里改成：

```python
on_event=lambda event: queue.put(event)
```

然后把事件通过 HTTP 流式返回给前端。

## V1 功能范围

### 必做

1. 单页聊天界面。
2. 输入框 + 发送按钮。
3. 对话区展示用户输入和助手回答。
4. 执行进度区实时显示 Agent 事件。
5. 错误信息显示。
6. 服务启动时预热本地 RAG。
7. 一个全局 `LegalConsultationSession`，先支持单用户本地使用。

### 暂不做

1. 登录注册。
2. 多用户。
3. 数据库。
4. 长期会话持久化。
5. React/Vue 工程。
6. WebSocket。
7. 复杂权限。
8. 生产部署。

## 后端接口设计

### 1. 首页

```http
GET /
```

返回：

```text
web_app/static/index.html
```

### 2. 静态资源

```http
GET /static/app.js
GET /static/style.css
```

由 FastAPI StaticFiles 提供。

### 3. 健康检查

```http
GET /api/health
```

返回：

```json
{
  "ok": true,
  "status": "ready"
}
```

### 4. RAG 预热

```http
POST /api/preload
```

返回：

```json
{
  "ok": true,
  "message": "本地法条 RAG 预热完成"
}
```

说明：

- 服务启动时可以自动预热。
- 这个接口用于前端手动重试或调试。

### 5. 发送消息并流式返回事件

```http
POST /api/chat
Content-Type: application/json
```

请求：

```json
{
  "message": "我在公司干了两年，没有签劳动合同，现在被辞退了"
}
```

响应类型：

```http
Content-Type: application/x-ndjson
```

一行一个 JSON 事件：

```json
{"type":"legal_step","data":{"name":"案件状态更新","status":"start"}}
{"type":"case_state_updated","data":{"version":1,"summary":"..."}}
{"type":"legal_step","data":{"name":"案情拆解 + 多 query RAG","status":"start"}}
{"type":"legal_rag_query_started","data":{"retrieval_type":"semantic","issue":"未签书面劳动合同的责任","query":"未签劳动合同 二倍工资"}}
{"type":"legal_case_rag_done","data":{"issue_count":2,"evidence_count":10}}
{"type":"legal_step","data":{"name":"最终回答生成","status":"start"}}
{"type":"message_done","data":{"text":"最终回答..."}}
```

如果失败：

```json
{"type":"error","data":{"error":"LLM Responses 非流式调用失败：Request timed out."}}
```

最后建议额外发一个结束事件：

```json
{"type":"done","data":{}}
```

这样前端知道可以恢复发送按钮。

## 后端实现要点

### server.py 基本结构

```python
"""
法律咨询 Agent Web 服务入口。

本服务提供一个本地 Web 控制台，用于浏览器交互、实时查看 Agent 执行事件和最终回答。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_system.agent.events import AgentEvent
from agent_system.legal_consultation import create_legal_consultation_session


class ChatRequest(BaseModel):
    """
    前端发送的聊天请求。
    """

    message: str


app = FastAPI(title="Legal Agent Web UI")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

session = create_legal_consultation_session()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def preload_resources() -> None:
    """
    服务启动时预热本地 RAG。

    这样用户第一次发送消息时，不会才开始加载 BGE-M3。
    """

    session.preload_resources()


@app.get("/")
def index() -> FileResponse:
    """
    返回前端首页。
    """

    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    """
    健康检查。
    """

    return {"ok": True, "status": "ready"}


@app.post("/api/preload")
def preload() -> dict[str, Any]:
    """
    手动预热本地 RAG。
    """

    session.preload_resources()
    return {"ok": True, "message": "本地法条 RAG 预热完成"}


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    """
    接收用户消息，并以 NDJSON 流式返回 Agent 执行事件。
    """

    async def event_stream():
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        def on_event(event: AgentEvent) -> None:
            queue.put_nowait({"type": event.type, "data": event.data})

        async def run_agent() -> None:
            try:
                answer, _ = await asyncio.to_thread(
                    session.ask_with_events,
                    request.message,
                    on_event=on_event,
                )
                await queue.put({"type": "assistant_message", "data": {"text": answer}})
                await queue.put({"type": "done", "data": {}})
            except Exception as error:
                await queue.put({"type": "error", "data": {"error": str(error)}})
                await queue.put({"type": "done", "data": {}})
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_agent())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield json.dumps(item, ensure_ascii=False) + "\n"
        finally:
            await task

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
```

注意：上面是实现草图。实际写代码时要确认 `asyncio.Queue` 和线程回调之间是否需要 `loop.call_soon_threadsafe`。更稳的写法是：

```python
loop = asyncio.get_running_loop()

def on_event(event: AgentEvent) -> None:
    loop.call_soon_threadsafe(
        queue.put_nowait,
        {"type": event.type, "data": event.data},
    )
```

这是推荐正式实现方式。

## 前端页面设计

### 布局

```text
┌──────────────────────────────────────────┐
│ 顶部：法律咨询 Agent                      │
├──────────────────────────────────────────┤
│ 左侧：对话区                              │
│  用户消息                                 │
│  助手回答                                 │
├──────────────────────────────────────────┤
│ 右侧：执行进度 / 事件流                    │
│  [步骤] 案件状态更新                       │
│  [检索中] query...                        │
│  [风险识别] ...                           │
├──────────────────────────────────────────┤
│ 底部：输入框 + 发送按钮                    │
└──────────────────────────────────────────┘
```

建议页面结构：

```html
<body>
  <main class="app-shell">
    <section class="chat-panel">
      <header>法律咨询 Agent</header>
      <div id="messages" class="messages"></div>
      <form id="chat-form" class="chat-form">
        <textarea id="message-input"></textarea>
        <button id="send-button" type="submit">发送</button>
      </form>
    </section>

    <aside class="event-panel">
      <header>执行进度</header>
      <div id="events" class="events"></div>
    </aside>
  </main>
</body>
```

## 前端 JS 逻辑

核心逻辑：

1. 用户提交消息。
2. 把用户消息追加到对话区。
3. 禁用发送按钮。
4. `fetch('/api/chat')`。
5. 读取 `response.body.getReader()`。
6. 按行解析 NDJSON。
7. 根据事件类型更新事件区。
8. 收到最终回答后追加助手消息。
9. 收到 `done` 后恢复按钮。

### app.js 草图

```javascript
const form = document.querySelector('#chat-form');
const input = document.querySelector('#message-input');
const button = document.querySelector('#send-button');
const messages = document.querySelector('#messages');
const events = document.querySelector('#events');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  appendMessage('user', text);
  input.value = '';
  events.innerHTML = '';
  button.disabled = true;

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text}),
    });

    if (!response.ok || !response.body) {
      throw new Error(`请求失败：${response.status}`);
    }

    await readNdjsonStream(response.body);
  } catch (error) {
    appendEvent('error', {error: String(error)});
  } finally {
    button.disabled = false;
    input.focus();
  }
});

async function readNdjsonStream(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const {value, done} = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, {stream: true});
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      handleAgentEvent(event);
    }
  }
}

function handleAgentEvent(event) {
  if (event.type === 'assistant_message') {
    appendMessage('assistant', event.data.text || '');
    return;
  }

  if (event.type === 'done') {
    appendEvent('done', {message: '本轮处理完成'});
    return;
  }

  appendEvent(event.type, event.data || {});
}

function appendMessage(role, text) {
  const item = document.createElement('div');
  item.className = `message ${role}`;
  item.textContent = text;
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
}

function appendEvent(type, data) {
  const item = document.createElement('div');
  item.className = `event ${type}`;
  item.textContent = formatEvent(type, data);
  events.appendChild(item);
  events.scrollTop = events.scrollHeight;
}

function formatEvent(type, data) {
  if (type === 'legal_step') {
    return `步骤：${data.name || ''}`;
  }
  if (type === 'legal_rag_query_started') {
    return `检索中：${data.retrieval_type || ''} / ${data.query || ''}`;
  }
  if (type === 'case_state_updated') {
    return `案件状态：${data.summary || ''}`;
  }
  if (type === 'legal_case_rag_done') {
    return `RAG 完成：事项 ${data.issue_count || 0} 个，证据 ${data.evidence_count || 0} 条`;
  }
  if (type === 'legal_risk_analyzed') {
    return `风险识别完成：${data.risk_count || 0} 项`;
  }
  if (type === 'legal_next_action_decided') {
    return `下一步：${data.action || ''}`;
  }
  if (type === 'error') {
    return `错误：${data.error || ''}`;
  }
  return `${type}: ${JSON.stringify(data)}`;
}
```

## 样式方向

`style.css` 第一版简单即可：

- 左右两栏布局。
- 左侧对话区宽一些。
- 右侧事件区窄一些。
- 用户消息靠右，助手消息靠左。
- 错误事件红色。
- 检索事件蓝色。
- 完成事件绿色。

## 运行方式

安装依赖：

```bash
pip install fastapi uvicorn
```

运行：

```bash
python web_app/server.py
```

或者：

```bash
uvicorn web_app.server:app --reload
```

如果用 `python web_app/server.py`，需要在 `server.py` 底部加：

```python
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app.server:app", host="127.0.0.1", port=8000, reload=True)
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## 环境变量

当前可用：

```bash
AGENT_LLM_API_KEY
AGENT_LLM_BASE_URL
AGENT_LLM_MODEL
AGENT_LLM_TIMEOUT
LEGAL_EMBEDDING_DEVICE
LEGAL_RAG_PRELOAD
```

Web 服务建议：

- 默认启动时预热 RAG。
- 如果本地调试想跳过预热，可以支持：

```bash
LEGAL_RAG_PRELOAD=0 uvicorn web_app.server:app --reload
```

## 错误处理要求

后端：

- 捕获 `session.ask_with_events()` 抛出的异常。
- 向前端发送：

```json
{"type":"error","data":{"error":"错误详情"}}
```

- 再发送：

```json
{"type":"done","data":{}}
```

前端：

- 错误显示到事件区。
- 恢复发送按钮。
- 不清空用户已经输入/发送的消息。

## 会话管理 V1

第一版使用单个全局 session：

```python
session = create_legal_consultation_session()
```

这意味着：

- 适合本地单用户测试。
- 浏览器刷新后，后端 session 仍在。
- 服务重启后，历史清空。

暂时不要实现复杂多会话。

## 会话管理 V2

后续可以升级为：

```text
POST /api/sessions
GET /api/sessions/{session_id}/history
POST /api/sessions/{session_id}/chat
DELETE /api/sessions/{session_id}
```

后端维护：

```python
sessions: dict[str, LegalConsultationSession]
```

再后续接入：

```text
data/sessions/<session_id>/snapshot.json
data/sessions/<session_id>/events.jsonl
```

## 测试建议

新增测试文件：

```text
tests/test_web_app.py
```

第一版可测：

1. `GET /api/health` 返回 ok。
2. `POST /api/chat` 空 message 返回 422 或错误事件。
3. fake session 下 `/api/chat` 能返回 NDJSON。
4. 错误时返回 `error` 和 `done`。

如果不想现在加复杂测试，至少手工验证：

```bash
uvicorn web_app.server:app --reload
```

浏览器输入：

```text
我在公司干了两年，没有签劳动合同，现在被辞退了，公司也没给补偿。
```

应看到：

- 右侧事件区实时滚动。
- 最终回答出现在左侧对话区。
- 如果 LLM timeout，右侧显示错误。

## 实施顺序

1. 新增 `web_app/server.py`。
2. 新增 `web_app/static/index.html`。
3. 新增 `web_app/static/app.js`。
4. 新增 `web_app/static/style.css`。
5. 安装 `fastapi`、`uvicorn`。
6. 启动服务并访问页面。
7. 调通 `/api/chat` NDJSON 事件流。
8. 优化事件展示文案。
9. 视情况新增 `tests/test_web_app.py`。
10. 同步更新 `CLAUDE.md` 的入口说明。

## 最小可接受完成标准

完成后应满足：

- `python web_app/server.py` 能启动服务。
- 浏览器打开 `http://127.0.0.1:8000` 能看到聊天页面。
- 输入案情后，右侧能实时显示步骤。
- RAG query 能显示出来。
- 最终回答能显示出来。
- 发生 timeout 或异常时，页面能显示错误而不是一直转圈。

## 不要做的事

第一版不要做：

- React/Vue。
- 用户系统。
- 数据库存储。
- 多租户。
- WebSocket。
- Docker 部署。
- 法条富文本复杂卡片。
- 复杂状态编辑器。

先把“浏览器可用 + 实时知道 Agent 进度”做好。
