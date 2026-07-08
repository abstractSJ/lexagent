# 项目说明

本文件给 Claude 提供项目级长期上下文。先读这里建立全局认知；改代码前再精准读取目标文件和对应测试，避免全项目扫描。

## 默认回答风格

本项目默认启用 caveman 风格：回答短、直接、少废话；保留必要技术信息。除非用户明确要求详细解释、写文档、写计划，或说 `stop caveman` / `normal mode`，否则保持精简表达。

## 项目定位

学习型 Python Agent 项目，当前能力：

- 默认本地 Web UI 入口：`main.py`；旧普通聊天 CLI：`python main.py --mode chat`。
- OpenAI-compatible Responses API 封装：`agent_system/llm/`。
- 本地 function tool 调用闭环：`agent_system/agent/`。
- 法律咨询 Agent：`legal_agent_demo.py`。
- 法律咨询多轮业务链路：`agent_system/legal_consultation/`。
- 本地 Web UI：`web_app/server.py` + React/Vite/Material UI 构建产物。
- 本地法条 RAG：BGE-M3 + Chroma，代码在 `agent_system/retrieval/`。
- 法条关键词精确匹配兜底：`search_legal_articles_by_keyword`。
- 法律案情 query planner：`agent_system/planning/`。
- 会话持久化（法律 Web 链路已实现）：`agent_system/storage/` + `data/sessions/`。
- 跨会话案件记忆（法律 Web 链路已实现）：`agent_system/memory/` + `data/memory/`，沉淀 case_state 知识并按关键词召回注入新会话。
- 链路可观测性与自修复（法律链路已实现）：子任务失败降级、最终回答自动重试（`legal_selfheal` 事件），轮级阶段耗时与 LLM usage 指标（`legal_turn_metrics` 事件），Web 端 `GET /api/metrics` 聚合。
- Web 历史对话侧边栏：刷新自动恢复最近会话，可切换/删除历史会话并续聊。

重要偏好：这是小项目，保持简单。不要过度分层，不要提前引入复杂 DI、插件系统、数据库、复杂前端工程、多层 planner/executor/state 架构。

## 常用入口

```text
main.py                         默认启动法律咨询本地 Web UI；`--mode chat` 启动旧普通聊天 CLI
legal_agent_demo.py             法律咨询 Agent CLI
web_app/server.py               法律咨询本地 Web UI
scripts/build_legal_chroma.py   构建法条 Chroma 向量库
scripts/test_legal_chroma.py    测试 Chroma 法条库
scripts/plan_legal_queries.py   法律案情 query planner CLI
```

常用命令：

```bash
python main.py
python main.py --mode chat
python legal_agent_demo.py
python -m uvicorn web_app.server:app --reload
npm install
npm run dev
npm run build
python scripts/build_legal_chroma.py
python scripts/test_legal_chroma.py --metadata-only
python scripts/test_legal_chroma.py --query "公司不签劳动合同怎么办" --legal-name "劳动合同法" --top-k 15
python scripts/plan_legal_queries.py --case-text "公司一直没签劳动合同，也没交社保"
python -m unittest tests.test_legal_consultation
python -m unittest discover
```

## 目录职责

```text
agent_system/config.py      项目配置 dataclass 和加载函数
agent_system/llm/           LLM client、消息格式、普通聊天 session
agent_system/agent/         工具定义、工具注册、AgentRunner、AgentSession、legal tools
agent_system/legal_consultation/ 法律咨询多轮状态、子任务和执行链路
agent_system/retrieval/     法条预处理、embedding、Chroma、检索器
agent_system/planning/      法律案情 query planner
agent_system/storage/       会话持久化文件存储（SessionStore）
agent_system/memory/        跨会话案件记忆文件存储（MemoryStore）
frontend/src/               React + Material UI 前端源码
web_app/                    FastAPI 本地 Web UI 和 Vite 构建静态产物
tests/                      unittest 测试
data/                       法条原始数据、Chroma 持久化目录、会话存档（data/sessions/）和案件记忆（data/memory/）
docs/                       项目文档（architecture.html 架构图，自包含单文件，浏览器直接打开）
```

## 配置

文件：`agent_system/config.py`

核心配置：

- `LLMConfig`
- `LLMCallOptions`
- `EmbeddingConfig`
- `ChromaConfig`
- `RetrievalConfig`
- `WebSearchConfig`

环境变量：

- `AGENT_LLM_API_KEY`
- `AGENT_LLM_BASE_URL`
- `AGENT_LLM_MODEL`
- `AGENT_LLM_TIMEOUT`
- `LEGAL_AGENT_LEGACY`
- `LEGAL_EMBEDDING_MODEL`
- `LEGAL_EMBEDDING_DEVICE`
- `LEGAL_EMBEDDING_BATCH_SIZE`
- `LEGAL_TORCH_NUM_THREADS`
- `LEGAL_RAG_PRELOAD`
- `BOCHA_WEB_SEARCH_API_KEY`
- `BOCHA_WEB_SEARCH_URL`
- `BOCHA_WEB_SEARCH_TIMEOUT`
- `BOCHA_WEB_SEARCH_COUNT`
- `BOCHA_WEB_SEARCH_MAX_COUNT`
- `BOCHA_WEB_SEARCH_SUMMARY`
- `BOCHA_WEB_SEARCH_FRESHNESS`

注意：本项目已用 git 管理。真实 API key 不进 git：放环境变量，或放 `agent_system/config_local.py`（已被 `.gitignore` 排除，模板 `agent_system/config_local.example.py`），读取优先级为环境变量 > config_local > 内置默认值。`data/chroma/`、`data/sessions/`、`data/memory/`、`web_app/static/`、`node_modules/` 不入库。BGE-M3 首次运行可能下载模型；embedding 默认使用 CPU，确认 CUDA 环境稳定后可设 `LEGAL_EMBEDDING_DEVICE=cuda`。

## LLM 层

目录：`agent_system/llm/`

核心文件：

- `messages.py`：`system_message()`、`user_message()`、`assistant_message()`。
- `openai_client.py`：`OpenAIChatClient`，封装 Responses API；按完成的调用累计 `usage_totals`（calls/input/output/total tokens），`snapshot_usage_totals()` 供轮级指标做差。
- `session.py`：`ChatSession`，普通聊天历史。
- `factory.py`：`build_llm_client()`、`create_chat_session()`。

普通聊天流程：

```text
main.py --mode chat
  -> create_chat_session()
  -> ChatSession.stream_ask()
  -> OpenAIChatClient.chat()
  -> client.responses.create(..., stream=True)
```

`ChatSession` 成功后追加 assistant message；失败或流式中断时回滚当前 user message。

## Agent 层

目录：`agent_system/agent/`

核心文件：

- `events.py`：`AgentEvent`，事件类型含 `message_done`、`tool_call`、`tool_result`、`error`。
- `tools.py`：`LocalTool`、`ToolRegistry`。
- `runner.py`：`AgentRunner`、`AgentRunOptions`。
- `session.py`：`AgentSession`。
- `legal_tools.py`：法律检索工具包装。
- `web_search_tools.py`：博查 Web Search 工具包装。
- `factory.py`：`create_agent_session()`。

Agent 调用流程：

```text
legal_agent_demo.py
  -> create_agent_session()
    -> build_llm_client()
    -> build_legal_tools() + build_web_search_tools()
    -> ToolRegistry(...)
    -> AgentRunner(...)
    -> AgentSession(...)
  -> AgentSession.ask_with_events()
  -> AgentRunner.run()
  -> tools.to_openai_tools()
  -> llm.create_response(..., tools=...)
  -> model function_call
  -> ToolRegistry.run(name, arguments)
  -> LocalTool.handler(arguments)
  -> function_call_output
  -> llm.create_response(...)
  -> final answer
```

重要规则：

- `AgentSession.messages` 只保存 system/user/assistant。
- 工具过程只走 `AgentEvent`，不要塞进长期 `messages`。
- `AgentRunner` 手动维护 Responses input 链，不依赖 `previous_response_id`。
- 默认工具注册点是 `agent_system/agent/factory.py`。
- 新增 legal tool 优先扩展 `agent_system/agent/legal_tools.py` 的 `build_legal_tools()`。
- 新增公网搜索 tool 优先扩展 `agent_system/agent/web_search_tools.py` 的 `build_web_search_tools()`。
- 旧学习 demo tools 已移除；默认 Agent 注册 legal tools 和 Bocha web_search。

## 法律咨询业务链路

目录：`agent_system/legal_consultation/`

核心文件：

- `models.py`：`LegalCaseState`、`LegalCaseRagResult`、`LegalWebSearchResearchResult`、`LegalRiskFinding`、`LegalAnalysisCatalog`、`LegalNextAction`、`LegalCaseAnalysis`。
- `subtasks.py`：状态更新、案情拆解 + 多 query RAG、确定性公网案例/司法实践检索、案情综合分析（`LegalCaseAnalyzer`，一次 LLM 调用同时产出风险识别 + 案情目录 + 下一步动作）。
- `session.py`：`LegalConsultationSession`，串联一轮法律咨询 workflow；含 `StreamingAnswerSanitizer` 流式答案行级清洗，以及 `export_snapshot()` / `restore_snapshot()` 会话快照导出与恢复。
- `factory.py`：`create_legal_consultation_session()`；多会话场景用 `create_legal_consultation_session_factory()`，LLM client 和 BGE-M3 检索器跨会话共享、惰性构建。

当前法律咨询主流程：

```text
legal_agent_demo.py
  -> create_legal_consultation_session()
  -> LegalConsultationSession.ask_with_events()
  -> LegalCaseStateUpdater.update()                       [LLM 调用 1]
       -> 先发 legal_missing_details_suggested 事件，提示可补充的关键信息
       -> 如 should_pause_for_supplement=true，发 legal_supplement_required，提交状态并暂停等待用户补充
       -> 调用方传 allow_pause=False 时忽略暂停判定，发 legal_supplement_skipped 后继续完整链路
  -> 启动 LegalDeterministicWebSearchSubtask.run() 后台 future
       -> 状态更新后立即启动，rag/risks/catalog/next_action 传 None
       -> 与本地 RAG 和综合分析全程并行；三条固定 query 并发检索（相似案例 / 司法解释与权威规定 / 裁判规则与司法实践），失败转 warning
       -> query 核心词取少量关键事实 + 剥掉“可能涉及”前缀的法律概念，保持短聚焦；服务端用博查 include/exclude 过滤站点
       -> 结果按域名权威度分级（high/medium/normal/low）加内容特征词重排，低置信度站点仅在结果不足时兜底回填
  -> LegalCaseRagSubtask.run()
       -> LegalQueryPlanner.plan()                        [LLM 调用 2]
       -> 预热 retriever，避免并发检索首次懒加载竞争
       -> 并发执行 LegalArticleRetriever.search_legal_articles()
       -> 并发执行 LegalArticleRetriever.search_legal_articles_by_keyword()
  -> LegalCaseAnalyzer.analyze()                          [LLM 调用 3]
       -> 一次结构化调用产出 risks + catalog + next_action
       -> 对外仍按 legal_risk_analyzed / legal_analysis_catalog_built / legal_next_action_decided 三个事件发出
  -> 等待公网检索 future，整理 legal_reference_materials 资料事件
  -> AgentRunner.run(runtime_messages, on_delta=...)      [LLM 调用 4，流式]
       -> 最终回答阶段不再开放检索工具，避免法条/案例绕过资料栏进入聊天气泡
       -> 空工具注册表 + on_delta 时走流式路径；增量经 StreamingAnswerSanitizer 行级清洗后
          以 answer_delta 事件实时推送（只推 on_event 回调，不写入 events 列表）
  -> 最终回答
```

重要规则：

- 内部子调用的原始 prompt/response 不写入主会话 `history()`。
- `LegalConsultationSession.history()` 只保存公开 system/user/assistant。
- 子任务结果通过 `AgentEvent` 暴露，如 `legal_step`、`legal_selfheal`、`legal_memory_recalled`、`legal_rag_query_started`、`case_state_updated`、`legal_missing_details_suggested`、`legal_supplement_required`、`legal_supplement_skipped`、`legal_case_rag_done`、`legal_web_search_started/done`、`legal_reference_materials`、`legal_next_action_decided`、`legal_turn_metrics`、`answer_delta`。
- “案情拆解 + 多 query RAG”是一个子任务，复用已有 query planner 和 retriever；planner 串行执行，后续多 query/关键词检索可并发 fan-out；语义检索和关键词兜底都按 planner 推荐的法律名上限展开，避免关键词只搜第一部法律造成跨法域漏召回。
- 法条证据用融合分数重排（`evidence_rank_score`）：检索分数为主体，多 query 重复命中和 planner 的 positive_terms 小幅加分、negative_terms 小幅减分；语义检索每条 query 前两名保底，其余低于 0.30 的长尾剔除。不要改回 hit_count 绝对优先的排序。
- 风险识别、案情目录和下一步动作由 `LegalCaseAnalyzer` 一次 LLM 调用合并产出（一轮成功链路共 4 次 LLM 调用：状态更新、query 规划、综合分析、最终回答）；不要拆回三次串行调用。
- 确定性公网检索复用 `web_search` 工具，在案件状态更新后立即后台启动（早于 RAG，rag 等参数传 None），与本地检索和综合分析全程并行；三条固定 query 分别面向相似案例、司法解释/权威规定和裁判规则/实务口径，司法解释 query 用 include 限定官方与专业法律站点，其余 query 用 exclude 排除低质站点；每条 query 取 10 条候选、按权威度重排后保留 5 条，`LegalWebSearchItem.authority_level` 进入最终 prompt 和资料栏（权威/专业/低置信标识）；结果进入最终临时 runtime input，并通过 `legal_reference_materials` 的安全白名单字段进入右侧参考资料栏，不写入公开 history；Web 进度区只展示脱敏计数。
- 公网来源权威度分级和站点名单（`HIGH/MEDIUM/LOW_AUTHORITY_DOMAIN_SUFFIXES`）维护在 `subtasks.py` 顶部常量区；调整名单不需要动排序逻辑。
- 状态更新器可通过 `should_pause_for_supplement` 暂停后续链路；暂停轮次作为成功轮次提交 `case_state` 和公开 user/assistant 追问历史，下一轮用户补充后再继续完整链路。用户确实无法补充时不能卡死流程：`ask_with_events(allow_pause=False)` 忽略暂停判定、发 `legal_supplement_skipped` 后继续完整链路（Web 端由 `/api/chat` 的 `skip_supplement` 字段触发，前端阻塞补充弹窗提供“无法补充，直接分析”按钮）；状态更新 prompt 同时约束“用户明确表示无法补充时不得再次暂停”。
- 链路自修复：案件状态更新、RAG、案情综合分析失败时发 `legal_selfheal`（action=degraded）并以降级产物继续（沿用旧状态 / 空法条证据 / 空风险 + 默认追问动作）；最终回答失败自动非流式重试一次（action=retried），重试再失败才整轮回滚。`detail` 字段只留内部事件，Web 白名单剥掉。
- 每轮结束（含暂停补充轮）发 `legal_turn_metrics` 事件：`stages` 各阶段耗时与状态（ok/degraded/retried）、`total_duration_ms`、`llm_usage` 轮增量（共享 `OpenAIChatClient` usage 轮前后快照做差）、`selfheal_count`。观测数据走事件通道，不加返回值。
- `AgentRunner` 仍保持通用工具调用闭环，不写法律业务逻辑；工具注册表为空且传入 `on_delta` 时走流式路径。法律咨询默认最终回答 runner 使用空工具注册表，避免模型在最后阶段调用检索工具后把长篇法条/案例塞回聊天气泡。
- 最终回答走执业律师口吻，固定章节：「结论」「当前关键点」「法律风险」（按刑事/民事/行政分层，可基于通用法律知识给罪名和法定刑档，不做个案刑期/罚金/赔偿数额预测）「现在该做什么」「还缺哪些关键信息」；章节名要和 `session.py` 的 `CORE_ANSWER_SECTION_KEYWORDS` 保持同步，改结构时两处一起改。
- 最终回答流式增量必须先过 `StreamingAnswerSanitizer` 行级清洗（与 `strip_reference_sections_from_answer` 同一套判定，按“标题形态 + 关键词”识别资料章节，正文句子里的“司法实践”等词不触发），资料章节和 URL 行不得在打字机阶段闪现；`answer_delta` 只推 `on_event` 回调，不写入返回的 events 列表。
- `legal_agent_demo.py` 默认启动时预热本地 RAG；设置 `LEGAL_RAG_PRELOAD=0` 可跳过预热。

## Web UI

目录：`web_app/` + `frontend/src/`。

核心文件：

- `web_app/server.py`：FastAPI 服务入口，提供首页、健康检查、RAG 预热、NDJSON 聊天接口、历史会话 API（`GET /api/sessions`、`GET/DELETE /api/sessions/{id}`）和运行指标聚合（`GET /api/metrics`）。
- `frontend/src/App.jsx`：React 页面顶层状态，编排聊天、事件流、预热、pause 补充和历史会话恢复/切换。
- `frontend/src/api.js`：调用 `/api/health`、`/api/preload`、`/api/chat`、`/api/sessions`。
- `frontend/src/stream.js`：读取 `/api/chat` 的 NDJSON 事件流。
- `frontend/src/eventFormatters.js`：事件标题、摘要、颜色和补充内容展示文本。
- `frontend/src/theme.js`：品牌视觉 token（深蓝/金色渐变、面板圆角阴影、焦点环常量导出）+ MUI 主题；改配色/圆角先改这里，组件从这里 import 常量。
- `frontend/src/icons.jsx`：内联 SVG 描边图标集（feather 风格工厂函数生成）；刻意不引入 @mui/icons-material 依赖。
- `frontend/src/components/PanelShell.jsx`：进度/对话/资料三栏面板的统一外壳（头部图标+标题+操作区+可选 footer），正文滚动行为仍由各面板自管。
- `frontend/src/components/MarkdownMessage.jsx`：把助手 Markdown 答复渲染成标题、列表、引用等结构化内容。
- `frontend/src/components/MaterialsPanel.jsx`：右侧参考资料栏，法条和案例/实务资料默认只显示标题，点击后展开详情。
- `frontend/src/components/SupplementDialog.jsx`：补充信息弹窗，支持阻塞性 pause 和非阻塞追问建议。
- `frontend/src/components/SessionDrawer.jsx`：左侧历史对话抽屉，支持切换、开始新对话和两步确认删除。
- `frontend/src/components/`：Material UI 页面组件。
- `web_app/static/`：Vite 构建产物，由 FastAPI 托管；不要在这里手写长期维护的前端源码。
- `vite.config.js`：`base=/static/`，构建输出到 `web_app/static/`，开发时代理 `/api` 到本地 FastAPI。

Web 调用流程：

```text
浏览器
  -> 启动时 GET /api/sessions + GET /api/sessions/{最近会话} 自动恢复上次对话
  -> POST /api/chat（带 session_id；为空则后端新建会话并在流首回传 session 事件）
  -> web_app.server.create_app()
  -> LegalConsultationSession.ask_with_events(on_event=...)
  -> queue.Queue NDJSON stream
  -> 前端实时展示 session / legal_step / legal_missing_details_suggested / legal_rag_query_started / legal_reference_materials / answer_delta / pause / error / final / done
  -> 轮末后端把快照写入 data/sessions/<session_id>/
```

重要规则：

- Web UI 是本地单用户调试/咨询入口，不做登录、多用户或数据库。
- 会话持久化走 `agent_system/storage/SessionStore` 文件方案（meta/snapshot/events），存 `data/sessions/`；每轮成功（含 pause）后原子写快照，失败轮只追加 turn_failed 事件不写快照。
- 轮级指标链路：`legal_turn_metrics` 事件过 Web 白名单后只透出阶段耗时、总耗时、llm_usage 数值和 selfheal_count；同一份白名单结果随 `turn_committed` 写入 events.jsonl；`GET /api/metrics` 扫描各会话 events.jsonl 聚合轮数、失败轮数、成功率、总耗时和 LLM usage，不维护内存计数器。
- `/api/chat` 带可选 `session_id`：为空时后端新建会话并以流首 `{"type": "session", "session_id"}` 事件回传；不存在的 session_id 返回 404。会话实例按 ID 缓存在内存，未命中时从磁盘快照 `restore_snapshot()` 恢复。
- 快照保存脱敏后的白名单资料（materials）和未完成补充请求（pending_supplement）；detail 接口返回前会再过一遍白名单，system prompt 不透出给浏览器。
- 前端刷新后自动恢复最近会话；SessionDrawer 支持切换、新对话和删除，请求进行中禁止这些操作。turn_count 为 0 的空会话不进入历史列表。
- `create_app(session=...)` 注入单会话是兼容模式：不发 session 事件、不持久化，供旧测试使用；多会话测试用 `create_app(session_factory=..., store=SessionStore(tmp))`。
- `/api/chat` 使用 `application/x-ndjson`，不要改成一次性 JSON，否则前端无法实时展示进度。
- `/api/chat` 支持补充表单字段：`supplement_answers`、`selected_questions`、`selected_evidence_gaps`、`free_text`，后端会合成为下一轮用户输入；`skip_supplement=true` 表示用户无法补充，后端合成“无法补充”声明并以 `allow_pause=False` 强制继续完整链路。
- 后端遇到 `legal_supplement_required` 会输出顶层 `pause` 事件和 `done`，不输出 `final`；前端展示补充面板，用户提交后再继续。
- 最终回答流式：后端把 `answer_delta` 业务事件转成顶层 `{"type": "answer_delta", "delta": "..."}` 流事件；前端聊天区把连续 delta 累积到同一条带打字机光标的 streaming 气泡，`final` 到达时用后端兜底清洗后的完整答案替换气泡文本。
- `answer_delta` 增量在前端先进缓冲，每 80ms 合并渲染一次（`App.jsx` 的 `ANSWER_DELTA_FLUSH_INTERVAL_MS`）；不要改回逐 delta 直接 setState，否则高频增量会把主线程打满，流式期间补充弹窗无法输入和关闭。
- 聊天区滚动容器默认吸附底部（用户主动上翻超过阈值则暂停自动滚动）；执行进度区始终跟随最新事件。
- `MessageBubble` / `EventCard` 用 `memo` 包裹，`SupplementDialog` / `ChatInput` / `EventPanel` / `MaterialsPanel` / `AppHeader` 也用 `memo` 隔离：流式增量只重渲染最后一条消息，弹窗、输入框、面板和头部子树全部跳过。
- 大屏布局为左侧执行进度、中间对话、右侧参考资料；小屏优先显示对话，再显示参考资料和执行进度。
- 参考资料栏只展示 `legal_reference_materials` 白名单字段：法条/案例标题、摘要、来源和安全 URL；默认折叠，点击标题后展开详情。
- 执行进度区只展示概括状态，不展示具体检索 query、工具原始参数、检索结果或最终回答正文。
- 后端用全局 `chat_lock` 串行处理咨询（多会话共用一把锁），避免并发写坏会话状态和底层共享 RAG 资源。
- Web 单测必须注入 fake session，不调用真实 LLM、BGE-M3 或 Chroma。
- 前端源码在 `frontend/src/`，修改后运行 `npm run build` 生成 `web_app/static/` 产物。
- Vite 生产资源路径依赖 `base=/static/`，不要随意改成 `/`，否则 FastAPI 托管时资源会 404。
- 最终聊天答复走律师口吻：结论定性、当前关键点、分刑事/民事/行政的具体风险和关键行动；详细法条、案例、司法实践和来源链接放右侧参考资料栏。
- React 前端保持轻量：使用组件内 state，不引入 Redux、React Router 或复杂前端工程层。
- `LEGAL_RAG_PRELOAD=0` 可跳过启动预热。

## Legal tools

文件：`agent_system/agent/legal_tools.py`

当前工具：

1. `search_legal_articles`
   - 语义检索。
   - 用 BGE-M3 query embedding + Chroma。
   - 适合自然语言法律问题、同义表达、法律依据查询。

2. `search_legal_articles_by_keyword`
   - 关键词精确匹配。
   - 直接从原始法条 JSON 构建轻量索引，不加载 BGE-M3。
   - 用来补 RAG 漏召回。
   - 适合明确关键词、条文片段、法律名、条号。
   - `match_mode=all` 精确；`match_mode=any` 扩召回。

3. `web_search`
   - 调用博查 Web Search 公网搜索。
   - 配置来自 `WebSearchConfig` / `BOCHA_WEB_SEARCH_*`。
   - 支持 `include` / `exclude` 服务端站点过滤参数（多域名用 `|` 分隔，博查官方上限 100 个），strict schema 中为必填字段，不需要时传空字符串。
   - 适合最新政策、新闻、公开网页、外部事实背景或本地法条库覆盖不到的信息。
   - 法律咨询中特别用于相似案例、最高法指导性/公报/典型案例、司法解释、地方裁判口径、量刑/赔偿/补偿区间和执行实务。
   - 法律咨询链路会在最终回答前确定性调用该工具补充案例/实务资料；工具失败只形成 warning，不中断本地法条分析。
   - 返回标题、链接、站点、snippet、summary、抓取时间；回答时要区分公网资料和正式法条依据。

## Retrieval 层

目录：`agent_system/retrieval/`

核心文件：

- `legal_preprocess.py`：读取 `data/最核心法条_9k.json`，构造 `LegalArticleDocument`。
- `embedding.py`：`LocalBGEEmbeddingModel`，生成 document/query embedding。
- `chroma_store.py`：`LegalChromaStore`，管理 Chroma collection。
- `legal_retriever.py`：`LegalArticleRetriever`，面向 Agent tool 的检索接口。

`LegalArticleRetriever` 关键方法：

- `search_legal_articles()`：语义检索。
- `search_legal_articles_by_keyword()`：关键词检索。
- `build_where_filter()`：metadata 过滤。
- `normalize_keywords()`：关键词规范化。
- `match_keywords_in_document()`：多字段 substring 匹配。

语义检索链路：

```text
search_legal_articles()
  -> _ensure_ready()
  -> validate_collection()
  -> embed_query()
  -> query_by_embedding()
  -> format_result_item()
```

关键词检索链路：

```text
search_legal_articles_by_keyword()
  -> normalize_keywords()
  -> _get_document_index()
  -> metadata_matches_filters()
  -> match_keywords_in_document()
  -> format_result_item()
```

Chroma 数据目录：`data/chroma/`。原始法条：`data/最核心法条_9k.json`。

## Planning 层

文件：`agent_system/planning/legal_query_planner.py`

职责：把复杂案情拆成检索 query，不做最终法律判断。

核心对象：

- `LegalIssueQuery`
- `LegalQueryPlan`
- `LegalQueryPlanner`
- `LegalQueryPlanError`

核心函数：

- `plan_legal_queries()`
- `plan_to_dict()`
- `extract_json_object()`
- `parse_json_object()`
- `validate_and_normalize_plan()`
- `filter_safe_queries()`
- `filter_safe_terms()`

边界：planner 不输出具体条号、罪名结论、违法/犯罪结论、刑期、罚金、赔偿金额。

## 测试

测试框架：`unittest`。

测试文件：

- `tests/test_openai_client.py`
- `tests/test_llm_session.py`
- `tests/test_agent_runner.py`
- `tests/test_web_search_tools.py`
- `tests/test_legal_query_planner.py`
- `tests/test_legal_consultation.py`
- `tests/test_web_app.py`
- `tests/test_session_store.py`
- `tests/test_memory_store.py`
- `tests/test_main_entry.py`
- `frontend/src/App.test.jsx`（前端 Vitest）

常用：

```bash
python -m unittest discover
python -m unittest tests.test_agent_runner
python -m unittest tests.test_legal_consultation
python -m unittest tests.test_web_app
npm run build
npm run test:ui
```

多数测试使用 fake/mock，不发真实 LLM 请求，不加载真实 BGE-M3。Chroma 实链路用 `scripts/test_legal_chroma.py`。

## 会话持久化

状态：法律咨询 Web 链路已实现（`agent_system/storage/session_store.py` + `web_app/server.py`）；CLI 和通用 AgentSession/ChatSession 尚未接入。实际 schema 和接入范围以本节与源码为准。

实际结构：

```text
data/sessions/<session_id>/
  meta.json       标题、创建/更新时间、轮次（legal_session_meta.v1）
  snapshot.json   公开 messages + case_state + materials + pending_supplement（legal_session_snapshot.v1）
  events.jsonl    session_created / turn_committed（data 含 turn_count 和白名单轮级 metrics）/ turn_failed（legal_session_event.v1）
```

原则：

- `snapshot.json` 保存可恢复 Chat-style messages，原子写（tmp + rename + fsync）。
- `events.jsonl` append-only，每行带单调递增 seq。
- 工具调用不写入 `messages`。
- 失败轮只记 turn_failed，不写快照，与会话内存回滚语义一致。
- session_id 白名单正则校验（`sess_日期_时间_4hex`），防路径穿越。
- 不上 SQLite；不保存 API key、token、cookie、私钥；不长期保存 base64。

## CLAUDE.md 同步规则

后续项目结构或关键流程变化时，要同步更新本文件，让它和项目保持对齐。

需要写入本文件的变化：

- 新增或删除入口文件。
- 新增模块或目录。
- 核心调用链变化。
- 工具注册方式变化。
- legal tools 新增、删除或语义变化。
- RAG 数据结构、索引构建方式、检索策略变化。
- 配置项或环境变量变化。
- 测试框架、测试入口、常用测试命令变化。
- 会话持久化实现状态变化。
- 用户明确给出的长期项目约束或开发偏好。

不需要写入本文件的变化：

- 小 bug fix。
- 函数内部实现细节微调。
- 临时 debug。
- 单次实验代码。
- 不影响结构的测试补充。
- 可直接从源码看出的局部细节。

本文件定位：项目地图 + 当前约束 + 常用流程。不要写成完整源码说明书，否则常驻上下文成本过高。

## 改代码时优先级

1. 先看本文件。
2. 精准读要改的源码。
3. 精准读对应测试。
4. 保持现有风格：中文 docstring/注释，轻量结构。
5. 小改动优先在现有模块完成。
6. 改 Agent 工具时沿用 `LocalTool` / `ToolRegistry` / `AgentRunner`。
