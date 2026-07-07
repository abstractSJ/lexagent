# LexAgent — 法律咨询智能体

基于 LLM 的端到端法律咨询 Agent 应用：多轮案情咨询业务链路，本地法条 RAG 与公网案例检索双源增强，全链路流式 Web UI，内置链路自修复与运行指标观测。

> 本项目仅用于 Agent 工程与 RAG 技术学习研究，输出内容不构成正式法律意见。涉及真实法律事务请咨询执业律师。

---

## 核心特性

- **多轮咨询编排**：案件状态更新 → 缺失信息追问（可暂停等待补充）→ 案情拆解 → 多路检索 → 结构化综合分析 → 律师口吻流式回答；一轮成功链路仅 4 次 LLM 调用。
- **本地法条 RAG**：BGE-M3 + Chroma 覆盖 9000+ 核心法条；LLM query planner 将案情拆解为多路检索子问题，语义检索与关键词精确匹配双路并发召回，融合分数重排。
- **并行公网检索**：相似案例 / 司法解释 / 裁判规则三路固定 query 在链路早期后台启动，与本地 RAG、综合分析全程并行；结果按域名权威度分级重排。
- **自研 Agent 框架**：基于 OpenAI-compatible Responses API 的 Function Calling 工具调用闭环（AgentRunner + ToolRegistry + 事件流），框架层与业务层解耦。
- **链路自修复**：子任务失败自动降级续跑，最终回答失败自动重试；每轮输出各阶段耗时与 LLM token 用量指标，后端聚合为成功率 / 耗时统计接口。
- **会话持久化与跨会话记忆**：快照原子写 + append-only 事件日志，刷新自动恢复；案件状态沉淀为跨会话记忆并按关键词召回注入新会话。
- **流式 Web UI**：FastAPI NDJSON 流式接口 + React（Material UI）前端，打字机式流式渲染、执行进度面板、参考资料栏与历史会话侧边栏。

---

## 单轮咨询链路

```text
用户输入
  -> 案件状态更新                    [LLM 1] 可暂停追问关键缺失信息
  -> 公网案例检索（后台并行启动）      三路固定 query，权威度分级重排
  -> 案情拆解 query 规划              [LLM 2]
  -> 本地法条 RAG                     语义 + 关键词双路并发，融合重排
  -> 综合分析                         [LLM 3] 风险识别 + 案情目录 + 下一步动作
  -> 汇合公网检索结果 -> 参考资料栏
  -> 最终回答                         [LLM 4] 流式输出 + 行级清洗
```

---

## 快速开始

### 1. 环境准备

推荐 Python 3.10+、Node.js 18+。

```bash
pip install -r requirements.txt
npm install
```

### 2. 配置 API Key

方式一：环境变量。

```bash
export AGENT_LLM_API_KEY="你的 Key"
export AGENT_LLM_BASE_URL="https://你的 OpenAI-compatible 端点/v1"
export AGENT_LLM_MODEL="模型名"
export BOCHA_WEB_SEARCH_API_KEY="博查 Web Search Key"   # 可选，公网检索用
```

方式二：本地配置文件（不入 git）。

```bash
cp agent_system/config_local.example.py agent_system/config_local.py
# 编辑 config_local.py 填入真实 Key
```

读取优先级：环境变量 > `config_local.py` > 内置默认值。

### 3. 构建法条向量库

首次运行会下载 BGE-M3 模型；embedding 默认使用 CPU。确认 CUDA 环境稳定后，可设 `LEGAL_EMBEDDING_DEVICE=cuda` 加速。

```bash
python scripts/build_legal_chroma.py
python scripts/test_legal_chroma.py --metadata-only   # 验证
```

### 4. 构建前端并启动

```bash
npm run build          # 生成 web_app/static/
python main.py         # 默认启动本地 Web UI
```

其他入口：

```bash
python main.py --mode chat        # 普通聊天 CLI
python legal_agent_demo.py        # 法律咨询 Agent CLI
```

---

## 目录结构

```text
agent_system/
  config.py            项目配置与加载函数
  llm/                 LLM client、消息格式、普通聊天 session
  agent/               工具注册、AgentRunner、legal / web_search 工具
  legal_consultation/  法律咨询多轮状态、子任务与执行链路
  retrieval/           法条预处理、BGE-M3 embedding、Chroma、检索器
  planning/            法律案情 query planner
  storage/             会话持久化（SessionStore）
  memory/              跨会话案件记忆（MemoryStore）
frontend/src/          React + Material UI 前端源码
web_app/               FastAPI 服务与 Vite 构建产物
scripts/               向量库构建 / 检索测试 / query 规划 CLI
tests/                 unittest 单元测试
data/                  法条原始数据（向量库、会话、记忆运行时生成，不入库）
```

---

## 测试

后端与前端测试全链路使用 fake / mock，不调用真实 LLM、BGE-M3 或 Chroma。

```bash
python -m unittest discover     # 后端 148 个用例
npm run test:ui                 # 前端 Vitest
```

---

## 设计要点

- **检索质量**：法条证据用融合分数重排——检索分数为主体，多 query 重复命中与 planner 正负词小幅调权，每路语义 query 前两名保底、低分长尾剔除；关键词精确匹配按推荐法律名展开，兜底 RAG 漏召回。
- **调用效率**：风险识别、案情目录、下一步动作合并为一次结构化 LLM 调用；公网检索与本地链路全程并行，减少串行等待。
- **边界隔离**：工具调用过程只走事件通道，不污染公开会话历史；最终回答阶段关闭检索工具，法条与案例统一进入右侧参考资料栏而非聊天气泡。
- **可靠性**：状态更新 / RAG / 综合分析失败时降级续跑，最终回答失败自动重试，重试再失败才整轮回滚；失败轮不写会话快照，与内存回滚语义一致。
- **安全**：session_id 白名单校验防路径穿越；快照与接口输出均过白名单字段，system prompt 与工具原始参数不透出给浏览器。
