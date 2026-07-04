import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Box, Stack } from '@mui/material';
import {
  deleteSession,
  fetchHealth,
  fetchSessionDetail,
  fetchSessions,
  openChatStream,
  preloadRag,
  stringifyError,
} from './api.js';
import { buildSupplementDisplayText } from './eventFormatters.js';
import { readNdjsonStream } from './stream.js';
import AppHeader from './components/AppHeader.jsx';
import ChatPanel from './components/ChatPanel.jsx';
import EventPanel from './components/EventPanel.jsx';
import MaterialsPanel from './components/MaterialsPanel.jsx';
import SessionDrawer from './components/SessionDrawer.jsx';

const INITIAL_MESSAGES = [
  {
    id: 'initial-assistant',
    role: 'assistant',
    text: '请描述案情。我会检索本地法条后，给出一般信息参考。',
  },
];

const INITIAL_EVENTS = [
  {
    id: 'initial-event',
    event_type: 'waiting',
    title: '等待输入',
    data: {
      message: '发送问题后，会看到案件状态更新、检索状态和处理进度。',
    },
  },
];

const EMPTY_MATERIALS = { laws: [], web: [], warnings: [] };

// 最终回答流式增量的合并渲染间隔（毫秒）。
// 后端每个 LLM chunk 都会推送一行 answer_delta，本地环回下每秒可能到达几十上百次；
// 若每次都直接 setState，整棵页面树会以同样频率重渲染，且打字机气泡的 Markdown
// 需要对越来越长的全文反复重新解析，主线程被占满后弹窗输入、按钮点击全部无法响应。
// 以 80ms 为窗口合并增量，肉眼仍是连续打字机效果，渲染频率被限制在约 12 次/秒。
const ANSWER_DELTA_FLUSH_INTERVAL_MS = 80;

/**
 * 从后端事件中提取非阻塞补充入口数据。
 *
 * 只处理 legal_missing_details_suggested：后端 sanitize_event_for_web 对风险、目录和下一步
 * 动作事件只透出计数类字段，不再包含具体问题列表，因此这里不需要（也无法）从那些事件取数。
 *
 * @param {object} item 后端规范化事件。
 * @returns {object|null} 可展示的补充入口数据；无可补充内容时返回 null。
 */
function buildSupplementCandidateFromEvent(item) {
  const data = item.data || {};
  if (item.event_type === 'legal_missing_details_suggested') {
    const questions = Array.isArray(data.questions) ? data.questions : [];
    const evidenceGaps = Array.isArray(data.evidence_gaps) ? data.evidence_gaps : [];
    if (questions.length > 0 || evidenceGaps.length > 0) {
      return {
        message: data.message || '可以补充这些关键信息，让后续分析更准确。',
        reason: data.message || '',
        questions,
        evidence_gaps: evidenceGaps,
      };
    }
  }

  return null;
}

/**
 * 给补充问题快照附加用户已填写的草稿请求。
 *
 * 自动提交失败时需要把弹窗恢复到用户点击提交前的内容，而不是只恢复空问题列表。
 * 草稿只保存在前端内存里，不进入公开聊天历史；真正展示给后端的仍是 payload 本身。
 *
 * @param {object|null} supplement 补充问题快照。
 * @param {object} payload 用户提交的补充 payload。
 * @returns {object|null} 带草稿的补充问题快照。
 */
function buildDraftSupplement(supplement, payload) {
  if (!supplement) {
    return null;
  }
  return { ...supplement, draft_payload: payload };
}

/**
 * 法律咨询 Web UI 顶层组件。
 *
 * 本组件只承担页面级状态编排：服务状态、聊天消息、事件流和 pause 补充状态。所有后端协议
 * 仍然沿用 FastAPI 当前接口，尤其是 `/api/chat` 的 NDJSON 流式读取。保持状态集中在这里，
 * 是为了这个小项目不额外引入 Redux/Zustand 等复杂状态库。
 *
 * @returns {JSX.Element} Web UI 页面。
 */
export default function App() {
  const idRef = useRef(0);
  const [messages, setMessages] = useState(INITIAL_MESSAGES);
  const [events, setEvents] = useState(INITIAL_EVENTS);
  const [materials, setMaterials] = useState(EMPTY_MATERIALS);
  const [status, setStatus] = useState({ text: '正在连接服务', kind: 'pending' });
  const [isSending, setIsSending] = useState(false);
  const [isRequestActive, setIsRequestActive] = useState(false);
  const [isPreloading, setIsPreloading] = useState(false);
  const [isSupplementSubmitting, setIsSupplementSubmitting] = useState(false);
  const [supplement, setSupplement] = useState(null);
  const [supplementBlocking, setSupplementBlocking] = useState(false);
  const [supplementDialogOpen, setSupplementDialogOpen] = useState(false);
  const [queuedSupplementRequest, setQueuedSupplementRequest] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [sessionDrawerOpen, setSessionDrawerOpen] = useState(false);
  const requestActiveRef = useRef(false);
  const supplementSubmittingRef = useRef(false);
  const queuedSupplementRequestRef = useRef(null);
  // 会话 ID 同时存 ref：发送请求和流事件回调都在异步链里，必须避开闭包中的旧 state。
  const sessionIdRef = useRef(null);
  // 尚未渲染的流式增量缓冲。用 ref 而不是 state，是为了高频 answer_delta 到达时不触发渲染。
  const pendingAnswerDeltaRef = useRef('');
  const answerDeltaTimerRef = useRef(null);

  const makeId = useCallback((prefix) => {
    idRef.current += 1;
    return `${prefix}-${idRef.current}`;
  }, []);

  const appendMessage = useCallback(
    (role, text) => {
      setMessages((current) => [...current, { id: makeId('message'), role, text }]);
    },
    [makeId],
  );

  const appendAssistantDelta = useCallback(
    (text) => {
      // 备用 id 在 updater 外生成。原因是 setMessages 的 updater 必须保持纯函数，
      // 不能在里面自增 idRef，否则 React 严格模式的双调用会产生跳号。
      const fallbackId = makeId('message');
      setMessages((current) => {
        const last = current[current.length - 1];
        if (last && last.role === 'assistant' && last.streaming) {
          return [...current.slice(0, -1), { ...last, text: last.text + text }];
        }
        return [...current, { id: fallbackId, role: 'assistant', text, streaming: true }];
      });
    },
    [makeId],
  );

  const flushAnswerDelta = useCallback(() => {
    if (answerDeltaTimerRef.current !== null) {
      clearTimeout(answerDeltaTimerRef.current);
      answerDeltaTimerRef.current = null;
    }
    const buffered = pendingAnswerDeltaRef.current;
    if (buffered) {
      pendingAnswerDeltaRef.current = '';
      appendAssistantDelta(buffered);
    }
  }, [appendAssistantDelta]);

  const queueAnswerDelta = useCallback(
    (text) => {
      pendingAnswerDeltaRef.current += text;
      // 只在窗口内的第一个增量上启动定时器；窗口到期时一次性渲染窗口内累计的全部文本。
      if (answerDeltaTimerRef.current === null) {
        answerDeltaTimerRef.current = setTimeout(() => {
          answerDeltaTimerRef.current = null;
          flushAnswerDelta();
        }, ANSWER_DELTA_FLUSH_INTERVAL_MS);
      }
    },
    [flushAnswerDelta],
  );

  useEffect(
    () => () => {
      // 组件卸载时清掉未触发的定时器，避免测试环境残留异步回调。
      if (answerDeltaTimerRef.current !== null) {
        clearTimeout(answerDeltaTimerRef.current);
        answerDeltaTimerRef.current = null;
      }
    },
    [],
  );

  const finalizeAssistantMessage = useCallback(
    (text) => {
      const fallbackId = makeId('message');
      setMessages((current) => {
        const last = current[current.length - 1];
        if (last && last.role === 'assistant' && last.streaming) {
          // final 携带的是后端兜底清洗后的完整答案，直接替换流式累计文本，
          // 保证聊天历史里保存的内容和公开 history 完全一致。
          return [...current.slice(0, -1), { ...last, text, streaming: false }];
        }
        return [...current, { id: fallbackId, role: 'assistant', text }];
      });
    },
    [makeId],
  );

  const stopStreamingMessage = useCallback(() => {
    setMessages((current) => {
      const last = current[current.length - 1];
      if (last && last.role === 'assistant' && last.streaming) {
        return [...current.slice(0, -1), { ...last, streaming: false }];
      }
      return current;
    });
  }, []);

  const appendEvent = useCallback(
    (event) => {
      setEvents((current) => [...current, { id: makeId('event'), ...event }]);
    },
    [makeId],
  );

  const clearEvents = useCallback(() => {
    setEvents([]);
  }, []);

  const updateQueuedSupplementRequest = useCallback((request) => {
    queuedSupplementRequestRef.current = request;
    setQueuedSupplementRequest(request);
  }, []);

  const updateSupplementSubmitting = useCallback((submitting) => {
    supplementSubmittingRef.current = submitting;
    setIsSupplementSubmitting(submitting);
  }, []);

  const updateSessionId = useCallback((value) => {
    sessionIdRef.current = value;
    setSessionId(value);
  }, []);

  const refreshSessionList = useCallback(async () => {
    // 列表刷新失败不打断主流程：历史侧边栏属于辅助功能，聊天链路不应因它报错。
    try {
      const data = await fetchSessions();
      setSessions(Array.isArray(data.sessions) ? data.sessions : []);
    } catch {
      /* 忽略列表刷新失败，下次操作会再次尝试。 */
    }
  }, []);

  const applySessionDetail = useCallback(
    (detail) => {
      const restoredMessages = (Array.isArray(detail.messages) ? detail.messages : [])
        .filter((item) => item && (item.role === 'user' || item.role === 'assistant'))
        .map((item) => ({ id: makeId('message'), role: item.role, text: String(item.text || '') }));
      setMessages(restoredMessages.length > 0 ? restoredMessages : INITIAL_MESSAGES);

      const materialsData = detail.materials || {};
      setMaterials({
        laws: Array.isArray(materialsData.laws) ? materialsData.laws : [],
        web: Array.isArray(materialsData.web) ? materialsData.web : [],
        warnings: Array.isArray(materialsData.warnings) ? materialsData.warnings : [],
      });

      const pending = detail.pending_supplement;
      if (pending && (Array.isArray(pending.questions) ? pending.questions.length : 0) + (Array.isArray(pending.evidence_gaps) ? pending.evidence_gaps.length : 0) > 0) {
        // 上次以“等待补充”结束的会话恢复为阻塞补充状态；不自动弹窗，
        // 聊天区的补充入口卡片会提示用户继续填写。
        setSupplement({
          message: pending.message || '请先补充关键信息。',
          reason: pending.reason || '',
          questions: Array.isArray(pending.questions) ? pending.questions : [],
          evidence_gaps: Array.isArray(pending.evidence_gaps) ? pending.evidence_gaps : [],
        });
        setSupplementBlocking(true);
      } else {
        setSupplement(null);
        setSupplementBlocking(false);
      }
      setSupplementDialogOpen(false);
      updateQueuedSupplementRequest(null);
      updateSessionId(detail.session_id);
      setEvents([
        {
          id: makeId('event'),
          event_type: 'legal_step',
          title: '已载入历史会话',
          data: { message: `已恢复 ${detail.turn_count || 0} 轮对话，可以继续追问。` },
        },
      ]);
    },
    [makeId, updateQueuedSupplementRequest, updateSessionId],
  );

  const loadSession = useCallback(
    async (targetSessionId) => {
      if (requestActiveRef.current) {
        return false;
      }
      try {
        const detail = await fetchSessionDetail(targetSessionId);
        applySessionDetail(detail);
        return true;
      } catch (error) {
        appendEvent({
          event_type: 'error',
          title: '载入历史会话失败',
          data: { error: stringifyError(error) },
        });
        return false;
      }
    },
    [appendEvent, applySessionDetail],
  );

  useEffect(() => {
    let cancelled = false;

    /**
     * 初始化后端健康状态。
     *
     * 这里保留旧前端的反馈语义：成功、跳过预热、启动预热失败、服务不可达分别展示不同状态。
     * 这样用户能直接判断是页面问题、后端问题，还是本地 RAG 尚未加载。
     */
    async function refreshHealth() {
      setStatus({ text: '正在连接服务', kind: 'pending' });
      try {
        const data = await fetchHealth();
        if (cancelled) {
          return;
        }
        if (data.preloaded) {
          setStatus({ text: 'RAG 已预热', kind: 'ok' });
          return;
        }
        if (data.preload_error) {
          setStatus({ text: 'RAG 预热失败，可手动重试', kind: 'error' });
          appendEvent({
            event_type: 'error',
            title: '启动预热失败',
            data: { error: data.preload_error },
          });
          return;
        }
        if (data.startup_preload_enabled === false) {
          setStatus({ text: '已跳过自动预热', kind: 'pending' });
          return;
        }
        setStatus({ text: 'RAG 尚未预热', kind: 'pending' });
      } catch (error) {
        if (cancelled) {
          return;
        }
        const message = stringifyError(error);
        setStatus({ text: '服务连接失败', kind: 'error' });
        appendEvent({
          event_type: 'error',
          title: '无法连接后端服务',
          data: { error: message },
        });
      }
    }

    void refreshHealth();
    return () => {
      cancelled = true;
    };
  }, [appendEvent]);

  useEffect(() => {
    let cancelled = false;

    /**
     * 启动时恢复最近一次会话。
     *
     * 刷新页面后前端内存状态全部丢失；这里从后端历史列表取最近更新的会话并载入，
     * 让“刷新后内容消失”变成“刷新后自动回到上次对话”。没有历史时保持空白新对话。
     */
    async function restoreLatestSession() {
      try {
        const data = await fetchSessions();
        if (cancelled) {
          return;
        }
        const list = Array.isArray(data.sessions) ? data.sessions : [];
        setSessions(list);
        if (list.length === 0) {
          return;
        }
        const detail = await fetchSessionDetail(list[0].session_id);
        if (!cancelled) {
          applySessionDetail(detail);
        }
      } catch {
        // 历史恢复失败不阻塞新对话；用户仍可直接发送消息开始咨询。
      }
    }

    void restoreLatestSession();
    return () => {
      cancelled = true;
    };
    // 仅在挂载时执行一次；applySessionDetail 的依赖都是稳定的 useCallback。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handlePreload = useCallback(async () => {
    setIsPreloading(true);
    setStatus({ text: '正在预热 RAG', kind: 'pending' });
    appendEvent({
      event_type: 'legal_step',
      title: '开始预热本地法条 RAG',
      data: { message: '正在加载本地 embedding 模型、Chroma collection 和关键词索引。' },
    });

    try {
      const data = await preloadRag();
      setStatus({ text: 'RAG 已预热', kind: 'ok' });
      appendEvent({
        event_type: 'done',
        title: data.message || '本地法条 RAG 预热完成',
        data,
      });
    } catch (error) {
      const message = stringifyError(error);
      setStatus({ text: 'RAG 预热失败', kind: 'error' });
      appendEvent({
        event_type: 'error',
        title: '预热失败',
        data: { error: message },
      });
    } finally {
      setIsPreloading(false);
    }
  }, [appendEvent]);

  const handleStreamItem = useCallback(
    (item, outcome) => {
      if (item.type === 'session') {
        // 后端在流首回传本轮实际使用的会话 ID；新会话由后端分配，
        // 前端必须记住它，后续轮次和刷新恢复才能路由到同一会话。
        if (typeof item.session_id === 'string' && item.session_id) {
          updateSessionId(item.session_id);
        }
        return;
      }

      if (item.type === 'event') {
        appendEvent(item);
        if (item.event_type === 'legal_reference_materials') {
          setMaterials({
            laws: Array.isArray(item.data?.laws) ? item.data.laws : [],
            web: Array.isArray(item.data?.web) ? item.data.web : [],
            warnings: Array.isArray(item.data?.warnings) ? item.data.warnings : [],
          });
        }
        const supplementCandidate = buildSupplementCandidateFromEvent(item);
        if (supplementCandidate) {
          // 非阻塞补充不会中断最终回答；保存为聊天区按钮入口，用户可稍后主动打开弹窗逐条补充。
          setSupplement(supplementCandidate);
          setSupplementBlocking(false);
        }
        return;
      }

      if (item.type === 'answer_delta') {
        const deltaText = typeof item.delta === 'string' ? item.delta : '';
        if (deltaText) {
          queueAnswerDelta(deltaText);
        }
        return;
      }

      if (item.type === 'final') {
        outcome.hasFinal = true;
        // 先渲染缓冲中的残余增量，让 finalize 稳定命中“替换最后一条流式气泡”分支，
        // 而不是在缓冲未渲染时误走“新增一条消息”的兜底分支。
        flushAnswerDelta();
        finalizeAssistantMessage(item.answer || '');
        setStatus({ text: '答复已生成', kind: 'ok' });
        return;
      }

      if (item.type === 'pause') {
        outcome.hasPause = true;
        // pause 表示本轮已经转入“等待用户补充”阶段。这里提前解除按钮文案的发送态，
        // 但 requestActiveRef 仍保持为 true，直到 NDJSON done 到达后才允许补充内容自动提交。
        setIsSending(false);
        setSupplement(item);
        setSupplementBlocking(true);
        setSupplementDialogOpen(true);
        appendMessage('assistant', item.message || '请先补充关键信息。');
        setStatus({ text: '等待补充信息', kind: 'pending' });
        return;
      }

      if (item.type === 'error') {
        const message = item.message || '未知错误';
        outcome.hasError = true;
        // 流式中途失败时先渲染缓冲增量再固化文本，既不丢已收到的内容，
        // 也避免残留一个永远闪烁的打字机气泡。
        flushAnswerDelta();
        stopStreamingMessage();
        appendMessage('error', message);
        appendEvent({
          event_type: 'error',
          title: '执行出错',
          data: { error: message },
        });
        setStatus({ text: '处理失败', kind: 'error' });
        return;
      }

      if (item.type === 'done') {
        appendEvent({
          event_type: 'done',
          title: '本轮处理完成',
          data: {},
        });
        if (!outcome.hasError && !outcome.hasPause) {
          setStatus({ text: '空闲', kind: 'ok' });
        }
        return;
      }

      appendEvent({
        event_type: item.type || 'unknown',
        title: item.type || '未知事件',
        data: item,
      });
    },
    [appendEvent, appendMessage, finalizeAssistantMessage, flushAnswerDelta, queueAnswerDelta, stopStreamingMessage, updateSessionId],
  );

  const sendChatRequest = useCallback(
    async (payload, options = {}) => {
      if (requestActiveRef.current) {
        return false;
      }

      const displayText = options.displayUserText || payload.message || '我补充以下关键信息。';
      const outcome = { hasError: false, hasFinal: false, hasPause: false };

      requestActiveRef.current = true;
      setIsRequestActive(true);
      setIsSending(true);
      // 丢弃上一轮可能残留的未渲染增量，避免异常中断的旧文本混进新一轮回答。
      if (answerDeltaTimerRef.current !== null) {
        clearTimeout(answerDeltaTimerRef.current);
        answerDeltaTimerRef.current = null;
      }
      pendingAnswerDeltaRef.current = '';
      if (options.clearEventLog !== false) {
        clearEvents();
        // 每轮分析对应一组资料。清空旧资料可以避免用户把上一轮法条/案例误认为本轮依据。
        setMaterials(EMPTY_MATERIALS);
      }
      appendMessage('user', displayText);
      setStatus({ text: '正在处理咨询', kind: 'pending' });

      try {
        // 每轮请求都带上当前会话 ID；为空时后端会新建会话并通过流首 session 事件回传 ID。
        const body = await openChatStream({ ...payload, session_id: sessionIdRef.current });
        await readNdjsonStream(body, (item) => handleStreamItem(item, outcome));
      } catch (error) {
        const message = stringifyError(error);
        outcome.hasError = true;
        // fetch/解析异常不会经过 item.type === 'error' 分支，同样先渲染缓冲增量再收起流式光标，
        // 避免丢字或残留打字机状态。
        flushAnswerDelta();
        stopStreamingMessage();
        setStatus({ text: '处理失败', kind: 'error' });
        appendMessage('error', message);
        appendEvent({
          event_type: 'error',
          title: '请求失败',
          data: { error: message },
        });
      } finally {
        const queuedRequest = queuedSupplementRequestRef.current;
        if (outcome.hasError && queuedRequest) {
          // 上一轮已经失败时不能继续消费排队补充，否则会把失败链路后的上下文误当成可续跑状态。
          // 恢复弹窗和草稿，让用户看到失败原因后自行决定重试或修改补充内容。
          updateQueuedSupplementRequest(null);
          setSupplement(queuedRequest.previousSupplement);
          setSupplementBlocking(queuedRequest.previousBlocking);
          setSupplementDialogOpen(true);
          setMaterials(queuedRequest.previousMaterials || EMPTY_MATERIALS);
        }
        requestActiveRef.current = false;
        setIsRequestActive(false);
        setIsSending(false);
        // 本轮成功（含 pause）后后端已更新快照；刷新侧边栏的标题、时间和轮次。
        void refreshSessionList();
      }

      return !outcome.hasError;
    },
    [appendEvent, appendMessage, clearEvents, flushAnswerDelta, handleStreamItem, refreshSessionList, stopStreamingMessage, updateQueuedSupplementRequest],
  );

  useEffect(() => {
    if (isRequestActive || !queuedSupplementRequest) {
      return;
    }

    // 补充内容必须等上一轮流式响应完整结束后再发给后端。后端使用全局咨询锁串行处理，
    // 提前并发请求只会得到“正在处理”的错误；用前端队列可以让用户先完成编辑，再自动续跑。
    // 这里从 ref 取值，原因是 React StrictMode 可能重复执行 effect；ref 置空后可避免同一草稿重复提交。
    const request = queuedSupplementRequestRef.current;
    if (!request) {
      return;
    }
    const materialsBeforeQueuedSubmit = materials;
    updateQueuedSupplementRequest(null);
    void (async () => {
      const succeeded = await sendChatRequest(request.payload, {
        displayUserText: request.displayUserText,
        clearEventLog: true,
      });
      if (!succeeded) {
        setSupplement(request.previousSupplement);
        setSupplementBlocking(request.previousBlocking);
        setSupplementDialogOpen(true);
        setMaterials(materialsBeforeQueuedSubmit);
      }
    })();
  }, [isRequestActive, materials, queuedSupplementRequest, sendChatRequest, updateQueuedSupplementRequest]);

  const handleUserSubmit = useCallback(
    async (text) => {
      if (requestActiveRef.current || supplementBlocking || queuedSupplementRequestRef.current || supplementSubmittingRef.current) {
        return false;
      }
      setSupplement(null);
      setSupplementBlocking(false);
      setSupplementDialogOpen(false);
      return sendChatRequest({ message: text }, { displayUserText: text, clearEventLog: true });
    },
    [sendChatRequest, supplementBlocking],
  );

  const handleSupplementContinue = useCallback(
    async (payload) => {
      if (supplementSubmittingRef.current || queuedSupplementRequestRef.current || !supplement) {
        return false;
      }

      const displayUserText = buildSupplementDisplayText(payload);
      const previousSupplement = buildDraftSupplement(supplement, payload);
      const previousBlocking = supplementBlocking;
      const previousMaterials = materials;
      if (requestActiveRef.current) {
        // 当前回答仍在流式生成时，只在前端暂存用户已经编辑好的补充内容。
        // 这样既不锁住弹窗输入，也不会向仍被后端 chat_lock 占用的接口发起必然失败的并发请求。
        updateQueuedSupplementRequest({ payload, displayUserText, previousSupplement, previousBlocking, previousMaterials });
        setSupplement(null);
        setSupplementBlocking(false);
        setSupplementDialogOpen(false);
        setStatus({ text: '补充内容已暂存，等待当前答复完成', kind: 'pending' });
        appendEvent({
          event_type: 'legal_step',
          title: '补充内容已暂存',
          data: { message: '当前回答还在生成，补充内容会在本轮结束后自动提交。' },
        });
        return true;
      }

      updateSupplementSubmitting(true);
      setSupplement(null);
      setSupplementBlocking(false);
      setSupplementDialogOpen(false);
      try {
        const requestPromise = sendChatRequest(payload, {
          displayUserText,
          clearEventLog: true,
        });
        // sendChatRequest 的同步段已把 requestActiveRef 置为 true，且本弹窗对应的 supplement
        // 已被清空：此后再触发提交只会命中上方“暂存等待本轮结束”分支，双击保护不再依赖
        // submitting 锁。这里必须立即释放锁，而不是等整轮流式响应结束——本轮中途如果后端
        // 又给出新的补充建议，残留的 submitting 会把新弹窗的输入框和关闭按钮一起禁用到本轮
        // 结束，用户会遇到“弹窗打开后既不能输入也关不掉”的卡死体验。
        updateSupplementSubmitting(false);
        const succeeded = await requestPromise;
        if (!succeeded) {
          setSupplement(previousSupplement);
          setSupplementBlocking(previousBlocking);
          setSupplementDialogOpen(true);
          setMaterials(previousMaterials);
        }
        return succeeded;
      } finally {
        updateSupplementSubmitting(false);
      }
    },
    [appendEvent, materials, sendChatRequest, supplement, supplementBlocking, updateQueuedSupplementRequest, updateSupplementSubmitting],
  );

  const handleOpenSupplement = useCallback(() => {
    setSupplementDialogOpen(true);
  }, []);

  const handleCloseSupplement = useCallback(() => {
    if (!supplementSubmittingRef.current) {
      setSupplementDialogOpen(false);
    }
  }, []);

  const handleOpenSessions = useCallback(() => {
    // 打开抽屉时顺带刷新一次列表，保证标题和时间是最新的。
    void refreshSessionList();
    setSessionDrawerOpen(true);
  }, [refreshSessionList]);

  const handleCloseSessions = useCallback(() => {
    setSessionDrawerOpen(false);
  }, []);

  const resetToNewSession = useCallback(() => {
    updateSessionId(null);
    setMessages(INITIAL_MESSAGES);
    setEvents(INITIAL_EVENTS);
    setMaterials(EMPTY_MATERIALS);
    setSupplement(null);
    setSupplementBlocking(false);
    setSupplementDialogOpen(false);
    updateQueuedSupplementRequest(null);
    setStatus({ text: '空闲', kind: 'ok' });
  }, [updateQueuedSupplementRequest, updateSessionId]);

  const handleNewSession = useCallback(() => {
    if (requestActiveRef.current) {
      return;
    }
    // 新对话只重置前端状态；会话目录等到用户真正发送第一条消息时才由后端创建。
    resetToNewSession();
    setSessionDrawerOpen(false);
  }, [resetToNewSession]);

  const handleSelectSession = useCallback(
    async (targetSessionId) => {
      if (requestActiveRef.current) {
        return;
      }
      if (targetSessionId === sessionIdRef.current) {
        setSessionDrawerOpen(false);
        return;
      }
      const succeeded = await loadSession(targetSessionId);
      if (succeeded) {
        setSessionDrawerOpen(false);
      }
    },
    [loadSession],
  );

  const handleDeleteSession = useCallback(
    async (targetSessionId) => {
      if (requestActiveRef.current) {
        return;
      }
      try {
        await deleteSession(targetSessionId);
      } catch (error) {
        appendEvent({
          event_type: 'error',
          title: '删除会话失败',
          data: { error: stringifyError(error) },
        });
        return;
      }
      if (targetSessionId === sessionIdRef.current) {
        // 当前会话被删除后回到空白新对话，避免继续往已删除的会话里发消息。
        resetToNewSession();
      }
      void refreshSessionList();
    },
    [appendEvent, refreshSessionList, resetToNewSession],
  );

  const sessionActionsBusy = useMemo(
    () => isSending || isRequestActive || isSupplementSubmitting || Boolean(queuedSupplementRequest),
    [isRequestActive, isSending, isSupplementSubmitting, queuedSupplementRequest],
  );

  const inputDisabled = useMemo(
    () => isSending || isRequestActive || supplementBlocking || Boolean(queuedSupplementRequest),
    [isRequestActive, isSending, queuedSupplementRequest, supplementBlocking],
  );

  return (
    <Box
      component="main"
      aria-label="法律咨询 Agent 控制台"
      sx={{
        minHeight: '100vh',
        p: { xs: 1.5, md: 2.5 },
      }}
    >
      <Stack
        spacing={2.25}
        sx={{
          width: 'min(1680px, 100%)',
          mx: 'auto',
          minHeight: { xs: 'calc(100vh - 24px)', md: 'calc(100vh - 40px)' },
          height: { xs: 'auto', md: 'calc(100vh - 40px)' },
        }}
      >
        <AppHeader
          status={status}
          isPreloading={isPreloading}
          preloadDisabled={isPreloading || isRequestActive}
          onPreload={handlePreload}
          onOpenSessions={handleOpenSessions}
        />

        <SessionDrawer
          open={sessionDrawerOpen}
          sessions={sessions}
          activeSessionId={sessionId}
          busy={sessionActionsBusy}
          onClose={handleCloseSessions}
          onSelect={handleSelectSession}
          onNew={handleNewSession}
          onDelete={handleDeleteSession}
        />

        <Box
          component="section"
          aria-label="咨询工作区"
          sx={{
            display: 'grid',
            gridTemplateColumns: {
              xs: '1fr',
              lg: 'minmax(280px, 0.9fr) minmax(0, 1.7fr) minmax(320px, 0.95fr)',
            },
            gridTemplateAreas: {
              xs: '"chat" "materials" "events"',
              lg: '"events chat materials"',
            },
            gap: 2.25,
            minHeight: 0,
            flex: 1,
          }}
        >
          <Box sx={{ gridArea: 'chat', minHeight: 0, display: 'grid' }}>
            <ChatPanel
              messages={messages}
              supplement={supplement}
              supplementBlocking={supplementBlocking}
              supplementDialogOpen={supplementDialogOpen}
              inputDisabled={inputDisabled}
              isSending={isSending}
              supplementSubmitting={isSupplementSubmitting}
              onSubmit={handleUserSubmit}
              onOpenSupplement={handleOpenSupplement}
              onCloseSupplement={handleCloseSupplement}
              onSupplementContinue={handleSupplementContinue}
            />
          </Box>
          <Box sx={{ gridArea: 'events', minHeight: 0, display: 'grid' }}>
            <EventPanel events={events} onClear={clearEvents} />
          </Box>
          <Box sx={{ gridArea: 'materials', minHeight: 0, display: 'grid' }}>
            <MaterialsPanel materials={materials} />
          </Box>
        </Box>
      </Stack>
    </Box>
  );
}
