import '@testing-library/jest-dom/vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import App from './App.jsx';
import { deleteSession, fetchHealth, fetchSessionDetail, fetchSessions, openChatStream } from './api.js';
import { readNdjsonStream } from './stream.js';

vi.mock('./api.js', () => ({
  fetchHealth: vi.fn(),
  openChatStream: vi.fn(),
  preloadRag: vi.fn(),
  fetchSessions: vi.fn(),
  fetchSessionDetail: vi.fn(),
  deleteSession: vi.fn(),
  stringifyError: (error) => (error instanceof Error ? error.message : String(error)),
}));

vi.mock('./stream.js', () => ({
  readNdjsonStream: vi.fn(),
}));

/**
 * 创建可手动完成的 Promise。
 *
 * 前端补充排队逻辑必须覆盖“上一轮 NDJSON 仍在读取”的窗口；用 deferred 可以精确卡住
 * 第一轮回答，避免测试依赖真实网络速度或 setTimeout 之类不稳定等待。
 *
 * @returns {{promise: Promise<void>, resolve: () => void}} 可从测试中控制完成时机的 Promise。
 */
function createDeferred() {
  let resolve;
  const promise = new Promise((innerResolve) => {
    resolve = innerResolve;
  });
  return { promise, resolve };
}

describe('App 补充信息交互', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fetchHealth.mockResolvedValue({ preloaded: true });
    fetchSessions.mockResolvedValue({ sessions: [] });
  });

  afterEach(() => {
    cleanup();
  });

  test('最终回答流式生成期间，补充弹窗仍可输入和关闭，增量按节流窗口合并渲染', async () => {
    const user = userEvent.setup();
    const firstStreamDone = createDeferred();

    openChatStream.mockResolvedValueOnce({ id: 'streaming-answer' });

    readNdjsonStream.mockImplementation(async (body, onItem) => {
      onItem({
        type: 'event',
        event_type: 'legal_missing_details_suggested',
        title: '建议补充',
        data: {
          message: '可以补充这些关键信息。',
          questions: ['劳动合同是什么时间开始没有签的？'],
          evidence_gaps: [],
        },
      });
      // 连续多个增量模拟后端逐 chunk 推送；节流窗口到期后应合并成同一段文本一次渲染。
      onItem({ type: 'answer_delta', delta: '根据' });
      onItem({ type: 'answer_delta', delta: '劳动合同法' });
      onItem({ type: 'answer_delta', delta: '第八十二条分析如下。' });
      await firstStreamDone.promise;
      onItem({ type: 'final', answer: '最终清洗后的答复' });
      onItem({ type: 'done' });
    });

    render(<App />);

    await user.type(screen.getByRole('textbox', { name: '案情或追问' }), '公司一直不签劳动合同');
    await user.click(screen.getByRole('button', { name: '发送' }));

    await screen.findByText('根据劳动合同法第八十二条分析如下。');

    // 流式仍未结束：弹窗必须可以打开、输入和关闭，不能等回答完成。
    await user.click(await screen.findByRole('button', { name: '逐条补充' }));
    const freeTextInput = screen.getByRole('textbox', { name: '其他补充说明' });
    expect(freeTextInput).toBeEnabled();
    await user.type(freeTextInput, '我从 2024 年 1 月入职。');
    expect(freeTextInput).toHaveValue('我从 2024 年 1 月入职。');

    expect(screen.getByRole('button', { name: '关闭补充信息弹窗' })).toBeEnabled();
    await user.click(screen.getByRole('button', { name: '关闭补充信息弹窗' }));
    await waitFor(() => expect(screen.queryByRole('dialog', { name: /补充关键信息/ })).not.toBeInTheDocument());

    firstStreamDone.resolve();

    await screen.findByText('最终清洗后的答复');
    expect(screen.queryByText('根据劳动合同法第八十二条分析如下。')).not.toBeInTheDocument();
  });

  test('当前回答尚未结束时提交补充内容，应先在前端排队而不是并发请求后端', async () => {
    const user = userEvent.setup();
    const firstStreamDone = createDeferred();

    openChatStream
      .mockResolvedValueOnce({ id: 'first-stream' })
      .mockResolvedValueOnce({ id: 'queued-supplement-stream' });

    readNdjsonStream.mockImplementation(async (body, onItem) => {
      if (body.id === 'first-stream') {
        onItem({
          type: 'event',
          event_type: 'legal_missing_details_suggested',
          title: '建议补充',
          data: {
            message: '可以补充这些关键信息。',
            questions: ['劳动合同是什么时间开始没有签的？'],
            evidence_gaps: [],
          },
        });
        await firstStreamDone.promise;
        onItem({ type: 'final', answer: '第一轮答复' });
        onItem({ type: 'done' });
        return;
      }

      onItem({ type: 'final', answer: '已收到补充内容' });
      onItem({ type: 'done' });
    });

    render(<App />);

    await user.type(screen.getByRole('textbox', { name: '案情或追问' }), '公司一直不签劳动合同');
    await user.click(screen.getByRole('button', { name: '发送' }));

    await user.click(await screen.findByRole('button', { name: '逐条补充' }));
    expect(screen.getByRole('dialog', { name: /补充关键信息/ })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '关闭补充信息弹窗' }));
    await waitFor(() => expect(screen.queryByRole('dialog', { name: /补充关键信息/ })).not.toBeInTheDocument());

    await user.click(screen.getByRole('button', { name: '逐条补充' }));
    await user.type(screen.getByRole('textbox', { name: '其他补充说明' }), '我从 2024 年 1 月入职。');
    await user.click(screen.getByRole('button', { name: '提交补充并继续' }));

    expect(openChatStream).toHaveBeenCalledTimes(1);

    firstStreamDone.resolve();

    await waitFor(() => expect(openChatStream).toHaveBeenCalledTimes(2));
    expect(openChatStream).toHaveBeenLastCalledWith(
      expect.objectContaining({
        free_text: '我从 2024 年 1 月入职。',
        message: '我补充以下关键信息：',
      }),
    );
  });

  test('阻塞性 pause 的 NDJSON 尚未结束时提交补充，也必须等 done 后再自动提交', async () => {
    const user = userEvent.setup();
    const firstStreamDone = createDeferred();

    openChatStream
      .mockResolvedValueOnce({ id: 'pause-stream' })
      .mockResolvedValueOnce({ id: 'queued-after-pause-stream' });

    readNdjsonStream.mockImplementation(async (body, onItem) => {
      if (body.id === 'pause-stream') {
        onItem({
          type: 'pause',
          message: '请先补充关键信息。',
          questions: ['劳动合同是什么时间开始没有签的？'],
          evidence_gaps: [],
        });
        await firstStreamDone.promise;
        onItem({ type: 'done' });
        return;
      }

      onItem({ type: 'final', answer: '已根据补充继续分析' });
      onItem({ type: 'done' });
    });

    render(<App />);

    await user.type(screen.getByRole('textbox', { name: '案情或追问' }), '公司一直不签劳动合同');
    await user.click(screen.getByRole('button', { name: '发送' }));

    await screen.findByRole('dialog', { name: /请先补充关键信息/ });
    await user.type(screen.getByRole('textbox', { name: '其他补充说明' }), '我从 2024 年 1 月入职。');
    await user.click(screen.getByRole('button', { name: '提交补充并继续' }));

    expect(openChatStream).toHaveBeenCalledTimes(1);

    firstStreamDone.resolve();

    await waitFor(() => expect(openChatStream).toHaveBeenCalledTimes(2));
    expect(openChatStream).toHaveBeenLastCalledWith(
      expect.objectContaining({
        free_text: '我从 2024 年 1 月入职。',
        message: '我补充以下关键信息：',
      }),
    );
  });

  test('排队补充自动提交失败后，应恢复原补充弹窗和已填写内容', async () => {
    const user = userEvent.setup();
    const firstStreamDone = createDeferred();

    openChatStream
      .mockResolvedValueOnce({ id: 'first-stream' })
      .mockResolvedValueOnce({ id: 'failed-queued-supplement-stream' });

    readNdjsonStream.mockImplementation(async (body, onItem) => {
      if (body.id === 'first-stream') {
        onItem({
          type: 'event',
          event_type: 'legal_missing_details_suggested',
          title: '建议补充',
          data: {
            message: '可以补充这些关键信息。',
            questions: ['劳动合同是什么时间开始没有签的？'],
            evidence_gaps: ['工资流水'],
          },
        });
        await firstStreamDone.promise;
        onItem({
          type: 'event',
          event_type: 'legal_reference_materials',
          title: '参考资料',
          data: {
            laws: [{ id: 'law-1', material_type: 'law', title: '劳动合同法 第十条' }],
            web: [],
            warnings: [],
          },
        });
        onItem({ type: 'final', answer: '第一轮答复' });
        onItem({ type: 'done' });
        return;
      }

      onItem({ type: 'error', message: '后端忙碌，请稍后重试。' });
      onItem({ type: 'done' });
    });

    render(<App />);

    await user.type(screen.getByRole('textbox', { name: '案情或追问' }), '公司一直不签劳动合同');
    await user.click(screen.getByRole('button', { name: '发送' }));
    await user.click(await screen.findByRole('button', { name: '逐条补充' }));
    await user.type(screen.getByRole('textbox', { name: '其他补充说明' }), '我从 2024 年 1 月入职。');
    await user.click(screen.getByRole('button', { name: '提交补充并继续' }));

    firstStreamDone.resolve();

    await waitFor(() => expect(openChatStream).toHaveBeenCalledTimes(2));
    await screen.findByRole('dialog', { name: /补充关键信息/ });
    expect(screen.getByRole('textbox', { name: '其他补充说明' })).toHaveValue('我从 2024 年 1 月入职。');
    expect(screen.getByText('劳动合同法 第十条')).toBeInTheDocument();
  });

  test('上一轮请求失败时，不应消费已排队的补充内容', async () => {
    const user = userEvent.setup();
    const firstStreamDone = createDeferred();

    openChatStream.mockResolvedValueOnce({ id: 'failed-first-stream' });

    readNdjsonStream.mockImplementation(async (_body, onItem) => {
      onItem({
        type: 'event',
        event_type: 'legal_missing_details_suggested',
        title: '建议补充',
        data: {
          message: '可以补充这些关键信息。',
          questions: ['劳动合同是什么时间开始没有签的？'],
          evidence_gaps: [],
        },
      });
      await firstStreamDone.promise;
      onItem({ type: 'error', message: '第一轮处理失败。' });
      onItem({ type: 'done' });
    });

    render(<App />);

    await user.type(screen.getByRole('textbox', { name: '案情或追问' }), '公司一直不签劳动合同');
    await user.click(screen.getByRole('button', { name: '发送' }));
    await user.click(await screen.findByRole('button', { name: '逐条补充' }));
    await user.type(screen.getByRole('textbox', { name: '其他补充说明' }), '我从 2024 年 1 月入职。');
    await user.click(screen.getByRole('button', { name: '提交补充并继续' }));

    firstStreamDone.resolve();

    await screen.findByRole('dialog', { name: /补充关键信息/ });
    expect(openChatStream).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('textbox', { name: '其他补充说明' })).toHaveValue('我从 2024 年 1 月入职。');
  });
});

describe('App 历史会话侧边栏', () => {
  const SESSION_A = {
    session_id: 'sess_20260703_100000_ab01',
    title: '公司不签劳动合同',
    created_at: '2026-07-03T10:00:00+00:00',
    updated_at: '2026-07-03T10:05:00+00:00',
    turn_count: 2,
  };
  const SESSION_A_DETAIL = {
    session_id: SESSION_A.session_id,
    title: SESSION_A.title,
    updated_at: SESSION_A.updated_at,
    turn_count: 2,
    messages: [
      { role: 'user', text: '公司一直不签劳动合同怎么办？' },
      { role: 'assistant', text: '可以主张二倍工资。' },
    ],
    materials: {
      laws: [{ id: 'law-1', material_type: 'law', title: '劳动合同法 第八十二条' }],
      web: [],
      warnings: [],
    },
    pending_supplement: null,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    fetchHealth.mockResolvedValue({ preloaded: true });
    fetchSessions.mockResolvedValue({ sessions: [] });
  });

  afterEach(() => {
    cleanup();
  });

  test('启动时自动恢复最近一次历史会话', async () => {
    fetchSessions.mockResolvedValue({ sessions: [SESSION_A] });
    fetchSessionDetail.mockResolvedValue(SESSION_A_DETAIL);

    render(<App />);

    await screen.findByText('公司一直不签劳动合同怎么办？');
    expect(screen.getByText('可以主张二倍工资。')).toBeInTheDocument();
    expect(screen.getByText('劳动合同法 第八十二条')).toBeInTheDocument();
    expect(fetchSessionDetail).toHaveBeenCalledWith(SESSION_A.session_id);
  });

  test('新会话首轮记住后端分配的 session_id，下一轮请求自动携带', async () => {
    const user = userEvent.setup();
    openChatStream.mockResolvedValue({ id: 'stream' });
    readNdjsonStream.mockImplementation(async (_body, onItem) => {
      onItem({ type: 'session', session_id: 'sess_20260703_110000_cd02' });
      onItem({ type: 'final', answer: '阶段性答复' });
      onItem({ type: 'done' });
    });

    render(<App />);

    await user.type(screen.getByRole('textbox', { name: '案情或追问' }), '第一轮案情');
    await user.click(screen.getByRole('button', { name: '发送' }));
    await waitFor(() => expect(openChatStream).toHaveBeenCalledTimes(1));
    expect(openChatStream.mock.calls[0][0].session_id).toBeNull();

    await screen.findByText('阶段性答复');
    await user.type(screen.getByRole('textbox', { name: '案情或追问' }), '第二轮追问');
    await user.click(screen.getByRole('button', { name: '发送' }));
    await waitFor(() => expect(openChatStream).toHaveBeenCalledTimes(2));
    expect(openChatStream.mock.calls[1][0].session_id).toBe('sess_20260703_110000_cd02');
  });

  test('从历史抽屉选择另一个会话时载入该会话内容', async () => {
    const user = userEvent.setup();
    const sessionB = {
      session_id: 'sess_20260702_090000_ef03',
      title: '借款纠纷',
      created_at: '2026-07-02T09:00:00+00:00',
      updated_at: '2026-07-02T09:10:00+00:00',
      turn_count: 1,
    };
    fetchSessions.mockResolvedValue({ sessions: [SESSION_A, sessionB] });
    fetchSessionDetail.mockImplementation(async (sessionId) => {
      if (sessionId === SESSION_A.session_id) {
        return SESSION_A_DETAIL;
      }
      return {
        session_id: sessionB.session_id,
        title: sessionB.title,
        updated_at: sessionB.updated_at,
        turn_count: 1,
        messages: [
          { role: 'user', text: '朋友借钱三年不还怎么办？' },
          { role: 'assistant', text: '注意三年诉讼时效。' },
        ],
        materials: { laws: [], web: [], warnings: [] },
        pending_supplement: null,
      };
    });

    render(<App />);
    // 启动时自动恢复最近的 SESSION_A。
    await screen.findByText('公司一直不签劳动合同怎么办？');

    await user.click(screen.getByRole('button', { name: '历史对话' }));
    await user.click(await screen.findByText('借款纠纷'));

    await screen.findByText('朋友借钱三年不还怎么办？');
    expect(screen.getByText('注意三年诉讼时效。')).toBeInTheDocument();
    expect(screen.queryByText('公司一直不签劳动合同怎么办？')).not.toBeInTheDocument();
  });

  test('删除当前会话需要两步确认，删除后回到空白新对话', async () => {
    const user = userEvent.setup();
    fetchSessions.mockResolvedValue({ sessions: [SESSION_A] });
    fetchSessionDetail.mockResolvedValue(SESSION_A_DETAIL);
    deleteSession.mockResolvedValue({ ok: true });

    render(<App />);
    await screen.findByText('公司一直不签劳动合同怎么办？');

    await user.click(screen.getByRole('button', { name: '历史对话' }));
    await user.click(await screen.findByRole('button', { name: '删除' }));
    expect(deleteSession).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: '确认删除' }));

    await waitFor(() => expect(deleteSession).toHaveBeenCalledWith(SESSION_A.session_id));
    await screen.findByText('请描述案情。我会检索本地法条后，给出一般信息参考。');
    expect(screen.queryByText('公司一直不签劳动合同怎么办？')).not.toBeInTheDocument();
  });

  test('恢复以等待补充结束的会话时进入阻塞补充状态', async () => {
    fetchSessions.mockResolvedValue({ sessions: [SESSION_A] });
    fetchSessionDetail.mockResolvedValue({
      ...SESSION_A_DETAIL,
      messages: [
        { role: 'user', text: '公司不签劳动合同' },
        { role: 'assistant', text: '请先补充入职时间和工资信息。' },
      ],
      pending_supplement: {
        reason: '缺少入职时间和工资信息。',
        message: '请先补充入职时间和工资信息。',
        questions: ['入职时间是什么时候？'],
        evidence_gaps: ['工资流水'],
      },
    });

    render(<App />);

    await screen.findByText('需要先补充关键信息');
    expect(screen.getByRole('textbox', { name: '案情或追问' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '逐条补充' })).toBeEnabled();
  });
});
