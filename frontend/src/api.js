/**
 * 前端 API 封装。
 *
 * 所有请求都使用相对路径。开发模式下由 Vite proxy 转发到 FastAPI；生产模式下由
 * 浏览器同源访问 FastAPI。这样能避免在小项目里额外引入 CORS 配置。
 */

/**
 * 获取后端健康状态。
 *
 * @returns {Promise<object>} 服务状态、RAG 预热状态和最近一次预热错误。
 */
export async function fetchHealth() {
  const response = await fetch('/api/health');
  if (!response.ok) {
    throw new Error(`健康检查失败：${response.status}`);
  }
  return response.json();
}

/**
 * 手动预热本地法律 RAG 资源。
 *
 * @returns {Promise<object>} 后端返回的预热结果。
 */
export async function preloadRag() {
  const response = await fetch('/api/preload', { method: 'POST' });
  const data = await readJsonOrText(response);
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || data.message || `预热失败：${response.status}`);
  }
  return data;
}

/**
 * 发起聊天请求，并返回可逐行读取的 NDJSON 响应体。
 *
 * 这里故意不调用 response.json()。原因是 `/api/chat` 是流式 NDJSON，必须边收边解析，
 * 否则右侧执行进度会退化成整轮结束后才一次性显示。
 *
 * @param {object} payload 聊天或补充表单请求体。
 * @returns {Promise<ReadableStream>} 浏览器 fetch 响应体。
 */
export async function openChatStream(payload) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    const data = await readJsonOrText(response);
    throw new Error(data.detail || data.error || data.message || data.text || `请求失败：${response.status}`);
  }

  return response.body;
}

/**
 * 获取历史会话列表。
 *
 * @returns {Promise<{sessions: object[]}>} 按最近更新时间倒序的会话摘要列表。
 */
export async function fetchSessions() {
  const response = await fetch('/api/sessions');
  if (!response.ok) {
    throw new Error(`获取历史会话失败：${response.status}`);
  }
  return response.json();
}

/**
 * 获取单个历史会话的可恢复内容。
 *
 * @param {string} sessionId 会话 ID。
 * @returns {Promise<object>} 公开消息、参考资料和未完成的补充请求。
 */
export async function fetchSessionDetail(sessionId) {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
  const data = await readJsonOrText(response);
  if (!response.ok) {
    throw new Error(data.detail || data.message || `加载会话失败：${response.status}`);
  }
  return data;
}

/**
 * 删除历史会话。
 *
 * @param {string} sessionId 会话 ID。
 * @returns {Promise<object>} 删除结果。
 */
export async function deleteSession(sessionId) {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  const data = await readJsonOrText(response);
  if (!response.ok) {
    throw new Error(data.detail || data.message || `删除会话失败：${response.status}`);
  }
  return data;
}

/**
 * 读取 JSON 响应；如果后端返回纯文本或 HTML 错误页，则包装为 text 字段。
 *
 * @param {Response} response fetch 响应。
 * @returns {Promise<object>} JSON 对象或 `{ text }`。
 */
export async function readJsonOrText(response) {
  const rawText = await response.text();
  if (!rawText) {
    return {};
  }
  try {
    return JSON.parse(rawText);
  } catch {
    return { text: rawText };
  }
}

/**
 * 把未知异常统一转换为可展示的字符串。
 *
 * @param {unknown} error 任意异常对象。
 * @returns {string} 前端可直接展示的错误文本。
 */
export function stringifyError(error) {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}
