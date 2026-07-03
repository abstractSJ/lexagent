/**
 * NDJSON 事件流读取工具。
 *
 * 后端 `/api/chat` 每行输出一个 JSON 对象，并以 `done` 作为成功结束标记。网络 chunk
 * 可能只包含半行 JSON，也可能一次包含多行；因此必须保留尾部 buffer，不能简单地按 chunk
 * 直接 JSON.parse。
 */

/**
 * 逐行读取 NDJSON 流，并把每个解析出的事件交给回调处理。
 *
 * @param {ReadableStream} body fetch 返回的响应 body。
 * @param {(item: object) => void} onItem 每个事件对象的处理函数。
 * @returns {Promise<void>} 流结束后完成。
 */
export async function readNdjsonStream(body, onItem) {
  const reader = body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let receivedDone = false;

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }

      // 一个网络 chunk 可能包含半行或多行 JSON；未完成的最后一段必须留到下一次拼接。
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const item = parseNdjsonLine(line);
        if (!item) {
          continue;
        }
        onItem(item);
        if (item.type === 'done') {
          receivedDone = true;
        }
      }
    }

    const remaining = buffer.trim();
    if (remaining) {
      const item = parseNdjsonLine(remaining);
      if (item) {
        onItem(item);
        if (item.type === 'done') {
          receivedDone = true;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  if (!receivedDone) {
    throw new Error('事件流提前结束，未收到 done 结束事件。');
  }
}

/**
 * 解析单行 NDJSON。
 *
 * @param {string} line 单行 JSON 字符串。
 * @returns {object|null} 空行返回 null，否则返回解析后的事件对象。
 */
function parseNdjsonLine(line) {
  const trimmed = line.trim();
  if (!trimmed) {
    return null;
  }
  try {
    return JSON.parse(trimmed);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`事件解析失败：${message}`);
  }
}
