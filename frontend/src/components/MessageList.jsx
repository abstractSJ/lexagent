import { memo, useEffect, useRef } from 'react';
import { Box, Paper, Stack, Typography } from '@mui/material';
import MarkdownMessage from './MarkdownMessage.jsx';

/**
 * 聊天消息列表。
 *
 * 公开聊天区只展示用户输入、助手最终答复和可读错误。工具调用、检索 query 等内部过程继续放到
 * 右侧事件区，避免把业务调试信息混进长期对话历史。
 *
 * @param {object} props 组件参数。
 * @param {{id: string, role: 'user'|'assistant'|'error', text: string, streaming?: boolean}[]} props.messages 消息列表。
 * @param {boolean} [props.scrollable=true] 是否由本组件自己承担滚动。
 * @returns {JSX.Element} 消息列表。
 */
export default function MessageList({ messages, scrollable = true }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollable && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, scrollable]);

  return (
    <Stack
      ref={scrollRef}
      spacing={1.25}
      sx={{
        minHeight: 0,
        overflowY: scrollable ? 'auto' : 'visible',
        p: 2,
        alignItems: 'flex-start',
        flexShrink: 0,
      }}
      aria-live="polite"
      aria-label="聊天消息列表"
    >
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
    </Stack>
  );
}

/**
 * 单条消息气泡。
 *
 * 使用 memo 的原因是流式输出期间只有最后一条消息对象在变化；没有 memo 时，每个增量都会让
 * 整条历史里的 Markdown 重新解析渲染，长会话下打字机效果会越来越卡。
 *
 * @param {object} props 组件参数。
 * @param {{role: 'user'|'assistant'|'error', text: string, streaming?: boolean}} props.message 消息数据。
 * @returns {JSX.Element} 消息气泡。
 */
const MessageBubble = memo(function MessageBubble({ message }) {
  const isUser = message.role === 'user';
  const isError = message.role === 'error';

  return (
    <Box
      sx={{
        display: 'flex',
        width: '100%',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
      }}
    >
      <Paper
        elevation={0}
        sx={{
          maxWidth: isError ? '100%' : { xs: '100%', md: '86%' },
          width: isError ? '100%' : 'auto',
          px: 1.5,
          py: 1,
          border: '1px solid',
          borderColor: isError ? '#fecdd3' : isUser ? '#d7ebff' : '#e5e7eb',
          backgroundColor: isError ? '#fff1f3' : isUser ? '#f1f8ff' : '#f8fafc',
          color: isError ? 'error.main' : 'text.primary',
          borderRadius: 1.5,
          overflowWrap: 'anywhere',
          wordBreak: 'break-word',
        }}
      >
        {isUser || isError ? (
          <Typography sx={{ fontSize: 14, lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>{message.text}</Typography>
        ) : (
          <>
            <MarkdownMessage text={message.text} />
            {message.streaming && <StreamingCursor />}
          </>
        )}
      </Paper>
    </Box>
  );
});

/**
 * 流式输出时的打字机光标。
 *
 * @returns {JSX.Element} 闪烁竖条。
 */
function StreamingCursor() {
  return (
    <Box
      component="span"
      aria-hidden="true"
      sx={{
        display: 'inline-block',
        width: '2px',
        height: '1em',
        ml: 0.25,
        verticalAlign: 'text-bottom',
        backgroundColor: 'primary.main',
        animation: 'legalStreamingBlink 1s step-end infinite',
        '@keyframes legalStreamingBlink': {
          '50%': { opacity: 0 },
        },
      }}
    />
  );
}
