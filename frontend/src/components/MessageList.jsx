import { memo, useEffect, useRef } from 'react';
import { Box, Stack, Typography } from '@mui/material';
import MarkdownMessage from './MarkdownMessage.jsx';
import { AlertTriangleIcon, ScaleIcon } from '../icons.jsx';
import { BRAND_GRADIENT, BRAND_GRADIENT_SOFT } from '../theme.js';

/**
 * 聊天消息列表。
 *
 * 公开聊天区只展示用户输入、助手最终答复和可读错误。工具调用、检索 query 等内部过程继续放到
 * 事件进度区，避免把业务调试信息混进长期对话历史。
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
      spacing={1.75}
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
 * 视觉规则：助手消息带品牌头像 + 白底卡片气泡；用户消息右对齐、品牌渐变底白字；
 * 错误消息通栏红色软底并配警告图标，让失败一眼可辨。
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

  if (isError) {
    return (
      <Stack
        direction="row"
        spacing={1.25}
        sx={{
          width: '100%',
          px: 1.75,
          py: 1.25,
          alignItems: 'flex-start',
          border: '1px solid #f3c6c2',
          backgroundColor: '#fdf3f2',
          color: 'error.main',
          borderRadius: '14px',
          animation: 'legalMsgIn 0.3s ease both',
        }}
      >
        <AlertTriangleIcon sx={{ fontSize: 18, mt: 0.25, flexShrink: 0 }} />
        <Typography sx={{ fontSize: 14, lineHeight: 1.55, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
          {message.text}
        </Typography>
      </Stack>
    );
  }

  return (
    <Box
      sx={{
        display: 'flex',
        width: '100%',
        gap: 1.25,
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        animation: 'legalMsgIn 0.3s ease both',
      }}
    >
      {!isUser && <AssistantAvatar />}
      <Box
        sx={{
          maxWidth: { xs: '92%', md: '80%' },
          px: 1.75,
          py: 1.25,
          overflowWrap: 'anywhere',
          wordBreak: 'break-word',
          ...(isUser
            ? {
                // 用户气泡走品牌渐变：和主按钮同源，让"我发出的内容"与"我按下的按钮"在
                // 视觉上属于同一主体色系。
                background: BRAND_GRADIENT_SOFT,
                color: '#ffffff',
                borderRadius: '16px 16px 4px 16px',
                boxShadow: '0 8px 20px -10px rgba(43, 78, 203, 0.55)',
              }
            : {
                backgroundColor: '#ffffff',
                border: '1px solid #e6ebf6',
                borderRadius: '4px 16px 16px 16px',
                boxShadow: '0 2px 10px -4px rgba(21, 32, 54, 0.08)',
              }),
        }}
      >
        {isUser ? (
          <Typography sx={{ fontSize: 14, lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>{message.text}</Typography>
        ) : (
          <>
            <MarkdownMessage text={message.text} />
            {message.streaming && <StreamingCursor />}
          </>
        )}
      </Box>
    </Box>
  );
});

/**
 * 助手品牌头像。
 *
 * 深蓝渐变底 + 白色天平，与头部徽标同一符号体系；用户消息不配头像，
 * 依靠右对齐 + 渐变底就足够区分角色，还能给长文本留出更大宽度。
 *
 * @returns {JSX.Element} 头像方块。
 */
function AssistantAvatar() {
  return (
    <Box
      aria-hidden="true"
      sx={{
        width: 32,
        height: 32,
        mt: 0.25,
        flexShrink: 0,
        display: 'grid',
        placeItems: 'center',
        borderRadius: '10px',
        background: BRAND_GRADIENT,
        color: '#ffffff',
        boxShadow: '0 6px 14px -6px rgba(22, 41, 95, 0.55)',
      }}
    >
      <ScaleIcon sx={{ fontSize: 17 }} />
    </Box>
  );
}

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
