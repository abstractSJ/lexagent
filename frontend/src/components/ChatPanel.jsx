import { useCallback, useEffect, useRef } from 'react';
import { Box, Button, Paper, Stack, Typography } from '@mui/material';
import ChatInput from './ChatInput.jsx';
import MessageList from './MessageList.jsx';
import PanelShell from './PanelShell.jsx';
import SupplementDialog from './SupplementDialog.jsx';
import { AlertTriangleIcon, ChatIcon, SparklesIcon } from '../icons.jsx';

/**
 * 中间对话面板。
 *
 * @param {object} props 组件参数。
 * @param {object[]} props.messages 聊天消息列表。
 * @param {object|null} props.supplement 当前可补充的问题数据。
 * @param {boolean} props.supplementBlocking 是否为阻塞性补充。
 * @param {boolean} props.supplementDialogOpen 补充弹窗是否打开。
 * @param {boolean} props.inputDisabled 主输入是否禁用。
 * @param {boolean} props.isSending 当前是否正在发送。
 * @param {boolean} props.supplementSubmitting 当前是否正在提交补充表单。
 * @param {(text: string) => Promise<void>|void} props.onSubmit 普通问题提交。
 * @param {() => void} props.onOpenSupplement 打开补充弹窗。
 * @param {() => void} props.onCloseSupplement 关闭补充弹窗。
 * @param {(payload: object) => Promise<boolean|void>|boolean|void} props.onSupplementContinue 补充表单提交。
 * @returns {JSX.Element} 对话面板。
 */
export default function ChatPanel({
  messages,
  supplement,
  supplementBlocking,
  supplementDialogOpen,
  inputDisabled,
  isSending,
  supplementSubmitting,
  onSubmit,
  onOpenSupplement,
  onCloseSupplement,
  onSupplementContinue,
}) {
  const scrollRef = useRef(null);
  // 默认吸附底部。用滚动事件时的位置（而不是内容更新后的位置）判断用户是否主动上翻，
  // 原因是流式增量到达后 scrollHeight 已经变大，事后判断会把“本来在底部”误判成“已离开底部”。
  const stickToBottomRef = useRef(true);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (el) {
      stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    }
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickToBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, supplement]);

  return (
    <PanelShell
      component="section"
      titleId="chatTitle"
      icon={<ChatIcon />}
      accent={{ bg: '#e8edfb', fg: '#2b4ecb' }}
      title="对话"
      subtitle="公开聊天历史只展示用户输入和最终助手答复。"
      footer={
        <>
          <ChatInput disabled={inputDisabled} isSending={isSending} onSubmit={onSubmit} />
          <SupplementDialog
            supplement={supplement}
            open={supplementDialogOpen}
            blocking={supplementBlocking}
            disabled={supplementSubmitting}
            onClose={onCloseSupplement}
            onContinue={onSupplementContinue}
          />
        </>
      }
    >
      <Box
        ref={scrollRef}
        onScroll={handleScroll}
        sx={{
          minHeight: 0,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
        }}
        aria-label="聊天内容和补充信息"
      >
        <MessageList messages={messages} scrollable={false} />

        {supplement && (
          <SupplementCallout blocking={supplementBlocking} supplement={supplement} onOpen={onOpenSupplement} />
        )}
      </Box>
    </PanelShell>
  );
}

/**
 * 补充信息入口卡片。
 *
 * 只显示一个紧凑按钮，不把完整表单直接铺在聊天区里。这样既保留补充入口，又不会遮挡底部输入框。
 * 阻塞性补充用琥珀色（必须处理才能继续），非阻塞建议用品牌蓝（可选优化），色彩语义与事件区一致。
 *
 * @param {object} props 组件参数。
 * @param {boolean} props.blocking 是否为阻塞性补充。
 * @param {object} props.supplement 补充问题数据。
 * @param {() => void} props.onOpen 打开补充弹窗。
 * @returns {JSX.Element} 补充入口。
 */
function SupplementCallout({ blocking, supplement, onOpen }) {
  const questionCount = Array.isArray(supplement.questions) ? supplement.questions.length : 0;
  const evidenceCount = Array.isArray(supplement.evidence_gaps) ? supplement.evidence_gaps.length : 0;

  return (
    <Paper
      elevation={0}
      sx={{
        mx: 2,
        mb: 1.5,
        p: 1.5,
        display: 'flex',
        alignItems: 'center',
        gap: 1.25,
        // 聊天滚动容器是纵向 flex；不禁止收缩的话，消息一多此卡片会被压扁。
        flexShrink: 0,
        border: '1px solid',
        borderColor: blocking ? '#f0d49a' : '#c7d7f5',
        background: blocking
          ? 'linear-gradient(135deg, #fff9ee 0%, #fdf3dd 100%)'
          : 'linear-gradient(135deg, #f2f6ff 0%, #e9f0ff 100%)',
        borderRadius: '14px',
        animation: 'legalMsgIn 0.3s ease both',
      }}
    >
      <Box
        aria-hidden="true"
        sx={{
          width: 34,
          height: 34,
          flexShrink: 0,
          display: 'grid',
          placeItems: 'center',
          borderRadius: '10px',
          backgroundColor: blocking ? '#fbead0' : '#dfe8fc',
          color: blocking ? '#b45309' : '#2b4ecb',
        }}
      >
        {blocking ? <AlertTriangleIcon sx={{ fontSize: 17 }} /> : <SparklesIcon sx={{ fontSize: 17 }} />}
      </Box>
      <Stack spacing={0.25} sx={{ flex: 1, minWidth: 0 }}>
        <Typography sx={{ fontSize: 14, fontWeight: 700, color: blocking ? '#8a4c07' : '#1f3aa8' }}>
          {blocking ? '需要先补充关键信息' : '可补充关键信息'}
        </Typography>
        <Typography sx={{ fontSize: 12.5, lineHeight: 1.5, color: 'text.secondary' }}>
          {supplement.message || supplement.reason || '补充后可以让后续分析更准确。'}（问题 {questionCount} 个，材料{' '}
          {evidenceCount} 项）
        </Typography>
      </Stack>
      <Button
        type="button"
        variant="contained"
        size="small"
        color={blocking ? 'warning' : 'primary'}
        onClick={onOpen}
        sx={{ whiteSpace: 'nowrap', flexShrink: 0, borderRadius: '10px' }}
      >
        逐条补充
      </Button>
    </Paper>
  );
}
