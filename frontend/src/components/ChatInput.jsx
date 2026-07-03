import { memo, useState } from 'react';
import { Box, Button, Stack, TextField, Typography } from '@mui/material';

/**
 * 聊天输入框。
 *
 * Enter 默认发送，Shift+Enter 换行，Ctrl/Cmd+Enter 也发送。这里保留原生前端的快捷键习惯，
 * 原因是法律案情常需要多行描述，不能把所有 Enter 都简单理解为换行。
 *
 * 使用 memo 的原因是流式回答期间 props 保持稳定，输入框子树可以完全跳过高频增量渲染。
 *
 * @param {object} props 组件参数。
 * @param {boolean} props.disabled 是否禁用主输入。
 * @param {boolean} props.isSending 是否正在发送。
 * @param {(text: string) => Promise<boolean|void>|boolean|void} props.onSubmit 提交回调；返回 false 表示提交失败。
 * @returns {JSX.Element} 输入表单。
 */
const ChatInput = memo(function ChatInput({ disabled, isSending, onSubmit }) {
  const [text, setText] = useState('');

  const submit = async () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) {
      return;
    }
    const succeeded = await onSubmit(trimmed);
    // 只有请求成功进入正常流后才清空输入；失败保留原文，方便用户直接重试或修改。
    if (succeeded !== false) {
      setText('');
    }
  };

  const handleKeyDown = (event) => {
    const wantsSend = event.key === 'Enter' && (event.ctrlKey || event.metaKey || !event.shiftKey);
    if (!wantsSend) {
      return;
    }
    event.preventDefault();
    void submit();
  };

  return (
    <Box
      component="form"
      autoComplete="off"
      onSubmit={(event) => {
        event.preventDefault();
        void submit();
      }}
      sx={{
        display: 'grid',
        gap: 1.25,
        p: { xs: 2, md: 2.5 },
        borderTop: '1px solid',
        borderColor: 'divider',
        backgroundColor: 'rgba(248, 250, 252, 0.92)',
      }}
    >
      <TextField
        label="案情或追问"
        value={text}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        multiline
        minRows={4}
        maxRows={8}
        placeholder="例如：我在公司干了两年，没有签劳动合同，现在被辞退了，公司也没给补偿。"
        slotProps={{ htmlInput: { 'aria-label': '案情或追问' } }}
      />
      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        spacing={1.5}
        sx={{ alignItems: { xs: 'stretch', sm: 'center' }, justifyContent: 'space-between' }}
      >
        <Typography color="text.secondary" sx={{ fontSize: 12 }}>
          Enter 发送，Shift + Enter 换行，Ctrl/Cmd + Enter 也可发送。
        </Typography>
        <Button type="submit" variant="contained" disabled={disabled || !text.trim()} sx={{ minWidth: 96, minHeight: 40 }}>
          {isSending ? '处理中' : '发送'}
        </Button>
      </Stack>
    </Box>
  );
});

export default ChatInput;
