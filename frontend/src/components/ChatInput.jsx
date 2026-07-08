import { memo, useState } from 'react';
import { Box, Button, CircularProgress, Stack, TextField, Typography } from '@mui/material';
import { SendIcon } from '../icons.jsx';

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
        p: { xs: 1.75, md: 2 },
        borderTop: '1px solid',
        borderColor: 'divider',
        background: 'linear-gradient(180deg, #fafbff 0%, #f4f7fd 100%)',
      }}
    >
      <TextField
        label="案情或追问"
        value={text}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        multiline
        minRows={3}
        maxRows={8}
        placeholder="例如：我在公司干了两年，没有签劳动合同，现在被辞退了，公司也没给补偿。"
        slotProps={{ htmlInput: { 'aria-label': '案情或追问' } }}
      />
      <Stack
        direction="row"
        spacing={1.5}
        useFlexGap
        sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}
      >
        <Stack
          direction="row"
          spacing={0.75}
          sx={{ alignItems: 'center', color: 'text.secondary', display: { xs: 'none', sm: 'flex' } }}
        >
          <Kbd>Enter</Kbd>
          <HintText>发送</HintText>
          <HintText>·</HintText>
          <Kbd>Shift+Enter</Kbd>
          <HintText>换行</HintText>
          <HintText>·</HintText>
          <Kbd>Ctrl/⌘+Enter</Kbd>
          <HintText>发送</HintText>
        </Stack>
        <Button
          type="submit"
          variant="contained"
          disabled={disabled || !text.trim()}
          startIcon={isSending ? <CircularProgress size={14} color="inherit" /> : null}
          endIcon={isSending ? null : <SendIcon sx={{ fontSize: 15 }} />}
          sx={{ minWidth: 108, minHeight: 42, borderRadius: '12px', px: 2.25, ml: 'auto' }}
        >
          {isSending ? '处理中' : '发送'}
        </Button>
      </Stack>
    </Box>
  );
});

export default ChatInput;

/**
 * 键位提示胶囊。
 *
 * 把纯文字快捷键说明拆成 kbd 小块 + 说明词，扫一眼即可分清"哪个是按键、哪个是效果"。
 *
 * @param {object} props 组件参数。
 * @param {React.ReactNode} props.children 按键文本。
 * @returns {JSX.Element} 键位胶囊。
 */
function Kbd({ children }) {
  return (
    <Box
      component="kbd"
      sx={{
        px: 0.6,
        py: 0.1,
        borderRadius: '6px',
        border: '1px solid #d5ddee',
        backgroundColor: '#ffffff',
        boxShadow: '0 1px 0 #d5ddee',
        fontSize: 11,
        fontFamily: 'inherit',
        lineHeight: 1.6,
        whiteSpace: 'nowrap',
      }}
    >
      {children}
    </Box>
  );
}

/**
 * 键位提示的说明文字。
 *
 * @param {object} props 组件参数。
 * @param {React.ReactNode} props.children 说明词。
 * @returns {JSX.Element} 说明文字。
 */
function HintText({ children }) {
  return <Typography sx={{ fontSize: 12, whiteSpace: 'nowrap' }}>{children}</Typography>;
}
