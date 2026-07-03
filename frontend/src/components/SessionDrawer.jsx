import { memo, useState } from 'react';
import {
  Box,
  Button,
  Chip,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Stack,
  Typography,
} from '@mui/material';

/**
 * 左侧历史对话抽屉。
 *
 * 用 Drawer 而不是常驻第四栏，是因为大屏已经是“进度 / 对话 / 资料”三栏布局，
 * 再挤一栏会压缩对话区；历史列表属于低频入口，点开选完即收起更合适。
 *
 * 使用 memo 的原因是最终回答流式期间本组件 props 保持稳定，可跳过高频增量渲染。
 *
 * @param {object} props 组件参数。
 * @param {boolean} props.open 抽屉是否打开。
 * @param {object[]} props.sessions 历史会话摘要列表。
 * @param {string|null} props.activeSessionId 当前激活的会话 ID。
 * @param {boolean} props.busy 当前是否有请求进行中；进行中时禁止切换、新建和删除。
 * @param {() => void} props.onClose 关闭抽屉。
 * @param {(sessionId: string) => void} props.onSelect 选择历史会话。
 * @param {() => void} props.onNew 开始新对话。
 * @param {(sessionId: string) => void} props.onDelete 删除历史会话。
 * @returns {JSX.Element} 历史对话抽屉。
 */
const SessionDrawer = memo(function SessionDrawer({
  open,
  sessions,
  activeSessionId,
  busy,
  onClose,
  onSelect,
  onNew,
  onDelete,
}) {
  // 删除采用两步确认：第一次点删除只把该项标记为待确认，再点一次才真正删除。
  // 这样不引入额外确认弹窗组件，也能避免误触直接删掉整段咨询历史。
  const [confirmingId, setConfirmingId] = useState(null);

  const handleDeleteClick = (sessionId) => {
    if (confirmingId === sessionId) {
      setConfirmingId(null);
      onDelete(sessionId);
      return;
    }
    setConfirmingId(sessionId);
  };

  return (
    <Drawer
      anchor="left"
      open={open}
      onClose={onClose}
      slotProps={{ paper: { sx: { width: 320, display: 'flex', flexDirection: 'column' } } }}
    >
      <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Stack direction="row" spacing={1} sx={{ alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography component="h2" variant="h6" sx={{ fontWeight: 900 }}>
            历史对话
          </Typography>
          <IconButton type="button" aria-label="关闭历史对话" onClick={onClose} sx={{ color: 'text.secondary' }}>
            ×
          </IconButton>
        </Stack>
        <Button
          type="button"
          fullWidth
          variant="contained"
          disabled={busy}
          onClick={onNew}
          sx={{ mt: 1.5 }}
        >
          开始新对话
        </Button>
        {busy && (
          <Typography color="text.secondary" sx={{ mt: 1, fontSize: 12 }}>
            当前咨询处理中，结束后才能切换或删除会话。
          </Typography>
        )}
      </Box>

      <Box sx={{ flex: 1, minHeight: 0, overflowY: 'auto' }} aria-label="历史对话列表">
        {sessions.length === 0 ? (
          <Typography color="text.secondary" sx={{ p: 2, fontSize: 13 }}>
            还没有历史对话。发送第一条案情后，这里会自动保存会话记录。
          </Typography>
        ) : (
          <List disablePadding>
            {sessions.map((item) => (
              <ListItem
                key={item.session_id}
                disablePadding
                divider
                secondaryAction={
                  <Button
                    type="button"
                    size="small"
                    color="error"
                    variant={confirmingId === item.session_id ? 'contained' : 'text'}
                    disabled={busy}
                    onClick={() => handleDeleteClick(item.session_id)}
                    sx={{ minWidth: 'auto', px: 1 }}
                  >
                    {confirmingId === item.session_id ? '确认删除' : '删除'}
                  </Button>
                }
              >
                <ListItemButton
                  selected={item.session_id === activeSessionId}
                  disabled={busy}
                  onClick={() => onSelect(item.session_id)}
                  sx={{ pr: 11 }}
                >
                  <ListItemText
                    primary={
                      <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
                        <Typography
                          component="span"
                          sx={{
                            fontSize: 14,
                            fontWeight: item.session_id === activeSessionId ? 900 : 600,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {item.title || '未命名对话'}
                        </Typography>
                        <Chip size="small" variant="outlined" label={`${item.turn_count || 0} 轮`} sx={{ height: 20, fontSize: 11 }} />
                      </Stack>
                    }
                    secondary={formatSessionTime(item.updated_at)}
                    slotProps={{ secondary: { sx: { fontSize: 12 } } }}
                  />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        )}
      </Box>
    </Drawer>
  );
});

export default SessionDrawer;

/**
 * 把后端 UTC 时间格式化为本地可读时间。
 *
 * @param {string|undefined} value ISO-8601 时间字符串。
 * @returns {string} 本地时间文本；无法解析时返回原文或空串。
 */
function formatSessionTime(value) {
  if (!value) {
    return '';
  }
  const time = new Date(value);
  if (Number.isNaN(time.getTime())) {
    return String(value);
  }
  return time.toLocaleString();
}
