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
import { ChatIcon, CloseIcon, HistoryIcon, PlusIcon, TrashIcon } from '../icons.jsx';

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
      slotProps={{
        paper: {
          sx: {
            width: 344,
            // 抽屉右缘用大圆角 + 大范围柔和投影和页面内容分层，
            // 因此去掉默认的 1px 右描边，避免圆角外露出直线边框。
            borderRight: 'none',
            borderRadius: '0 20px 20px 0',
            boxShadow: '0 24px 64px -16px rgba(15, 26, 62, 0.25)',
            display: 'flex',
            flexDirection: 'column',
          },
        },
      }}
    >
      <Box
        sx={{
          p: 2,
          borderBottom: '1px solid',
          borderColor: 'divider',
          // 与三栏面板头部同款的微渐变底，让抽屉头部和主界面视觉语言一致。
          background: 'linear-gradient(180deg, #fbfcff 0%, #f4f7fd 100%)',
        }}
      >
        <Stack direction="row" spacing={1.25} sx={{ alignItems: 'center' }}>
          <Box
            aria-hidden="true"
            sx={{
              width: 32,
              height: 32,
              flexShrink: 0,
              display: 'grid',
              placeItems: 'center',
              borderRadius: '10px',
              backgroundColor: '#e8edfb',
              color: '#2b4ecb',
            }}
          >
            <HistoryIcon sx={{ fontSize: 16 }} />
          </Box>
          <Typography component="h2" sx={{ flex: 1, minWidth: 0, fontSize: 15.5, fontWeight: 700 }}>
            历史对话
          </Typography>
          <IconButton type="button" aria-label="关闭历史对话" onClick={onClose} sx={{ color: 'text.secondary' }}>
            <CloseIcon sx={{ fontSize: 18 }} />
          </IconButton>
        </Stack>
        <Button
          type="button"
          fullWidth
          variant="contained"
          disabled={busy}
          onClick={onNew}
          startIcon={<PlusIcon sx={{ fontSize: 16 }} />}
          sx={{ mt: 1.5, height: 40 }}
        >
          开始新对话
        </Button>
        {busy && (
          <Typography color="text.secondary" sx={{ mt: 1, fontSize: 12 }}>
            当前咨询处理中，结束后才能切换或删除会话。
          </Typography>
        )}
      </Box>

      <Box sx={{ flex: 1, minHeight: 0, overflowY: 'auto', p: 1 }} aria-label="历史对话列表">
        {sessions.length === 0 ? (
          <Stack spacing={1} sx={{ py: 5, alignItems: 'center' }}>
            <ChatIcon aria-hidden="true" sx={{ fontSize: 30, color: '#c3cede' }} />
            <Typography color="text.secondary" sx={{ fontSize: 13, textAlign: 'center', px: 3 }}>
              还没有历史对话。发送第一条案情后，这里会自动保存会话记录。
            </Typography>
          </Stack>
        ) : (
          <List disablePadding>
            {sessions.map((item) => (
              <ListItem
                key={item.session_id}
                disablePadding
                // 列表项之间用间距而不是分隔线区隔：每项本身是圆角卡片，
                // 通栏 divider 会切断圆角，观感反而更碎。
                sx={{ mb: 0.5 }}
                secondaryAction={
                  <Button
                    type="button"
                    size="small"
                    color="error"
                    variant={confirmingId === item.session_id ? 'contained' : 'text'}
                    disabled={busy}
                    startIcon={<TrashIcon sx={{ fontSize: 14 }} />}
                    onClick={() => handleDeleteClick(item.session_id)}
                    sx={{
                      minWidth: 'auto',
                      px: 1,
                      // 只给 text 态加浅红 hover 底；确认态是 contained error，
                      // 覆盖它的 hover 会削弱“再点一下就真的删除”的警示感。
                      ...(confirmingId === item.session_id
                        ? {}
                        : { '&:hover': { backgroundColor: '#fdecec' } }),
                    }}
                  >
                    {confirmingId === item.session_id ? '确认删除' : '删除'}
                  </Button>
                }
              >
                <ListItemButton
                  selected={item.session_id === activeSessionId}
                  disabled={busy}
                  onClick={() => onSelect(item.session_id)}
                  sx={{
                    // MUI 对带 secondaryAction 的列表项内置了两级选择器的 padding-right: 48px，
                    // 特异性高于 sx 直接写 pr；用 && 提升特异性才能真正给删除按钮（带图标后更宽）留位。
                    '&&': { pr: 13 },
                    borderRadius: '12px',
                    '&.Mui-selected': {
                      backgroundColor: '#e9eefb',
                      '&:hover': { backgroundColor: '#e2e9fa' },
                    },
                  }}
                >
                  <ListItemText
                    primary={
                      <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
                        <Typography
                          component="span"
                          sx={{
                            fontSize: 14,
                            fontWeight: item.session_id === activeSessionId ? 700 : 600,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            // nowrap 文本在 flex 里默认按完整内容占位；必须放开最小宽度，
                            // 长标题才会真正触发省略号，而不是把轮次徽章挤到删除按钮底下。
                            minWidth: 0,
                            flexShrink: 1,
                          }}
                        >
                          {item.title || '未命名对话'}
                        </Typography>
                        <Chip
                          size="small"
                          label={`${item.turn_count || 0} 轮`}
                          sx={{
                            height: 20,
                            fontSize: 11,
                            border: 'none',
                            backgroundColor: '#eef2fb',
                            color: '#5c6a84',
                            flexShrink: 0,
                          }}
                        />
                      </Stack>
                    }
                    secondary={formatSessionTime(item.updated_at)}
                    slotProps={{
                      // ListItemText 默认用 inline span 包裹 primary；行内盒不约束子元素宽度，
                      // 标题的 ellipsis 永远不会触发，轮次徽章会被推到删除按钮底下。
                      // 显式改成 block 并放开最小宽度后，内部 flex 行才能按容器宽度收缩截断。
                      primary: { sx: { display: 'block', minWidth: 0 } },
                      secondary: { sx: { fontSize: 12 } },
                    }}
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
