import { memo } from 'react';
import { Box, Button, Chip, CircularProgress, Paper, Stack, Typography } from '@mui/material';
import { colorForStatus } from '../eventFormatters.js';

/**
 * 顶部应用标题和服务状态区。
 *
 * 使用 memo 的原因是最终回答流式期间 status 等 props 保持稳定，头部可跳过高频增量渲染。
 *
 * @param {object} props 组件参数。
 * @param {{text: string, kind: 'ok'|'pending'|'error'}} props.status 当前服务状态。
 * @param {boolean} props.preloadDisabled 是否禁用预热按钮。
 * @param {boolean} props.isPreloading 是否正在预热。
 * @param {() => void} props.onPreload 触发 RAG 预热。
 * @param {() => void} props.onOpenSessions 打开历史对话抽屉。
 * @returns {JSX.Element} 顶部 Header。
 */
const AppHeader = memo(function AppHeader({ status, preloadDisabled, isPreloading, onPreload, onOpenSessions }) {
  return (
    <Paper
      elevation={0}
      sx={{
        p: { xs: 2, md: 2.5 },
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 2,
        backgroundColor: 'rgba(255, 255, 255, 0.88)',
        boxShadow: '0 20px 50px rgba(20, 31, 54, 0.08)',
        backdropFilter: 'blur(12px)',
      }}
    >
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', md: 'minmax(0, 1fr) auto' },
          gap: 2,
          alignItems: { xs: 'stretch', md: 'center' },
          width: '100%',
        }}
      >
        <Box>
          <Typography
            component="p"
            sx={{
              mb: 0.75,
              color: 'primary.main',
              fontSize: 12,
              fontWeight: 800,
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
            }}
          >
            Local Legal Agent
          </Typography>
          <Typography component="h1" variant="h4" sx={{ fontWeight: 900, lineHeight: 1.15 }}>
            法律咨询 Agent
          </Typography>
          <Typography color="text.secondary" sx={{ mt: 1, fontSize: 14 }}>
            输入案情，实时查看案件状态更新、RAG 检索和最终答复。
          </Typography>
        </Box>

        <Stack
          direction="row"
          spacing={1}
          useFlexGap
          sx={{
            alignItems: 'center',
            justifyContent: { xs: 'flex-start', md: 'flex-end' },
            flexWrap: 'wrap',
            ml: { md: 'auto' },
            flexShrink: 0,
          }}
        >
          <Chip
            label={status.text}
            color={colorForStatus(status.kind)}
            variant="outlined"
            size="small"
            sx={{ height: 30, fontWeight: 700 }}
          />
          <Button
            type="button"
            variant="outlined"
            size="small"
            onClick={onOpenSessions}
            sx={{ height: 32, minHeight: 32, minWidth: 82, px: 1.5, alignSelf: 'center' }}
          >
            历史对话
          </Button>
          <Button
            type="button"
            variant="outlined"
            size="small"
            disabled={preloadDisabled}
            onClick={onPreload}
            startIcon={isPreloading ? <CircularProgress size={14} color="inherit" /> : null}
            sx={{ height: 32, minHeight: 32, minWidth: 82, px: 1.5, alignSelf: 'center' }}
          >
            {isPreloading ? '预热中' : '预热 RAG'}
          </Button>
        </Stack>
      </Box>
    </Paper>
  );
});

export default AppHeader;
