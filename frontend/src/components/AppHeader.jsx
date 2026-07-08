import { memo } from 'react';
import { Box, Button, CircularProgress, Paper, Stack, Typography } from '@mui/material';
import { BoltIcon, HistoryIcon, ScaleIcon } from '../icons.jsx';
import { BRAND_GRADIENT, GOLD_GRADIENT, PANEL_RADIUS } from '../theme.js';

// 状态圆点配色：深蓝渐变底上 MUI 的语义色（success.main 等）偏暗、对比度不足，
// 这里换成一组更亮的点缀色，保证 8px 小圆点在深色背景上依然一眼可辨。
const STATUS_DOT_COLOR = {
  ok: '#4ade80',
  pending: '#fbbf24',
  error: '#f87171',
};

// 未知状态的兜底圆点色：中性灰，避免后端扩展新状态时圆点凭空消失。
const STATUS_DOT_FALLBACK = '#94a3b8';

// 深色 hero 上的玻璃态按钮共用样式：半透明白描边 + hover 提亮。
// 抽成模块级常量的原因是两个按钮必须保持完全一致的外观，各写一份后续微调容易改漏。
const GLASS_BUTTON_SX = {
  height: 34,
  borderRadius: '10px',
  px: 1.5,
  color: '#fff',
  borderColor: 'rgba(255, 255, 255, 0.32)',
  '&:hover': {
    borderColor: 'rgba(255, 255, 255, 0.6)',
    backgroundColor: 'rgba(255, 255, 255, 0.10)',
  },
  // 深色底上 MUI 默认禁用灰几乎不可见，显式压成半透明白让"禁用"状态可读。
  '&.Mui-disabled': {
    color: 'rgba(255, 255, 255, 0.4)',
    borderColor: 'rgba(255, 255, 255, 0.15)',
  },
};

/**
 * 顶部应用标题和服务状态区（深海军蓝 hero 头部）。
 *
 * 品牌底色 + 金色天平徽标构成整页的视觉锚点；右侧状态胶囊和玻璃态按钮浮在渐变上，
 * 与浅色三栏面板形成层次对比。
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
  const dotColor = STATUS_DOT_COLOR[status.kind] || STATUS_DOT_FALLBACK;

  return (
    <Paper
      elevation={0}
      sx={{
        position: 'relative',
        // 水印图标故意越界到卡片外，必须裁掉溢出部分才能保持圆角轮廓干净。
        overflow: 'hidden',
        borderRadius: PANEL_RADIUS,
        background: BRAND_GRADIENT,
        color: '#fff',
        border: '1px solid rgba(255, 255, 255, 0.06)',
        boxShadow: '0 18px 44px -18px rgba(22, 41, 95, 0.55)',
      }}
    >
      {/* 装饰层：两道白色径向光晕 + 大号天平水印。全部 aria-hidden 且不响应鼠标，
          只负责让大面积深蓝底产生光影层次，不参与布局和交互。 */}
      <Box
        aria-hidden="true"
        sx={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          background: 'radial-gradient(600px 200px at 80% -40%, rgba(255, 255, 255, 0.14), transparent 60%)',
        }}
      />
      <Box
        aria-hidden="true"
        sx={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          background: 'radial-gradient(460px 240px at -6% 130%, rgba(255, 255, 255, 0.08), transparent 62%)',
        }}
      />
      <ScaleIcon
        aria-hidden="true"
        sx={{
          position: 'absolute',
          right: -24,
          top: -34,
          fontSize: 190,
          opacity: 0.07,
          transform: 'rotate(-8deg)',
          pointerEvents: 'none',
        }}
      />

      {/* 内容层：zIndex 抬到装饰层之上；大屏两列（标题区 / 操作区），小屏纵向堆叠。 */}
      <Box
        sx={{
          position: 'relative',
          zIndex: 1,
          p: { xs: 2, md: 2.5 },
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', md: 'minmax(0, 1fr) auto' },
          gap: 2,
          alignItems: { xs: 'stretch', md: 'center' },
        }}
      >
        <Stack direction="row" spacing={1.75} sx={{ alignItems: 'center' }}>
          {/* 金色徽标：全页唯一的大面积金色元素，和深蓝底形成"法律 + 权威"的品牌组合。 */}
          <Box
            aria-hidden="true"
            sx={{
              width: 46,
              height: 46,
              flexShrink: 0,
              display: 'grid',
              placeItems: 'center',
              borderRadius: '13px',
              background: GOLD_GRADIENT,
              color: '#fff',
              boxShadow: '0 8px 20px -8px rgba(183, 121, 31, 0.65)',
            }}
          >
            <ScaleIcon sx={{ fontSize: 24 }} />
          </Box>
          <Box sx={{ minWidth: 0 }}>
            <Typography
              component="p"
              sx={{
                fontSize: 10.5,
                fontWeight: 700,
                letterSpacing: '0.2em',
                color: 'rgba(199, 213, 255, 0.85)',
                textTransform: 'uppercase',
              }}
            >
              LOCAL LEGAL AGENT
            </Typography>
            <Typography component="h1" variant="h5" sx={{ fontWeight: 800, color: '#fff', lineHeight: 1.2 }}>
              法律咨询 Agent
            </Typography>
            <Typography sx={{ mt: 0.5, fontSize: 12.5, color: 'rgba(214, 225, 255, 0.72)' }}>
              输入案情，实时查看案件状态更新、RAG 检索和最终答复。
            </Typography>
          </Box>
        </Stack>

        <Stack
          direction="row"
          useFlexGap
          sx={{
            gap: 1,
            flexWrap: 'wrap',
            alignItems: 'center',
            justifyContent: { xs: 'flex-start', md: 'flex-end' },
            flexShrink: 0,
          }}
        >
          {/* 状态胶囊：毛玻璃底 + 彩色圆点。pending 时圆点用全局 legalPulse 呼吸动画
              （关键帧注册在 main.jsx），只变透明度不变尺寸，避免相邻元素回流。 */}
          <Box
            role="status"
            sx={{
              height: 32,
              px: 1.5,
              display: 'inline-flex',
              alignItems: 'center',
              gap: 1,
              borderRadius: 999,
              background: 'rgba(255, 255, 255, 0.10)',
              border: '1px solid rgba(255, 255, 255, 0.22)',
              backdropFilter: 'blur(8px)',
            }}
          >
            <Box
              aria-hidden="true"
              sx={{
                width: 8,
                height: 8,
                flexShrink: 0,
                borderRadius: '50%',
                backgroundColor: dotColor,
                ...(status.kind === 'pending' && { animation: 'legalPulse 2s ease-in-out infinite' }),
              }}
            />
            <Typography component="span" sx={{ fontSize: 12.5, fontWeight: 600, color: '#fff', whiteSpace: 'nowrap' }}>
              {status.text}
            </Typography>
          </Box>
          <Button
            type="button"
            variant="outlined"
            size="small"
            onClick={onOpenSessions}
            startIcon={<HistoryIcon sx={{ fontSize: 15 }} />}
            sx={GLASS_BUTTON_SX}
          >
            历史对话
          </Button>
          <Button
            type="button"
            variant="outlined"
            size="small"
            disabled={preloadDisabled}
            onClick={onPreload}
            startIcon={isPreloading ? <CircularProgress size={14} color="inherit" /> : <BoltIcon sx={{ fontSize: 15 }} />}
            sx={GLASS_BUTTON_SX}
          >
            {isPreloading ? '预热中' : '预热 RAG'}
          </Button>
        </Stack>
      </Box>
    </Paper>
  );
});

export default AppHeader;
