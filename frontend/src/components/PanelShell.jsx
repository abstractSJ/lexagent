import { Box, Paper, Stack, Typography } from '@mui/material';
import { PANEL_BORDER_COLOR, PANEL_HEADER_BG, PANEL_RADIUS, PANEL_SHADOW } from '../theme.js';

/**
 * 三栏面板的统一外壳。
 *
 * 对话、执行进度、参考资料三块面板此前各自手写 Paper + 头部布局，视觉参数一旦微调就要改三处。
 * 抽出统一外壳后，卡片圆角/阴影/头部结构只维护一份，三栏在任何调整下都保持一致。
 * 外壳只管"壳"：正文区的滚动行为、ref 和事件仍由调用方在 children 里自己控制，
 * 因为三块面板的滚动策略并不相同（聊天区吸底可暂停、进度区强制跟随、资料区自由滚动）。
 *
 * @param {object} props 组件参数。
 * @param {string} [props.component='section'] Paper 渲染的语义标签（section/aside）。
 * @param {string} props.titleId 标题元素 id，同时作为 Paper 的 aria-labelledby 目标。
 * @param {JSX.Element} props.icon 头部图标（本项目内联 SVG 图标组件实例）。
 * @param {{bg: string, fg: string}} props.accent 图标底色与前景色，用于区分三栏身份。
 * @param {string} props.title 面板标题。
 * @param {string} [props.subtitle] 标题下方的说明文字。
 * @param {React.ReactNode} [props.action] 头部右侧的操作区（按钮/计数 Chip）。
 * @param {React.ReactNode} props.children 面板正文（由调用方控制滚动）。
 * @param {React.ReactNode} [props.footer] 面板底部固定区（如聊天输入框）。
 * @returns {JSX.Element} 面板外壳。
 */
export default function PanelShell({
  component = 'section',
  titleId,
  icon,
  accent,
  title,
  subtitle,
  action,
  children,
  footer,
}) {
  return (
    <Paper
      elevation={0}
      component={component}
      aria-labelledby={titleId}
      sx={{
        display: 'grid',
        // footer 存在与否决定两行还是三行；中间行 minmax(0, 1fr) 是滚动的关键，
        // 少了 minmax(0) 时 grid 子项默认 min-height:auto，内容一多整块面板会被撑破。
        gridTemplateRows: footer ? 'auto minmax(0, 1fr) auto' : 'auto minmax(0, 1fr)',
        minHeight: 0,
        border: `1px solid ${PANEL_BORDER_COLOR}`,
        borderRadius: PANEL_RADIUS,
        overflow: 'hidden',
        backgroundColor: '#ffffff',
        boxShadow: PANEL_SHADOW,
      }}
    >
      <Stack
        direction="row"
        spacing={1.5}
        sx={{
          alignItems: 'center',
          px: 2.25,
          py: 1.75,
          borderBottom: '1px solid',
          borderColor: 'divider',
          background: PANEL_HEADER_BG,
        }}
      >
        <Box
          aria-hidden="true"
          sx={{
            width: 36,
            height: 36,
            flexShrink: 0,
            display: 'grid',
            placeItems: 'center',
            borderRadius: '10px',
            backgroundColor: accent.bg,
            color: accent.fg,
            '& svg': { fontSize: 18 },
          }}
        >
          {icon}
        </Box>
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Typography
            id={titleId}
            component="h2"
            sx={{ fontSize: 15.5, fontWeight: 700, lineHeight: 1.3, letterSpacing: '0.01em' }}
          >
            {title}
          </Typography>
          {subtitle && (
            <Typography color="text.secondary" sx={{ mt: 0.25, fontSize: 12.25, lineHeight: 1.45 }}>
              {subtitle}
            </Typography>
          )}
        </Box>
        {action}
      </Stack>

      {children}

      {footer}
    </Paper>
  );
}
