import React from 'react';
import { createRoot } from 'react-dom/client';
import { CssBaseline, GlobalStyles, ThemeProvider } from '@mui/material';
import App from './App.jsx';
import { theme } from './theme.js';

/**
 * React 应用入口。
 *
 * ThemeProvider 统一注入 Material UI 主题；CssBaseline 清理浏览器默认样式差异；
 * GlobalStyles 只放真正页面级的东西：背景网格光晕、全局滚动条、选中色和共享动画关键帧。
 * 组件自身的样式仍然放在各组件 `sx` 中，避免全局 CSS 越积越多。
 */
createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <GlobalStyles
        styles={{
          html: { minHeight: '100%' },
          body: {
            minHeight: '100%',
            // 三层低饱和径向光晕（蓝紫 / 蓝 / 金）叠在浅灰底上，形成有空气感的背景。
            // fixed 附着让滚动时光晕保持不动，页面内容像浮在背景之上。
            background: [
              'radial-gradient(1100px 520px at 88% -12%, rgba(79, 70, 229, 0.10), transparent 60%)',
              'radial-gradient(900px 480px at -8% 18%, rgba(37, 99, 235, 0.08), transparent 55%)',
              'radial-gradient(820px 540px at 52% 112%, rgba(183, 121, 31, 0.07), transparent 62%)',
              '#eef2f9',
            ].join(', '),
            backgroundAttachment: 'fixed',
          },
          '#root': { minHeight: '100vh' },
          // 三栏都是独立滚动容器，默认粗滚动条会破坏卡片外观；统一改成细圆角低对比样式。
          '*::-webkit-scrollbar': { width: 8, height: 8 },
          '*::-webkit-scrollbar-thumb': {
            backgroundColor: '#c5cee2',
            borderRadius: 8,
            border: '2px solid transparent',
            backgroundClip: 'content-box',
          },
          '*::-webkit-scrollbar-thumb:hover': { backgroundColor: '#aab6d4' },
          '*::-webkit-scrollbar-track': { background: 'transparent' },
          '*': { scrollbarWidth: 'thin', scrollbarColor: '#c5cee2 transparent' },
          '::selection': { backgroundColor: 'rgba(43, 78, 203, 0.16)' },
          // 消息入场动画：轻微上浮 + 淡入。幅度刻意压小（6px），高频流式追加时不显得跳动。
          '@keyframes legalMsgIn': {
            from: { opacity: 0, transform: 'translateY(6px)' },
            to: { opacity: 1, transform: 'none' },
          },
          // 状态胶囊的呼吸灯：只变不透明度不变尺寸，避免引发相邻元素回流。
          '@keyframes legalPulse': {
            '0%': { opacity: 1 },
            '50%': { opacity: 0.35 },
            '100%': { opacity: 1 },
          },
        }}
      />
      <App />
    </ThemeProvider>
  </React.StrictMode>,
);
