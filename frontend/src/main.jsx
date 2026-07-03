import React from 'react';
import { createRoot } from 'react-dom/client';
import { CssBaseline, GlobalStyles, ThemeProvider } from '@mui/material';
import App from './App.jsx';
import { theme } from './theme.js';

/**
 * React 应用入口。
 *
 * ThemeProvider 统一注入 Material UI 主题；CssBaseline 清理浏览器默认样式差异；
 * GlobalStyles 只放极少量页面级样式，具体组件样式仍尽量放在组件 `sx` 中。
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
            background:
              'radial-gradient(circle at top left, #eaf1ff 0, #f4f6fb 38%, #f8fafc 100%)',
          },
          '#root': { minHeight: '100vh' },
        }}
      />
      <App />
    </ThemeProvider>
  </React.StrictMode>,
);
