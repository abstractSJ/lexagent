import { createTheme } from '@mui/material/styles';

/**
 * Material UI 主题配置。
 *
 * 这里把原控制台的浅色背景、蓝色主色、大圆角和中文字体栈沉淀为主题 token。
 * 这样做的原因是后续页面组件可以优先使用 MUI 的 `sx` 和主题变量，避免重新堆一份
 * 难维护的全局 CSS。
 */
export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#2563eb',
      dark: '#1d4ed8',
    },
    secondary: {
      main: '#7c3aed',
    },
    success: {
      main: '#15803d',
    },
    error: {
      main: '#b42318',
    },
    info: {
      main: '#0369a1',
    },
    background: {
      default: '#f4f6fb',
      paper: '#ffffff',
    },
    text: {
      primary: '#18202f',
      secondary: '#667085',
    },
    divider: '#d9e0ea',
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      '"Microsoft YaHei"',
      'sans-serif',
    ].join(','),
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 700,
          lineHeight: 1.2,
          whiteSpace: 'nowrap',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
    MuiTextField: {
      defaultProps: {
        variant: 'outlined',
      },
    },
  },
});
