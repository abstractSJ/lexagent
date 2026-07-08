import { createTheme } from '@mui/material/styles';

/**
 * 品牌视觉 token。
 *
 * 页面上所有面板、气泡、按钮的渐变、阴影和圆角都从这里取值。集中成常量导出的原因是
 * 三栏面板、头部 hero、聊天气泡分散在多个组件文件里，如果各自手写色值，很容易在后续
 * 微调时改漏一处，出现"面板圆角不一致 / 蓝色深浅不一"这类破坏整体感的问题。
 */

// 深海军蓝主渐变：用于头部 hero 和助手头像，是整个界面的品牌底色。
export const BRAND_GRADIENT = 'linear-gradient(135deg, #16295f 0%, #1e3a8a 45%, #3a53c5 100%)';

// 较亮的蓝紫渐变：用于用户气泡和主按钮。比 BRAND_GRADIENT 亮一档，
// 保证在浅色背景上作为"可点击/强调"元素时有足够的视觉权重。
export const BRAND_GRADIENT_SOFT = 'linear-gradient(135deg, #2b4ecb 0%, #4f46e5 100%)';

// 主按钮 hover 时的渐变：整体加深一档，配合阴影变化形成"按下去之前先亮起来"的反馈。
export const BRAND_GRADIENT_HOVER = 'linear-gradient(135deg, #2646bd 0%, #4338d6 100%)';

// 金色徽标渐变：法律行业的经典点缀色，只用于品牌徽标和资料区强调，避免大面积使用显得廉价。
export const GOLD_GRADIENT = 'linear-gradient(135deg, #f2c76d 0%, #d89a2e 55%, #b7791f 100%)';

// 面板卡片统一外观：大圆角 + 细边框 + 双层阴影（近处 1px 描边感 + 远处大范围柔和投影）。
export const PANEL_RADIUS = '18px';
export const PANEL_BORDER_COLOR = '#e3e8f3';
export const PANEL_SHADOW =
  '0 1px 2px rgba(16, 24, 40, 0.04), 0 18px 44px -20px rgba(30, 58, 138, 0.16)';

// 面板头部的微渐变底色：和纯白正文区拉开一点层次，又不至于变成明显色块。
export const PANEL_HEADER_BG = 'linear-gradient(180deg, #fbfcff 0%, #f5f7fd 100%)';

// 键盘焦点环：输入框聚焦和按钮 focus-visible 共用，保证键盘可达性反馈一致。
export const FOCUS_RING = '0 0 0 3px rgba(43, 78, 203, 0.16)';

/**
 * Material UI 主题配置。
 *
 * palette 以深蓝为主、金色为辅；typography 收敛此前"到处 900"的字重，改用 700/800 分层，
 * 因为中文字体（微软雅黑等）没有 900 字重，统一压到真实存在的档位可以避免不同平台渲染不一致。
 * 组件级覆写只做全局共性（按钮、输入框、弹窗圆角等），布局细节仍留在各组件 `sx` 中。
 */
export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#2b4ecb',
      dark: '#1f3aa8',
      light: '#6f88ea',
      contrastText: '#ffffff',
    },
    secondary: {
      // 金色作为辅助色：Chip/强调元素可以直接用 color="secondary" 取到品牌金。
      main: '#b7791f',
      dark: '#96610f',
      light: '#d89a2e',
      contrastText: '#ffffff',
    },
    success: {
      main: '#0f8a5f',
      dark: '#0b6b4a',
    },
    warning: {
      main: '#b45309',
    },
    error: {
      main: '#b42318',
      dark: '#8f1c13',
    },
    info: {
      main: '#0369a1',
    },
    background: {
      default: '#eef2f9',
      paper: '#ffffff',
    },
    text: {
      primary: '#152036',
      secondary: '#5c6a84',
    },
    divider: '#e3e8f3',
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      '"PingFang SC"',
      '"Hiragino Sans GB"',
      '"Microsoft YaHei UI"',
      '"Microsoft YaHei"',
      '"Noto Sans SC"',
      'sans-serif',
    ].join(','),
    h4: { fontWeight: 800, letterSpacing: '-0.5px' },
    h5: { fontWeight: 800, letterSpacing: '-0.3px' },
    h6: { fontWeight: 700 },
    button: { fontWeight: 600 },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          // 抗锯齿让深色渐变上的白色小字（头部副标题、状态胶囊）边缘更干净。
          WebkitFontSmoothing: 'antialiased',
          MozOsxFontSmoothing: 'grayscale',
        },
      },
    },
    MuiButton: {
      defaultProps: {
        // 全局关闭按钮默认投影，投影统一由下方 containedPrimary 的品牌阴影控制，
        // 避免 MUI 默认灰阴影和品牌蓝阴影混用。
        disableElevation: true,
      },
      styleOverrides: {
        root: {
          borderRadius: 10,
          textTransform: 'none',
          fontWeight: 600,
          lineHeight: 1.2,
          whiteSpace: 'nowrap',
          transition: 'all 0.18s ease',
          '&:focus-visible': { boxShadow: FOCUS_RING },
        },
        containedPrimary: {
          // 主按钮统一走品牌渐变而不是纯色，是整套 UI 最重要的品牌识别点之一。
          background: BRAND_GRADIENT_SOFT,
          boxShadow: '0 6px 16px -6px rgba(43, 78, 203, 0.45)',
          '&:hover': {
            background: BRAND_GRADIENT_HOVER,
            boxShadow: '0 10px 22px -8px rgba(43, 78, 203, 0.55)',
          },
          // 渐变是 background-image，MUI 的禁用态只改 background-color 盖不住它，
          // 必须显式把渐变替换成灰底，否则禁用按钮看起来仍然可点。
          '&.Mui-disabled': {
            background: '#c8d0e2',
            color: '#ffffff',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 600,
          borderRadius: 8,
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
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          backgroundColor: '#ffffff',
          transition: 'box-shadow 0.18s ease',
          '& .MuiOutlinedInput-notchedOutline': { borderColor: '#dbe2ef' },
          '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: '#b9c6e4' },
          // 聚焦时叠加柔和外圈而不是只加粗边框：多行案情输入是页面核心操作，
          // 需要比默认更明显、但不刺眼的聚焦反馈。
          '&.Mui-focused': { boxShadow: FOCUS_RING },
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderColor: '#2b4ecb',
            borderWidth: 1.5,
          },
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          borderRadius: 20,
          boxShadow: '0 24px 64px -16px rgba(15, 26, 62, 0.35)',
        },
      },
    },
    MuiAccordion: {
      styleOverrides: {
        root: {
          // 去掉 MUI Accordion 自带的顶部分隔线伪元素，资料卡的边界完全交给自己的 border。
          '&:before': { display: 'none' },
        },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 10,
        },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: '#1c2946',
          fontSize: 12,
          borderRadius: 8,
          padding: '6px 10px',
        },
      },
    },
  },
});
