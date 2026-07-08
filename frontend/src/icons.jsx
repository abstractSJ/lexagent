import { SvgIcon } from '@mui/material';

/**
 * 内联 SVG 图标集。
 *
 * 项目刻意不引入 @mui/icons-material：这个依赖体积大且本项目只需要二十个左右的线性图标，
 * 用 24x24 描边路径手工内联即可覆盖，保持 node_modules 和构建产物轻量。
 * 图标风格统一为 2px 圆头描边（feather/lucide 风格），颜色继承 currentColor，
 * 因此在深色头部、浅色面板、彩色事件节点里都能直接复用。
 */

/**
 * 描边图标工厂。
 *
 * @param {string} name 组件 displayName，方便 React DevTools 调试。
 * @param {string[]} paths SVG path 的 d 属性列表。
 * @returns {(props: object) => JSX.Element} 可当作 MUI SvgIcon 使用的图标组件。
 */
function createStrokeIcon(name, paths) {
  function StrokeIcon({ sx, ...rest }) {
    return (
      // SvgIcon 根节点默认 fill: currentColor，会把描边图标填成实心色块，
      // 必须在根上重置为 none，让视觉完全由 path 的 stroke 决定。
      <SvgIcon viewBox="0 0 24 24" {...rest} sx={[{ fill: 'none' }, ...(Array.isArray(sx) ? sx : [sx])]}>
        {paths.map((d) => (
          <path
            key={d}
            d={d}
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ))}
      </SvgIcon>
    );
  }
  StrokeIcon.displayName = name;
  return StrokeIcon;
}

// 天平：品牌徽标与助手头像，是整套 UI 的核心符号。
export const ScaleIcon = createStrokeIcon('ScaleIcon', [
  'm16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z',
  'm2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z',
  'M7 21h10',
  'M12 3v18',
  'M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2',
]);

// 纸飞机：发送按钮。
export const SendIcon = createStrokeIcon('SendIcon', ['M22 2 11 13', 'M22 2 15 22 11 13 2 9 22 2Z']);

// 历史（时钟回退箭头）：历史对话入口。
export const HistoryIcon = createStrokeIcon('HistoryIcon', [
  'M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8',
  'M3 3v5h5',
  'M12 7v5l4 2',
]);

// 闪电：RAG 预热与自修复类事件。
export const BoltIcon = createStrokeIcon('BoltIcon', ['M13 2 3 14h9l-1 8 10-12h-9l1-8z']);

// 心电波形：执行进度与指标类事件。
export const ActivityIcon = createStrokeIcon('ActivityIcon', ['M22 12h-4l-3 9L9 3l-3 9H2']);

// 打开的书：法条资料。
export const BookOpenIcon = createStrokeIcon('BookOpenIcon', [
  'M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z',
  'M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z',
]);

// 地球：公网案例/实务资料。
export const GlobeIcon = createStrokeIcon('GlobeIcon', [
  'M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0Z',
  'M2 12h20',
  'M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z',
]);

// 关闭。
export const CloseIcon = createStrokeIcon('CloseIcon', ['M18 6 6 18', 'm6 6 12 12']);

// 加号：开始新对话。
export const PlusIcon = createStrokeIcon('PlusIcon', ['M12 5v14', 'M5 12h14']);

// 垃圾桶：删除会话。
export const TrashIcon = createStrokeIcon('TrashIcon', [
  'M3 6h18',
  'M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2',
  'M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6',
  'M10 11v6',
  'M14 11v6',
]);

// 下箭头：资料手风琴展开指示。
export const ChevronDownIcon = createStrokeIcon('ChevronDownIcon', ['m6 9 6 6 6-6']);

// 对话气泡：聊天面板标题与等待类事件。
export const ChatIcon = createStrokeIcon('ChatIcon', [
  'M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z',
]);

// 文档：案件状态/补充信息类事件。
export const FileTextIcon = createStrokeIcon('FileTextIcon', [
  'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z',
  'M14 2v6h6',
  'M16 13H8',
  'M16 17H8',
  'M10 9H8',
]);

// 外链：打开公网资料来源。
export const ExternalLinkIcon = createStrokeIcon('ExternalLinkIcon', [
  'M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6',
  'M15 3h6v6',
  'M10 14 21 3',
]);

// 放大镜：检索类事件与资料区空状态。
export const SearchIcon = createStrokeIcon('SearchIcon', [
  'M21 21l-4.35-4.35',
  'M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16z',
]);

// 警告三角：错误与降级提示。
export const AlertTriangleIcon = createStrokeIcon('AlertTriangleIcon', [
  'M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z',
  'M12 9v4',
  'M12 17h.01',
]);

// 对勾圆圈：完成类事件。
export const CheckCircleIcon = createStrokeIcon('CheckCircleIcon', [
  'M22 11.08V12a10 10 0 1 1-5.93-9.14',
  'M22 4 12 14.01l-3 3',
]);

// 星光：分析/记忆类事件。
export const SparklesIcon = createStrokeIcon('SparklesIcon', [
  'M12 3l1.88 5.76a2 2 0 0 0 1.36 1.36L21 12l-5.76 1.88a2 2 0 0 0-1.36 1.36L12 21l-1.88-5.76a2 2 0 0 0-1.36-1.36L3 12l5.76-1.88a2 2 0 0 0 1.36-1.36L12 3z',
]);

// 数据库：本地法条 RAG 检索类事件。
export const DatabaseIcon = createStrokeIcon('DatabaseIcon', [
  'M12 8c4.97 0 9-1.34 9-3s-4.03-3-9-3-9 1.34-9 3 4.03 3 9 3z',
  'M21 12c0 1.66-4.03 3-9 3s-9-1.34-9-3',
  'M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5',
]);
