import { Box, Link, Paper, Typography } from '@mui/material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

/**
 * 助手消息 Markdown 渲染器。
 *
 * LLM 最终回答常包含标题、列表、加粗和分隔线；直接按纯文本展示会暴露 `##`、`**` 等标记。
 * react-markdown 默认会转义 HTML，不启用 rehypeRaw，是为了避免把模型输出中的 HTML 当作真实 DOM 执行。
 *
 * @param {object} props 组件参数。
 * @param {string} props.text Markdown 文本。
 * @returns {JSX.Element} 使用 MUI 组件渲染的 Markdown 内容。
 */
export default function MarkdownMessage({ text }) {
  return (
    <Box
      sx={{
        fontSize: 14,
        lineHeight: 1.65,
        '& > :first-of-type': { mt: 0 },
        '& > :last-child': { mb: 0 },
      }}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {text || ''}
      </ReactMarkdown>
    </Box>
  );
}

const markdownComponents = {
  p: ({ children }) => (
    <Typography component="p" sx={{ my: 1, fontSize: 14, lineHeight: 1.7 }}>
      {children}
    </Typography>
  ),
  h1: ({ children }) => (
    <Typography component="h1" variant="h6" sx={{ mt: 1.5, mb: 1, fontWeight: 800 }}>
      {children}
    </Typography>
  ),
  // h2 是律师口吻回答的章节标题（结论/当前关键点/法律风险等），用品牌蓝细左条强化章节切分，
  // 让长回答的骨架在快速滚动时也能被视觉锚定。
  h2: ({ children }) => (
    <Typography
      component="h2"
      sx={{
        mt: 2,
        mb: 0.75,
        pl: 1.25,
        fontSize: 15.5,
        fontWeight: 800,
        lineHeight: 1.4,
        borderLeft: '3px solid #2b4ecb',
      }}
    >
      {children}
    </Typography>
  ),
  h3: ({ children }) => (
    <Typography component="h3" sx={{ mt: 1.25, mb: 0.75, fontSize: 14.5, fontWeight: 800 }}>
      {children}
    </Typography>
  ),
  h4: ({ children }) => (
    <Typography component="h4" sx={{ mt: 1, mb: 0.5, fontSize: 14, fontWeight: 800 }}>
      {children}
    </Typography>
  ),
  ul: ({ children }) => (
    <Box component="ul" sx={{ my: 1, pl: 2.5 }}>
      {children}
    </Box>
  ),
  ol: ({ children }) => (
    <Box component="ol" sx={{ my: 1, pl: 2.5 }}>
      {children}
    </Box>
  ),
  li: ({ children }) => (
    <Typography component="li" sx={{ my: 0.4, fontSize: 14, lineHeight: 1.65 }}>
      {children}
    </Typography>
  ),
  strong: ({ children }) => (
    <Box component="strong" sx={{ fontWeight: 800, color: '#152036' }}>
      {children}
    </Box>
  ),
  a: ({ href, children }) => (
    <Link href={href} target="_blank" rel="noreferrer" underline="hover" sx={{ fontWeight: 600 }}>
      {children}
    </Link>
  ),
  blockquote: ({ children }) => (
    <Paper
      component="blockquote"
      elevation={0}
      sx={{
        m: 0,
        my: 1,
        px: 1.5,
        py: 0.75,
        borderLeft: '3px solid #a9bdf2',
        borderRadius: '0 10px 10px 0',
        backgroundColor: '#f2f6ff',
      }}
    >
      {children}
    </Paper>
  ),
  code: ({ inline, children }) => {
    if (inline) {
      return (
        <Box
          component="code"
          sx={{ px: 0.5, py: 0.1, borderRadius: '6px', backgroundColor: '#edf1fa', color: '#32406b', fontSize: 13 }}
        >
          {children}
        </Box>
      );
    }
    return <code>{children}</code>;
  },
  pre: ({ children }) => (
    <Box
      component="pre"
      sx={{
        my: 1,
        p: 1.25,
        overflowX: 'auto',
        borderRadius: '12px',
        backgroundColor: '#101a33',
        color: '#dfe6f5',
        fontSize: 13,
      }}
    >
      {children}
    </Box>
  ),
  // GFM 表格默认无边框，长表格会糊成一片；这里给最小可读样式并允许横向滚动。
  table: ({ children }) => (
    <Box sx={{ my: 1, overflowX: 'auto' }}>
      <Box
        component="table"
        sx={{
          width: '100%',
          borderCollapse: 'collapse',
          '& th, & td': {
            border: '1px solid #e3e8f3',
            px: 1,
            py: 0.5,
            fontSize: 13,
            textAlign: 'left',
            verticalAlign: 'top',
          },
          '& th': { backgroundColor: '#f4f7fd', fontWeight: 700 },
        }}
      >
        {children}
      </Box>
    </Box>
  ),
  hr: () => <Box component="hr" sx={{ my: 2, border: 0, borderTop: '1px solid #e3e8f3' }} />,
};
