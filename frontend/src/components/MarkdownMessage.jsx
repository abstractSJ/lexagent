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
    <Typography component="p" sx={{ my: 1, fontSize: 14, lineHeight: 1.65 }}>
      {children}
    </Typography>
  ),
  h1: ({ children }) => (
    <Typography component="h1" variant="h6" sx={{ mt: 1.5, mb: 1, fontWeight: 900 }}>
      {children}
    </Typography>
  ),
  h2: ({ children }) => (
    <Typography component="h2" sx={{ mt: 1.5, mb: 0.75, fontSize: 16, fontWeight: 900 }}>
      {children}
    </Typography>
  ),
  h3: ({ children }) => (
    <Typography component="h3" sx={{ mt: 1.25, mb: 0.75, fontSize: 15, fontWeight: 900 }}>
      {children}
    </Typography>
  ),
  h4: ({ children }) => (
    <Typography component="h4" sx={{ mt: 1, mb: 0.5, fontSize: 14, fontWeight: 900 }}>
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
    <Typography component="li" sx={{ my: 0.35, fontSize: 14, lineHeight: 1.6 }}>
      {children}
    </Typography>
  ),
  strong: ({ children }) => (
    <Box component="strong" sx={{ fontWeight: 900 }}>
      {children}
    </Box>
  ),
  a: ({ href, children }) => (
    <Link href={href} target="_blank" rel="noreferrer" underline="hover">
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
        borderLeft: '3px solid #93c5fd',
        backgroundColor: '#eff6ff',
      }}
    >
      {children}
    </Paper>
  ),
  code: ({ inline, children }) => {
    if (inline) {
      return (
        <Box component="code" sx={{ px: 0.5, py: 0.1, borderRadius: 0.75, backgroundColor: '#e2e8f0', fontSize: 13 }}>
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
        borderRadius: 1,
        backgroundColor: '#0f172a',
        color: '#e2e8f0',
        fontSize: 13,
      }}
    >
      {children}
    </Box>
  ),
  hr: () => <Box component="hr" sx={{ my: 2, border: 0, borderTop: '1px solid #d9e0ea' }} />,
};
