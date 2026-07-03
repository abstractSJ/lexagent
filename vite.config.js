import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * Vite 前端构建配置。
 *
 * 生产构建输出到 FastAPI 已挂载的 web_app/static 目录；这样浏览器访问后端 `/` 时，
 * 仍然走同源静态资源，不需要额外部署前端服务。base 固定为 `/static/`，是为了让
 * 构建后的 JS/CSS 资源路径与 FastAPI 的 StaticFiles 挂载点保持一致。
 */
export default defineConfig({
  plugins: [react()],
  base: '/static/',
  build: {
    outDir: 'web_app/static',
    emptyOutDir: true,
  },
  server: {
    host: '127.0.0.1',
    proxy: {
      // 开发模式下由 Vite 转发 API，生产模式下浏览器直接同源访问 FastAPI。
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});
