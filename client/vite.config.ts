import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 3000,
    proxy: {
      '/stream': {
        target: 'ws://localhost:7878',
        ws: true,
      },
      '/control': {
        target: 'ws://localhost:7878',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
