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
      // Dataset mode: fetch index.json + cells over HTTP from the Rust server
      '/universe': {
        target: 'http://localhost:7878',
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
