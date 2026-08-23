import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/tasks': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('error', (err, _req, res) => {
            console.warn('[proxy] /tasks error:', err.message)
            if (res && 'writeHead' in res && !res.headersSent) {
              (res as import('http').ServerResponse).writeHead(502, { 'Content-Type': 'application/json' })
              ;(res as import('http').ServerResponse).end(JSON.stringify({ detail: 'Backend unavailable' }))
            }
          })
        },
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('error', (err, _req, res) => {
            console.warn('[proxy] /health error:', err.message)
            if (res && 'writeHead' in res && !res.headersSent) {
              (res as import('http').ServerResponse).writeHead(502)
              ;(res as import('http').ServerResponse).end()
            }
          })
        },
      },
      '/readiness': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('error', (err, _req, res) => {
            console.warn('[proxy] /readiness error:', err.message)
            if (res && 'writeHead' in res && !res.headersSent) {
              (res as import('http').ServerResponse).writeHead(502)
              ;(res as import('http').ServerResponse).end()
            }
          })
        },
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        configure: (proxy) => {
          proxy.on('error', (err) => {
            console.warn('[proxy] /ws error:', err.message)
          })
          proxy.on('proxyReqWs', (_proxyReq, _req, socket) => {
            socket.on('error', (err) => {
              console.warn('[proxy] WebSocket socket error:', err.message)
            })
          })
        },
      },
    },
  },
})
