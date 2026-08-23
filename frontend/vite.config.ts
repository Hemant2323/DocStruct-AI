import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      // Vite resolves a leading "/" against the project root, so this needs no Node
      // path helpers — which keeps @types/node out of the dependency list.
      "@": "/src",
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    // Requests to /api are forwarded to the FastAPI backend during development.
    // This keeps the browser talking to a single origin, so no CORS preflight is
    // involved in dev and the frontend needs no absolute backend URL.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
