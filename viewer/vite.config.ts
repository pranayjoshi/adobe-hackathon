import { fileURLToPath } from "node:url";
import autoprefixer from "autoprefixer";
import tailwindcss from "tailwindcss";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = process.env.CHAMELEON_API ?? "http://127.0.0.1:8765";

export default defineConfig({
  root: fileURLToPath(new URL(".", import.meta.url)),
  plugins: [react()],
  css: {
    postcss: {
      plugins: [tailwindcss({ config: fileURLToPath(new URL("./tailwind.config.js", import.meta.url)) }), autoprefixer()],
    },
  },
  server: {
    port: Number(process.env.PORT ?? 5180),
    strictPort: true,
    proxy: { "/api": api, "/renders": api },
  },
});
