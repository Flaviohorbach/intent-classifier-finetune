import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// On GitHub Pages the site lives under /intent-classifier-finetune/.
export default defineConfig({
  base: process.env.PAGES_BASE ?? "/",
  plugins: [react(), tailwindcss()],
  worker: { format: "es" },
  test: { include: ["tests/**/*.test.ts"], environment: "node" },
});
