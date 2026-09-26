import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy optional — we call VITE_API_BASE_URL directly so errors stay visible.
  },
});
