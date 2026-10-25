import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const dir = dirname(fileURLToPath(import.meta.url));

/** @type {import('tailwindcss').Config} */
export default {
  content: [`${dir}/index.html`, `${dir}/src/**/*.{ts,tsx}`],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["'Instrument Serif'", "ui-serif", "Georgia", "serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "monospace"],
      },
      colors: {
        ink: { 950: "#09090c", 900: "#0f0f14", 850: "#14141b", 800: "#1a1a23", 700: "#262632", 600: "#3a3a48" },
      },
      keyframes: {
        shimmer: { "0%": { backgroundPosition: "-400px 0" }, "100%": { backgroundPosition: "400px 0" } },
        rise: { "0%": { opacity: 0, transform: "translateY(8px)" }, "100%": { opacity: 1, transform: "none" } },
      },
      animation: { shimmer: "shimmer 1.4s linear infinite", rise: "rise .45s cubic-bezier(.2,.7,.2,1) both" },
    },
  },
  plugins: [],
};
