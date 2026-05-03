import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/lib/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#08142B",
        ocean: "#0E7490",
        cyan: "#22D3EE",
        steel: "#1E293B",
        mist: "#E2E8F0",
        aurora: "#6366F1"
      },
      boxShadow: {
        glow: "0 12px 48px rgba(14, 116, 144, 0.28)",
        glass: "0 10px 30px rgba(8, 20, 43, 0.22)"
      },
      backgroundImage: {
        mesh: "radial-gradient(circle at 15% 20%, rgba(34, 211, 238, 0.18) 0%, transparent 40%), radial-gradient(circle at 80% 10%, rgba(99, 102, 241, 0.16) 0%, transparent 35%), radial-gradient(circle at 60% 80%, rgba(14, 116, 144, 0.16) 0%, transparent 45%)"
      }
    }
  },
  plugins: []
};

export default config;
