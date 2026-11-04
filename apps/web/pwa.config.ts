import type { NextConfig } from "next";

export const withPwa = (config: NextConfig): NextConfig => {
  // Static exports need these headers configured at the hosting layer.
  if (config.output === "export") {
    return config;
  }
  return {
    ...config,
    async headers() {
      return [
        ...((await config.headers?.()) ?? []),
        {
          headers: [
            {
              key: "Content-Type",
              value: "application/javascript; charset=utf-8",
            },
            {
              key: "Cache-Control",
              value: "no-cache, no-store, must-revalidate",
            },
            {
              key: "Content-Security-Policy",
              value: "default-src 'self'; script-src 'self'",
            },
          ],
          source: "/sw.js",
        },
      ];
    },
  };
};
