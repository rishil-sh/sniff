import { fileURLToPath } from "node:url";

import { initOpenNextCloudflareForDev } from "@opennextjs/cloudflare";
import type { NextConfig } from "next";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants";

import { withPwa } from "./pwa.config";

const nextConfig: NextConfig = {
  // Self-contained server for the Docker image (apps/web/Dockerfile); trace from the monorepo root so the
  // workspace packages (@sniff/ui, @sniff/api) are included.
  output: "standalone",
  outputFileTracingRoot: fileURLToPath(new URL("../../", import.meta.url)),
  reactCompiler: true,
  typedRoutes: true,
};

// Cloudflare bindings for `next dev` only; starting workerd during `next build` (e.g. in Docker) is unnecessary.
const config = (phase: string): NextConfig => {
  if (phase === PHASE_DEVELOPMENT_SERVER) {
    initOpenNextCloudflareForDev();
  }
  return withPwa(nextConfig);
};

export default config;
