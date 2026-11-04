import { defineConfig } from "oxlint";
import antiSlop from "ultracite/oxlint/anti-slop";
import core from "ultracite/oxlint/core";
import next from "ultracite/oxlint/next";
import react from "ultracite/oxlint/react";
import shadcn from "ultracite/oxlint/shadcn";

export default defineConfig({
  env: core.env,
  extends: [core, react, next, antiSlop, shadcn],
  ignorePatterns: core.ignorePatterns,
  jsPlugins: shadcn.jsPlugins,
  overrides: [
    {
      // The design system itself is allowed to style its primitives.
      files: ["packages/ui/src/components/**"],
      rules: {
        "shadcn/no-arbitrary-values": "off",
        "shadcn/no-restyle": "off",
        "shadcn/require-static-classes": "off",
      },
    },
  ],
  // Oxlint doesn't merge `settings` from extended configs, so these live here.
  settings: {
    shadcn: { ui: "@sniff/ui/components" },
  },
});
