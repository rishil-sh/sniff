import { defineConfig } from "oxfmt";
import ultracite from "ultracite/oxfmt";

export default defineConfig({
  ...ultracite,
  // Vendored agent skills: pinned by hash in skills-lock.json, so never reformat them.
  ignorePatterns: [...(ultracite.ignorePatterns ?? []), ".agents/**"],
});
