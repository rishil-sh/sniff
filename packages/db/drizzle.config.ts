import { existsSync } from "node:fs";

import { defineConfig } from "drizzle-kit";

// The database URL lives in the server's env file (pnpm bootstrap creates it).
const serverEnv = "../../apps/server/.env";
if (existsSync(serverEnv)) {
  process.loadEnvFile(serverEnv);
}

export default defineConfig({
  dbCredentials: {
    url: process.env.DATABASE_URL ?? "",
  },
  dialect: "turso",
  out: "./src/migrations",
  schema: "./src/schema/index.ts",
});
