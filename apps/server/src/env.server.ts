import { z } from "zod";

// Bun loads apps/server/.env in development; Docker passes real environment variables. Documented in .env.example.
// Empty values (`KEY=` in .env) count as unset.
const optional = z.preprocess(
  (value) => (value === "" ? undefined : value),
  z.string().optional()
);
const withDefault = (fallback: string) =>
  z.preprocess(
    (value) => (value === "" || value === undefined ? fallback : value),
    z.string()
  );

const schema = z.object({
  ADMIN_EMAILS: optional,
  BETTER_AUTH_SECRET: z
    .string()
    .min(32, "BETTER_AUTH_SECRET must be at least 32 characters"),
  BETTER_AUTH_URL: z.url(),
  // Parent domain shared by web and API hosts in production (e.g. brewlogic.tech), so the web host sees the session.
  COOKIE_DOMAIN: optional,
  CORS_ORIGIN: z.url(),
  DATABASE_URL: z.string().min(1),
  DETECTOR_URL: withDefault("http://localhost:7860").pipe(z.url()),
  MASTRA_DATABASE_URL: withDefault("file:../../.data/mastra.db"),
  MIGRATIONS_DIR: optional,
  MODEL_API_KEY: optional,
  MUSE_MODEL: withDefault("muse-wonder-1.0"),
  NODE_ENV: z
    .enum(["development", "production", "test"])
    .default("development"),
  WHATSAPP_ACCESS_TOKEN: optional,
  WHATSAPP_APP_SECRET: optional,
  WHATSAPP_PHONE_NUMBER_ID: optional,
  WHATSAPP_VERIFY_TOKEN: optional,
});

const parsed = schema.safeParse(process.env);
if (!parsed.success) {
  throw new Error(
    `Invalid server environment (see apps/server/.env.example):\n${z.prettifyError(parsed.error)}`
  );
}

export const ENV = parsed.data;
