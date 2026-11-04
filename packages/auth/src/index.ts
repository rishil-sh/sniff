import { drizzleAdapter } from "@better-auth/drizzle-adapter/relations-v2";
import type { Database } from "@sniff/db";
import * as schema from "@sniff/db/schema/auth";
import { betterAuth } from "better-auth";

export interface AuthConfig {
  BETTER_AUTH_URL: string;
  BETTER_AUTH_SECRET: string;
  CORS_ORIGIN: string;
  COOKIE_DOMAIN?: string;
}

export const createAuth = (
  env: AuthConfig,
  database: Database,
  desktopOrigins: readonly string[] = []
) =>
  betterAuth({
    advanced: {
      // Web and API on different hosts: scope the cookie to their parent domain so the web server's session check gets it.
      crossSubDomainCookies: {
        domain: env.COOKIE_DOMAIN,
        enabled: Boolean(env.COOKIE_DOMAIN),
      },
      defaultCookieAttributes: {
        httpOnly: true,
        sameSite: "none",
        secure: true,
      },
    },
    baseURL: env.BETTER_AUTH_URL,
    database: drizzleAdapter(database, {
      provider: "sqlite",
      schema,
    }),
    emailAndPassword: { enabled: true },
    plugins: [],
    secret: env.BETTER_AUTH_SECRET,
    trustedOrigins: [env.CORS_ORIGIN, ...desktopOrigins],
  });

export type Session = ReturnType<typeof createAuth>["$Infer"]["Session"];
