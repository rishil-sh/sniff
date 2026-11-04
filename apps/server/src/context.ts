import type { Context as ApiContext } from "@sniff/api/context";
import type { Context as HonoContext } from "hono";

import { runCheck } from "./check";
import { ENV } from "./env.server";
import { auth, db } from "./services";

export interface CreateContextOptions {
  context: HonoContext;
}

const adminEmails = new Set(
  (ENV.ADMIN_EMAILS ?? "")
    .split(",")
    .map((email) => email.trim().toLowerCase())
    .filter(Boolean)
);

export const createContext = async ({
  context,
}: CreateContextOptions): Promise<ApiContext> => {
  const session = await auth.api.getSession({
    headers: context.req.raw.headers,
  });
  const email = session?.user.email.toLowerCase();
  return {
    db,
    isAdmin: email !== undefined && adminEmails.has(email),
    runCheck,
    session,
  };
};

export type Context = Awaited<ReturnType<typeof createContext>>;
