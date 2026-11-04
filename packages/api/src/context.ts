import type { Session } from "@sniff/auth";
import type { Database } from "@sniff/db";

export type Context = {
  session: Session | null;
  db: Database;
};
