import type { Session } from "@sniff/auth";
import type { Database } from "@sniff/db";

import type { RunCheck } from "./verdict";

export interface Context {
  session: Session | null;
  db: Database;
  /** Email is in ADMIN_EMAILS: may see WhatsApp checks in history. */
  isAdmin: boolean;
  runCheck: RunCheck;
}
