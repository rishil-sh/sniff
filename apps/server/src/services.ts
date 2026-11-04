import { createAuth } from "@sniff/auth";
import { createDb } from "@sniff/db";

import { ENV } from "./env.server";

export const db = createDb(ENV);
export const auth = createAuth(ENV, db);
