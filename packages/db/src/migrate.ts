import { migrate } from "drizzle-orm/libsql/migrator";

import type { Database } from "./index";

/** Applies pending migrations from `folder` (the server runs this on boot in Docker; locally `pnpm db:migrate`). */
export const runMigrations = async (
  db: Database,
  folder: string
): Promise<void> => {
  await migrate(db, { migrationsFolder: folder });
};
