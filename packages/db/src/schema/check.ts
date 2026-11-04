import { sql } from "drizzle-orm";
import {
  index,
  integer,
  real,
  sqliteTable,
  text,
} from "drizzle-orm/sqlite-core";

import { user } from "./auth";

/** One analysed input (web upload or WhatsApp message) and the reply we sent. Media itself is not stored. */
export const check = sqliteTable(
  "check",
  {
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .default(sql`(cast(unixepoch('subsecond') * 1000 as integer))`)
      .notNull(),
    filename: text("filename"),
    id: text("id").primaryKey(),
    modality: text("modality").notNull(),
    pAi: real("p_ai"),
    reply: text("reply").notNull(),
    source: text("source", { enum: ["web", "whatsapp"] }).notNull(),
    textPreview: text("text_preview"),
    userId: text("user_id").references(() => user.id, { onDelete: "set null" }),
    verdict: text("verdict").notNull(),
    verdictJson: text("verdict_json").notNull(),
    waFrom: text("wa_from"),
  },
  (table) => [
    index("check_userId_idx").on(table.userId),
    index("check_createdAt_idx").on(table.createdAt),
  ]
);
