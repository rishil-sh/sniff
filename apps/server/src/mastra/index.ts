import { Mastra } from "@mastra/core";
import { LibSQLStore } from "@mastra/libsql";

import { ENV } from "../env.server";
import { sniffAgent } from "./sniff-agent";

export const mastra = new Mastra({
  agents: { sniffAgent },
  storage: new LibSQLStore({
    id: "sniff-mastra",
    url: ENV.MASTRA_DATABASE_URL,
  }),
});
