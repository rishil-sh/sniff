import type { RouterClient } from "@orpc/server";

import { publicProcedure } from "../index";
import { checkRouter } from "./check";

export const appRouter = {
  check: checkRouter,
  healthCheck: publicProcedure.handler(() => "OK"),
};
export type AppRouter = typeof appRouter;
export type AppRouterClient = RouterClient<typeof appRouter>;
