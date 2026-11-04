import { ORPCError } from "@orpc/server";
import { z } from "zod";

import { protectedProcedure } from "../index";
import type { Verdict } from "../verdict";

// The detector's own caps: images 20 MB, audio 32 MB, video 100 MB.
const MAX_FILE_BYTES = 100 * 1024 * 1024;
const MAX_TEXT_CHARS = 100_000;
const LIST_LIMIT = 50;

const createInput = z
  .object({
    file: z
      .instanceof(File)
      .refine((value) => value.size <= MAX_FILE_BYTES, "Files up to 100 MB")
      .optional(),
    text: z.string().trim().max(MAX_TEXT_CHARS).optional(),
  })
  .refine((value) => value.file || value.text, "Send a file or some text.");

// WhatsApp checks belong to strangers' phone numbers, so only admins see them.
const visibleTo = (userId: string, isAdmin: boolean) =>
  isAdmin ? { OR: [{ userId }, { source: "whatsapp" as const }] } : { userId };

export const checkRouter = {
  create: protectedProcedure
    .input(createInput)
    .handler(async ({ context, input }) => {
      const file = input.file
        ? {
            bytes: new Uint8Array(await input.file.arrayBuffer()),
            filename: input.file.name,
            mimeType: input.file.type || undefined,
          }
        : undefined;
      return await context.runCheck({
        file,
        source: "web",
        text: input.text,
        userId: context.session.user.id,
      });
    }),

  get: protectedProcedure
    .input(z.object({ id: z.string() }))
    .handler(async ({ context, input }) => {
      const row = await context.db.query.check.findFirst({
        where: {
          AND: [
            { id: input.id },
            visibleTo(context.session.user.id, context.isAdmin),
          ],
        },
      });
      if (!row) {
        throw new ORPCError("NOT_FOUND");
      }
      // SAFETY: verdictJson is only ever written by runCheck from the detector's Verdict response.
      return { ...row, verdict: JSON.parse(row.verdictJson) as Verdict };
    }),

  list: protectedProcedure.handler(
    async ({ context }) =>
      await context.db.query.check.findMany({
        columns: { verdictJson: false },
        limit: LIST_LIMIT,
        orderBy: { createdAt: "desc" },
        where: visibleTo(context.session.user.id, context.isAdmin),
      })
  ),
};
