import { createHmac, timingSafeEqual } from "node:crypto";

import { log } from "evlog";
import { z } from "zod";

import { runCheck } from "./check";
import { ENV } from "./env.server";

const GRAPH_API_URL = "https://graph.facebook.com/v26.0";
// Ported from logicsense apps/server/src/whatsapp/adapter.ts: WhatsApp serves media from these hosts only, and the
// access token goes nowhere else.
const MEDIA_HOSTS = ["fbsbx.com", "fbcdn.net"];
const MEDIA_LIMIT_BYTES = 25 * 1024 * 1024;
const MEDIA_TIMEOUT_MS = 30_000;
const MEDIA_REDIRECTS = 5;
const SEEN_LIMIT = 1000;

const HELP_TEXT =
  "👋 I'm *Sniff*. Send me a photo, a voice note or a paragraph of English text and I'll tell you if AI made it.";
const ERROR_TEXT =
  "😕 Something went wrong while checking that. Please try again in a minute.";

const mediaSchema = z.looseObject({
  id: z.string(),
  mime_type: z.string().optional(),
});
const messageSchema = z.looseObject({
  audio: mediaSchema.optional(),
  document: mediaSchema.extend({ filename: z.string().optional() }).optional(),
  from: z.string(),
  id: z.string(),
  image: mediaSchema.optional(),
  text: z.looseObject({ body: z.string() }).optional(),
  type: z.string(),
  video: mediaSchema.optional(),
});
const webhookSchema = z.looseObject({
  entry: z
    .array(
      z.looseObject({
        changes: z.array(
          z.looseObject({
            value: z.looseObject({
              messages: z.array(messageSchema).optional(),
            }),
          })
        ),
      })
    )
    .default([]),
});
const mediaInfoSchema = z.looseObject({ url: z.string() });

export type InboundMessage = z.infer<typeof messageSchema>;

export const isValidSignature = (
  body: string,
  header: string | null,
  secret: string
): boolean => {
  if (!header?.startsWith("sha256=")) {
    return false;
  }
  const expected = Buffer.from(
    `sha256=${createHmac("sha256", secret).update(body).digest("hex")}`
  );
  const given = Buffer.from(header);
  return expected.length === given.length && timingSafeEqual(expected, given);
};

/** Every inbound message in a webhook body; status updates and other changes are ignored. */
export const extractMessages = (body: string): InboundMessage[] => {
  const parsed = webhookSchema.safeParse(JSON.parse(body));
  if (!parsed.success) {
    return [];
  }
  return parsed.data.entry.flatMap((entry) =>
    entry.changes.flatMap((change) => change.value.messages ?? [])
  );
};

// ponytail: in-memory dedupe of Meta's retries, single process only; use a DB unique key if the server scales out.
const seen = new Set<string>();
const firstTime = (id: string): boolean => {
  if (seen.has(id)) {
    return false;
  }
  seen.add(id);
  if (seen.size > SEEN_LIMIT) {
    seen.delete(seen.values().next().value ?? id);
  }
  return true;
};

type GraphPayload =
  | {
      context: { message_id: string };
      text: { body: string };
      to: string;
      type: "text";
    }
  | { message_id: string; status: "read"; typing_indicator: { type: "text" } };

const graph = async (body: GraphPayload): Promise<void> => {
  const response = await fetch(
    `${GRAPH_API_URL}/${ENV.WHATSAPP_PHONE_NUMBER_ID}/messages`,
    {
      body: JSON.stringify({ messaging_product: "whatsapp", ...body }),
      headers: {
        authorization: `Bearer ${ENV.WHATSAPP_ACCESS_TOKEN}`,
        "content-type": "application/json",
      },
      method: "POST",
    }
  );
  if (!response.ok) {
    throw new Error(
      `WhatsApp send: HTTP ${response.status} ${await response.text()}`
    );
  }
};

const sendText = (to: string, replyTo: string, text: string) =>
  graph({
    context: { message_id: replyTo },
    text: { body: text },
    to,
    type: "text",
  });

// Marks the message read and shows "typing…" for up to 25 s while the detector runs.
const showTyping = (messageId: string) =>
  graph({
    message_id: messageId,
    status: "read",
    typing_indicator: { type: "text" },
  });

const isMediaHost = (url: URL): boolean =>
  url.protocol === "https:" &&
  (url.origin === new URL(GRAPH_API_URL).origin ||
    MEDIA_HOSTS.some(
      (host) => url.hostname === host || url.hostname.endsWith(`.${host}`)
    ));

/** Ported from logicsense TenantWhatsAppAdapter.downloadMedia: host allowlist at every redirect, timeout, size cap. */
const downloadMedia = async (mediaId: string): Promise<Uint8Array> => {
  const signal = AbortSignal.timeout(MEDIA_TIMEOUT_MS);
  const headers = { authorization: `Bearer ${ENV.WHATSAPP_ACCESS_TOKEN}` };
  const info = await fetch(`${GRAPH_API_URL}/${mediaId}`, { headers, signal });
  if (!info.ok) {
    throw new Error(`WhatsApp media ${mediaId}: HTTP ${info.status}`);
  }
  let target = new URL(mediaInfoSchema.parse(await info.json()).url);

  for (let hop = 0; hop <= MEDIA_REDIRECTS; hop += 1) {
    if (!isMediaHost(target)) {
      throw new Error(
        `Refusing to send the WhatsApp token to ${target.hostname}.`
      );
    }
    // oxlint-disable-next-line no-await-in-loop
    const response = await fetch(target, {
      headers,
      redirect: "manual",
      signal,
    });
    const location = response.headers.get("location");
    if (response.status >= 300 && response.status < 400 && location) {
      target = new URL(location, target);
      continue;
    }
    if (!response.ok) {
      throw new Error(`WhatsApp media ${mediaId}: HTTP ${response.status}`);
    }
    if (Number(response.headers.get("content-length")) > MEDIA_LIMIT_BYTES) {
      throw new Error(`WhatsApp media ${mediaId} is over 25 MB.`);
    }
    // oxlint-disable-next-line no-await-in-loop
    const data = new Uint8Array(await response.arrayBuffer());
    if (data.byteLength > MEDIA_LIMIT_BYTES) {
      throw new Error(`WhatsApp media ${mediaId} is over 25 MB.`);
    }
    return data;
  }
  throw new Error(`WhatsApp media ${mediaId}: too many redirects.`);
};

/**
 * WhatsApp voice notes are Ogg/Opus, which the detector's router doesn't accept (wav/mp3/m4a/flac). Transcode to
 * WAV here so apps/detector stays an exact copy of ai-detection-suite. Needs ffmpeg on PATH (or FFMPEG).
 */
const oggToWav = async (bytes: Uint8Array): Promise<Uint8Array> => {
  const ffmpeg = Bun.spawn(
    [
      process.env.FFMPEG ?? "ffmpeg",
      "-loglevel",
      "error",
      "-i",
      "pipe:0",
      "-f",
      "wav",
      "pipe:1",
    ],
    {
      stderr: "pipe",
      stdin: bytes,
      stdout: "pipe",
    }
  );
  const [wav, stderr, code] = await Promise.all([
    new Response(ffmpeg.stdout).arrayBuffer(),
    new Response(ffmpeg.stderr).text(),
    ffmpeg.exited,
  ]);
  if (code !== 0) {
    throw new Error(`ffmpeg exited ${code}: ${stderr}`);
  }
  return new Uint8Array(wav);
};

const EXTENSIONS = new Map([
  ["audio/aac", "m4a"],
  ["audio/mp4", "m4a"],
  ["audio/mpeg", "mp3"],
  ["audio/wav", "wav"],
  ["image/jpeg", "jpg"],
  ["image/png", "png"],
  ["image/webp", "webp"],
  ["video/mp4", "mp4"],
]);

const fileFor = async (message: InboundMessage) => {
  const media =
    message.image ?? message.audio ?? message.video ?? message.document;
  if (!media) {
    return null;
  }
  const mimeType = media.mime_type?.split(";")[0]?.trim() ?? "";
  const bytes = await downloadMedia(media.id);
  if (mimeType === "audio/ogg") {
    return {
      bytes: await oggToWav(bytes),
      filename: "voice-note.wav",
      mimeType: "audio/wav",
    };
  }
  const filename =
    message.document?.filename ??
    `${message.type}.${EXTENSIONS.get(mimeType) ?? "bin"}`;
  return { bytes, filename, mimeType };
};

const handleMessage = async (message: InboundMessage): Promise<void> => {
  const { from, id } = message;
  try {
    const file = await fileFor(message);
    const text = message.text?.body;
    if (!(file || text)) {
      await sendText(from, id, HELP_TEXT);
      return;
    }
    try {
      await showTyping(id);
    } catch (error) {
      log.warn({ action: "whatsapp.typing_failed", error: String(error) });
    }
    const result = await runCheck({
      file: file ?? undefined,
      source: "whatsapp",
      text,
      waFrom: from,
    });
    await sendText(from, id, result.reply);
  } catch (error) {
    log.error({
      action: "whatsapp.message_failed",
      error: String(error),
      messageId: id,
    });
    try {
      await sendText(from, id, ERROR_TEXT);
    } catch {
      // Nothing left to tell the user; the failure is already logged above.
    }
  }
};

const verifyChallenge = (url: URL): Response => {
  const mode = url.searchParams.get("hub.mode");
  const token = url.searchParams.get("hub.verify_token");
  const challenge = url.searchParams.get("hub.challenge");
  const expected = ENV.WHATSAPP_VERIFY_TOKEN;
  if (mode === "subscribe" && expected && token === expected && challenge) {
    return new Response(challenge);
  }
  return new Response("Verification failed", { status: 403 });
};

export const handleWhatsAppWebhook = async (
  request: Request
): Promise<Response> => {
  if (
    !(
      ENV.WHATSAPP_ACCESS_TOKEN &&
      ENV.WHATSAPP_PHONE_NUMBER_ID &&
      ENV.WHATSAPP_APP_SECRET
    )
  ) {
    return new Response("WhatsApp is not configured", { status: 503 });
  }
  if (request.method === "GET") {
    return verifyChallenge(new URL(request.url));
  }
  const body = await request.text();
  if (
    !isValidSignature(
      body,
      request.headers.get("x-hub-signature-256"),
      ENV.WHATSAPP_APP_SECRET
    )
  ) {
    log.warn({ action: "whatsapp.webhook_rejected" });
    return new Response("Invalid signature", { status: 401 });
  }
  // Answer Meta right away (it retries slow webhooks); the detector can take a while.
  for (const message of extractMessages(body)) {
    if (firstTime(message.id)) {
      void handleMessage(message);
    }
  }
  return new Response("OK");
};
