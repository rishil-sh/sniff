import type { CheckInput, CheckResult, Verdict } from "@sniff/api/verdict";
import { check } from "@sniff/db/schema/check";
import { log } from "evlog";

import { ENV } from "./env.server";
import { sniffAgent } from "./mastra/sniff-agent";
import { db } from "./services";

// First request loads the torch models; later ones take seconds.
const DETECTOR_TIMEOUT_MS = 5 * 60_000;
const MAX_WHY_CHARS = 400;
const PREVIEW_CHARS = 200;

const analyze = async ({ file, text }: CheckInput): Promise<Verdict> => {
  const form = new FormData();
  if (file) {
    form.append(
      "file",
      new Blob([file.bytes], { type: file.mimeType }),
      file.filename
    );
  }
  if (text) {
    form.append("text", text);
  }
  // Muse runs in the Mastra agent below, so the detector returns its deterministic template report.
  form.append("send_to_muse", "false");

  const response = await fetch(`${ENV.DETECTOR_URL}/v1/analyze`, {
    body: form,
    method: "POST",
    signal: AbortSignal.timeout(DETECTOR_TIMEOUT_MS),
  });
  if (!response.ok) {
    throw new Error(
      `Detector returned HTTP ${response.status}: ${await response.text()}`
    );
  }
  // SAFETY: the detector's FastAPI response_model is Verdict (ai_detector/schemas.py), mirrored by this type.
  return (await response.json()) as Verdict;
};

/** "Couldn't check this" verdicts carry no score; the detector's message already says what to send instead. */
const isUnchecked = (verdict: Verdict) =>
  verdict.verdict === "inconclusive" && verdict.p_ai === null;

/** Short text the router rejected as too short is conversation ("hi", "why?"), not something to check. */
const isChat = (input: CheckInput, verdict: Verdict) =>
  !input.file &&
  isUnchecked(verdict) &&
  verdict.evidence.some(
    (e) => e.id === "sys.router" && e.details.reason?.includes("shorter than")
  );

/** The verdict without bulky per-token / per-window lists, as in muse/client.py `_payload`. */
const agentPayload = (verdict: Verdict) =>
  JSON.stringify({
    ...verdict,
    evidence: verdict.evidence.map(
      ({ details: { timeline: _timeline, ...details }, ...e }) => ({
        ...e,
        details: { ...details, tokens: undefined },
      })
    ),
    report: undefined,
  });

const BOLD_PAIRS = /\*\*(?<text>.+?)\*\*/gu;

/**
 * report.answer is "headline\n\nwhy[\n\nfooter]" (muse/client.py `whatsapp_message`). Keep code's headline and
 * footer, swap in the agent's why. Any agent failure keeps the template why.
 */
const explain = async (
  verdict: Verdict,
  memory: { resource: string; thread: string }
) => {
  const answer = verdict.report?.answer ?? "";
  const [headline, templateWhy, ...footer] = answer.split("\n\n");
  if (!(ENV.MODEL_API_KEY && headline && templateWhy)) {
    return answer;
  }
  try {
    const result = await sniffAgent.generate(
      `VERDICT JSON\n${agentPayload(verdict)}`,
      { memory }
    );
    const why = result.text.trim().replaceAll(BOLD_PAIRS, "*$<text>*");
    const usable = why.length > 0 && why.length <= MAX_WHY_CHARS;
    return [headline, usable ? why : templateWhy, ...footer].join("\n\n");
  } catch (error) {
    log.error({ action: "check.agent_failed", error: String(error) });
    return answer;
  }
};

const chat = async (
  text: string,
  memory: { resource: string; thread: string },
  fallback: string
) => {
  if (!ENV.MODEL_API_KEY) {
    return fallback;
  }
  try {
    const result = await sniffAgent.generate(text, { memory });
    return result.text.trim().replaceAll(BOLD_PAIRS, "*$<text>*") || fallback;
  } catch (error) {
    log.error({ action: "check.chat_failed", error: String(error) });
    return fallback;
  }
};

export const runCheck = async (input: CheckInput): Promise<CheckResult> => {
  const verdict = await analyze(input);
  const owner = input.waFrom
    ? `wa:${input.waFrom}`
    : `web:${input.userId ?? "anonymous"}`;
  const memory = { resource: owner, thread: owner };

  if (isChat(input, verdict) && input.text) {
    return {
      id: null,
      reply: await chat(input.text, memory, verdict.report?.answer ?? ""),
      verdict: null,
    };
  }

  const reply = isUnchecked(verdict)
    ? (verdict.report?.answer ?? "")
    : await explain(verdict, memory);
  const id = crypto.randomUUID();
  await db.insert(check).values({
    filename: input.file?.filename ?? null,
    id,
    modality: verdict.modality,
    pAi: verdict.p_ai,
    reply,
    source: input.source,
    textPreview: input.text?.slice(0, PREVIEW_CHARS) ?? null,
    userId: input.userId ?? null,
    verdict: verdict.verdict,
    verdictJson: JSON.stringify(verdict),
    waFrom: input.waFrom ?? null,
  });
  return { id, reply, verdict };
};
