// Mirrors apps/detector/ai_detector/schemas.py (the fields the TS side reads). Keep in sync when the Python changes.

export type Modality = "image" | "video" | "audio" | "text" | "unknown";
export type VerdictLabel =
  | "declared_ai"
  | "declared_capture"
  | "likely_ai"
  | "inconclusive"
  | "no_ai_signal";

/** The `details` keys the TS side reads; the detector sends more. */
export interface EvidenceDetails {
  reason?: string;
  seconds?: number;
  window_s?: number;
  timeline?: { t: number; p_ai: number }[];
}

export interface Evidence {
  id: string;
  detector: string;
  version: string;
  modality: Modality;
  signal: "provenance" | "watermark" | "learned" | "zero_shot" | "vlm" | "rule";
  status: "ok" | "skipped" | "error" | "out_of_domain";
  raw_score: number | null;
  p_ai: number | null;
  points_to: "ai" | "real" | "none";
  weight: number;
  details: EvidenceDetails;
  latency_ms: number;
}

export interface Verdict {
  request_id: string;
  modality: Modality;
  sha256: string;
  verdict: VerdictLabel;
  p_ai: number | null;
  bands: { no_ai_signal_below: number; likely_ai_above: number };
  decided_by: "provenance" | "watermark" | "fusion";
  evidence: Evidence[];
  report: {
    answer: string;
    summary: string;
    lean: "ai" | "human" | "unclear";
  } | null;
  transcript: string | null;
  generated_at: string;
}

export interface CheckInput {
  source: "web" | "whatsapp";
  file?: { bytes: Uint8Array; filename: string; mimeType?: string };
  text?: string;
  userId?: string;
  waFrom?: string;
}

/** `id`/`verdict` are null when the input was conversation (e.g. "hi", "why?") rather than something to check. */
export interface CheckResult {
  id: string | null;
  verdict: Verdict | null;
  reply: string;
}

export type RunCheck = (input: CheckInput) => Promise<CheckResult>;
