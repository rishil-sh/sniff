import type { VerdictLabel } from "@sniff/api/verdict";
import { Badge } from "@sniff/ui/components/badge";
import { cn } from "@sniff/ui/lib/utils";
import Image from "next/image";

type Mood =
  | "likely-ai"
  | "inconclusive"
  | "no-ai-signal"
  | "declared-ai"
  | "real-capture";

/**
 * Sherlock's expression and colours for each verdict (docs/brand/sticker-set.png). Each mood has a full pose at
 * /brand/pose-<mood>.png and a small face at /brand/face-<mood>.png.
 */
export const MOODS = {
  "declared-ai": {
    className: "bg-verdict-declared-soft text-verdict-declared",
    expression: "holds up the AI label he found",
    label: "Declared AI",
  },
  inconclusive: {
    className: "bg-verdict-unsure-soft text-verdict-unsure",
    expression: "tilts his head: he can't tell",
    label: "Inconclusive",
  },
  "likely-ai": {
    className: "bg-verdict-ai-soft text-verdict-ai",
    expression: "points: this looks AI-made",
    label: "Likely AI",
  },
  "no-ai-signal": {
    className: "bg-verdict-clear-soft text-verdict-clear",
    expression: "wags his tail: no AI signal",
    label: "No AI signal found",
  },
  "real-capture": {
    className: "bg-verdict-clear-soft text-verdict-clear",
    expression: "holds up a camera: a real capture",
    label: "Real capture",
  },
} as const satisfies Record<
  Mood,
  { className: string; expression: string; label: string }
>;

/** Badge variant per mood: a real capture shares the no-AI-signal colours. */
const BADGE_VARIANT = {
  "declared-ai": "declared-ai",
  inconclusive: "inconclusive",
  "likely-ai": "likely-ai",
  "no-ai-signal": "no-ai-signal",
  "real-capture": "no-ai-signal",
} as const satisfies Record<Mood, string>;

export const MOOD_ORDER = [
  "likely-ai",
  "inconclusive",
  "no-ai-signal",
  "declared-ai",
] as const satisfies readonly Mood[];

const LABELS = {
  declared_ai: { label: "Declared AI", mood: "declared-ai" },
  declared_capture: { label: "Real capture", mood: "real-capture" },
  inconclusive: { label: "Inconclusive", mood: "inconclusive" },
  likely_ai: { label: "Likely AI", mood: "likely-ai" },
  no_ai_signal: { label: "No AI signal", mood: "no-ai-signal" },
} as const satisfies Record<VerdictLabel, { label: string; mood: Mood }>;

const isVerdictLabel = (value: string): value is VerdictLabel =>
  Object.hasOwn(LABELS, value);

/** Which Sherlock to show for a verdict; anything unknown or unchecked gets the head tilt. */
export const moodOf = (verdict: string | undefined): Mood =>
  verdict && isVerdictLabel(verdict) ? LABELS[verdict].mood : "inconclusive";

export const SherlockFace = ({
  className,
  mood,
  size = 32,
  ...props
}: {
  className?: string;
  "data-icon"?: string;
  mood: Mood;
  size?: number;
}) => (
  <Image
    {...props}
    alt={`Sherlock ${MOODS[mood].expression}`}
    className={cn("object-contain", className)}
    height={size}
    src={`/brand/face-${mood}.png`}
    width={size}
  />
);

const POSE_ALT = {
  "lost-scent":
    "Sherlock sitting with an empty magnifying glass: he lost the scent",
  sniffing: "Sherlock sniffing the ground with a magnifying glass",
} as const;

type Pose = Mood | keyof typeof POSE_ALT;

const isWorkingPose = (pose: Pose): pose is keyof typeof POSE_ALT =>
  Object.hasOwn(POSE_ALT, pose);

/** A full Sherlock sticker: a verdict pose, or sniffing (working) and lost-scent (something went wrong). */
export const SherlockPose = ({
  className,
  height,
  pose,
  priority,
  width,
}: {
  className?: string;
  height: number;
  pose: Pose;
  priority?: boolean;
  width: number;
}) => (
  <Image
    alt={
      isWorkingPose(pose)
        ? POSE_ALT[pose]
        : `Sherlock ${MOODS[pose].expression}`
    }
    className={cn("object-contain", className)}
    height={height}
    priority={priority}
    src={`/brand/pose-${pose}.png`}
    width={width}
  />
);

/** `pAi === null` on an inconclusive verdict means the detector couldn't check the input at all. */
export const VerdictBadge = ({
  pAi,
  verdict,
}: {
  pAi: number | null;
  verdict: string;
}) => {
  if (!isVerdictLabel(verdict)) {
    return <Badge variant="outline">{verdict}</Badge>;
  }
  const unchecked = verdict === "inconclusive" && pAi === null;
  const { label, mood } = LABELS[verdict];
  return (
    <Badge variant={BADGE_VARIANT[mood]}>
      <SherlockFace
        className="-my-0.5 size-4"
        data-icon="inline-start"
        mood={mood}
        size={16}
      />
      {unchecked ? "Couldn't check" : label}
    </Badge>
  );
};
