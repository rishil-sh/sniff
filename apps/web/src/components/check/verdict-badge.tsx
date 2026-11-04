import type { VerdictLabel } from "@sniff/api/verdict";
import { Badge } from "@sniff/ui/components/badge";

const LABELS = {
  declared_ai: { label: "Declared AI", variant: "destructive" },
  declared_capture: { label: "Real capture", variant: "default" },
  inconclusive: { label: "Inconclusive", variant: "secondary" },
  likely_ai: { label: "Likely AI", variant: "destructive" },
  no_ai_signal: { label: "No AI signal", variant: "default" },
} as const satisfies Record<
  VerdictLabel,
  { label: string; variant: "default" | "destructive" | "secondary" }
>;

const isVerdictLabel = (value: string): value is VerdictLabel =>
  Object.hasOwn(LABELS, value);

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
  if (verdict === "inconclusive" && pAi === null) {
    return <Badge variant="outline">Couldn&apos;t check</Badge>;
  }
  const { label, variant } = LABELS[verdict];
  return <Badge variant={variant}>{label}</Badge>;
};
