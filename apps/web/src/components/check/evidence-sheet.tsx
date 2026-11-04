"use client";

import type { Evidence, Verdict } from "@sniff/api/verdict";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@sniff/ui/components/sheet";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@sniff/ui/components/table";

import { WhatsAppText } from "@/components/chat/whatsapp-text";

import { VerdictBadge } from "./verdict-badge";

const percent = (value: number | null) =>
  value === null ? "—" : `${Math.round(value * 100)}%`;

const DECIDED_BY = {
  fusion: "Detectors combined",
  provenance: "File's Content Credentials",
  watermark: "Embedded watermark",
} as const satisfies Record<Verdict["decided_by"], string>;

const TIMELINE_HEIGHT = 48;

/** Per-window AI probability for a voice clip, as a plain SVG area line. */
const Timeline = ({ evidence }: { evidence: Evidence }) => {
  const points = evidence.details.timeline ?? [];
  const last = points.at(-1);
  if (points.length < 2 || !last) {
    return null;
  }
  const end = last.t + (evidence.details.window_s ?? 0);
  const path = points
    .map(
      ({ p_ai, t }) =>
        `${((t / end) * 100).toFixed(2)},${((1 - p_ai) * TIMELINE_HEIGHT).toFixed(2)}`
    )
    .join(" ");
  return (
    <figure className="flex flex-col gap-1">
      <figcaption className="text-muted-foreground text-xs">
        How AI-like each part of the clip sounded ({Math.round(end)} s)
      </figcaption>
      <svg
        aria-hidden
        className="text-primary bg-muted/50 h-12 w-full rounded"
        preserveAspectRatio="none"
        viewBox={`0 0 100 ${TIMELINE_HEIGHT}`}
      >
        <polyline
          fill="none"
          points={path}
          stroke="currentColor"
          strokeWidth={1.5}
          vectorEffect="non-scaling-stroke"
        />
      </svg>
    </figure>
  );
};

export const EvidenceSheet = ({
  onOpenChange,
  open,
  reply,
  verdict,
}: {
  onOpenChange: (open: boolean) => void;
  open: boolean;
  reply?: string;
  verdict: Verdict | null;
}) => (
  <Sheet onOpenChange={onOpenChange} open={open}>
    <SheetContent className="w-full overflow-y-auto sm:max-w-xl!">
      {verdict && (
        <>
          <SheetHeader>
            <SheetTitle>
              <span className="flex items-center gap-2">
                <VerdictBadge pAi={verdict.p_ai} verdict={verdict.verdict} />
                <span className="capitalize">{verdict.modality}</span>
              </span>
            </SheetTitle>
            <SheetDescription>
              {DECIDED_BY[verdict.decided_by]} · AI likelihood{" "}
              {percent(verdict.p_ai)}
            </SheetDescription>
          </SheetHeader>
          <div className="flex flex-col gap-6 px-4 pb-6">
            {verdict.p_ai !== null && (
              <progress
                aria-label={`AI likelihood ${percent(verdict.p_ai)}`}
                className="[&::-moz-progress-bar]:bg-primary [&::-webkit-progress-bar]:bg-muted [&::-webkit-progress-value]:bg-primary h-2 w-full appearance-none overflow-hidden rounded-full"
                max={1}
                value={verdict.p_ai}
              />
            )}
            {reply && (
              <div className="bg-wa-wallpaper rounded-lg p-3">
                <div className="bg-wa-bubble text-wa-text w-fit rounded-lg px-3 py-2 text-sm shadow-xs">
                  <WhatsAppText text={reply} />
                </div>
              </div>
            )}
            {verdict.evidence.map((evidence) =>
              evidence.details.timeline ? (
                <Timeline evidence={evidence} key={evidence.id} />
              ) : null
            )}
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Detector</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">AI</TableHead>
                  <TableHead className="text-right">Weight</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {verdict.evidence.map((evidence) => (
                  <TableRow key={evidence.id}>
                    <TableCell>
                      <div
                        className={`font-mono text-xs ${evidence.weight > 0 ? "" : "text-muted-foreground"}`}
                      >
                        {evidence.id}
                      </div>
                      <div className="text-muted-foreground text-xs">
                        {evidence.signal}
                      </div>
                    </TableCell>
                    <TableCell>
                      <span className="text-xs">
                        {evidence.status.replaceAll("_", " ")}
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      <span className="tabular-nums">
                        {percent(evidence.p_ai)}
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      <span className="tabular-nums">{evidence.weight}</span>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            <p className="text-muted-foreground font-mono text-xs break-all">
              sha256 {verdict.sha256}
            </p>
          </div>
        </>
      )}
    </SheetContent>
  </Sheet>
);
