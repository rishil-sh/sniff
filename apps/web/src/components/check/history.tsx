"use client";

import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@sniff/ui/components/empty";
import { Skeleton } from "@sniff/ui/components/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@sniff/ui/components/table";
import { useQuery } from "@tanstack/react-query";
import { GlobeIcon, HistoryIcon, MessageCircleIcon } from "lucide-react";
import { useState } from "react";

import { formatDateTime } from "@/lib/format";
import { orpc } from "@/utils/orpc";

import { EvidenceSheet } from "./evidence-sheet";
import { VerdictBadge } from "./verdict-badge";

const SKELETON_ROWS = ["a", "b", "c", "d"];

export const History = () => {
  const list = useQuery(orpc.check.list.queryOptions());
  const [openId, setOpenId] = useState<string | null>(null);
  const detail = useQuery(
    orpc.check.get.queryOptions({
      enabled: openId !== null,
      input: { id: openId ?? "" },
    })
  );

  if (list.isPending) {
    return (
      <div className="flex flex-col gap-2">
        {SKELETON_ROWS.map((key) => (
          <Skeleton className="h-10 w-full" key={key} />
        ))}
      </div>
    );
  }

  if (!list.data?.length) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <HistoryIcon />
          </EmptyMedia>
          <EmptyTitle>No checks yet</EmptyTitle>
          <EmptyDescription>
            Everything you check on the web shows up here.
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    );
  }

  return (
    <>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Input</TableHead>
            <TableHead>Verdict</TableHead>
            <TableHead className="hidden md:table-cell">Reply</TableHead>
            <TableHead className="text-right">When</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {list.data.map((row) => (
            <TableRow key={row.id}>
              <TableCell>
                <button
                  className="flex max-w-56 items-center gap-2 text-left hover:underline"
                  onClick={() => setOpenId(row.id)}
                  type="button"
                >
                  {row.source === "whatsapp" ? (
                    <MessageCircleIcon
                      aria-label="WhatsApp"
                      className="text-primary size-4 shrink-0"
                    />
                  ) : (
                    <GlobeIcon
                      aria-label="Web"
                      className="text-muted-foreground size-4 shrink-0"
                    />
                  )}
                  <span className="truncate">
                    <span className="capitalize">{row.modality}</span>
                    {(row.filename ?? row.textPreview) && (
                      <span className="text-muted-foreground">
                        {" "}
                        · {row.filename ?? row.textPreview}
                      </span>
                    )}
                  </span>
                </button>
              </TableCell>
              <TableCell>
                <VerdictBadge pAi={row.pAi} verdict={row.verdict} />
              </TableCell>
              <TableCell className="hidden md:table-cell">
                <span className="text-muted-foreground line-clamp-1 max-w-md whitespace-normal">
                  {row.reply.replaceAll(/[*_~]/gu, "")}
                </span>
              </TableCell>
              <TableCell className="text-right">
                <span className="text-muted-foreground text-xs">
                  {formatDateTime(row.createdAt)}
                </span>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <EvidenceSheet
        onOpenChange={(next) => !next && setOpenId(null)}
        open={openId !== null}
        reply={detail.data?.reply}
        verdict={detail.data?.verdict ?? null}
      />
    </>
  );
};
