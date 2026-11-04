"use client";

import type { CheckResult } from "@sniff/api/verdict";
import {
  Attachment,
  AttachmentAction,
  AttachmentActions,
  AttachmentContent,
  AttachmentGroup,
  AttachmentMedia,
  AttachmentTitle,
} from "@sniff/ui/components/attachment";
import { Bubble, BubbleContent } from "@sniff/ui/components/bubble";
import { Button } from "@sniff/ui/components/button";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@sniff/ui/components/empty";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupTextarea,
} from "@sniff/ui/components/input-group";
import { Message, MessageAvatar } from "@sniff/ui/components/message";
import {
  MessageScroller,
  MessageScrollerButton,
  MessageScrollerContent,
  MessageScrollerItem,
  MessageScrollerProvider,
  MessageScrollerViewport,
} from "@sniff/ui/components/message-scroller";
import { useMutation } from "@tanstack/react-query";
import { FileIcon, PaperclipIcon, SendIcon, XIcon } from "lucide-react";
import Image from "next/image";
import { useRef, useState } from "react";

import { WhatsAppText } from "@/components/chat/whatsapp-text";
import { orpc } from "@/utils/orpc";

import { EvidenceSheet } from "./evidence-sheet";
import {
  MOOD_ORDER,
  MOODS,
  moodOf,
  SherlockFace,
  SherlockPose,
  VerdictBadge,
} from "./verdict-badge";

const ACCEPT = "image/*,audio/*,video/mp4,.txt";

interface Sent {
  id: string;
  file?: { name: string; type: string; url: string };
  text?: string;
}

interface Turn {
  sent: Sent;
  result?: CheckResult;
  error?: string;
}

const SentMedia = ({ file }: { file: NonNullable<Sent["file"]> }) => {
  if (file.type.startsWith("image/")) {
    return (
      <Image
        alt={file.name}
        className="max-h-64 w-auto rounded-md"
        height={256}
        src={file.url}
        unoptimized
        width={256}
      />
    );
  }
  if (file.type.startsWith("audio/")) {
    return (
      <audio className="max-w-full" controls src={file.url}>
        <track kind="captions" />
      </audio>
    );
  }
  if (file.type.startsWith("video/")) {
    return (
      <video className="max-h-64 rounded-md" controls src={file.url}>
        <track kind="captions" />
      </video>
    );
  }
  return (
    <span className="flex items-center gap-2">
      <FileIcon className="size-4" /> {file.name}
    </span>
  );
};

export const CheckChat = () => {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [open, setOpen] = useState<CheckResult | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const create = useMutation(orpc.check.create.mutationOptions());

  const update = (id: string, patch: Partial<Turn>) =>
    setTurns((current) =>
      current.map((turn) =>
        turn.sent.id === id ? { ...turn, ...patch } : turn
      )
    );

  const send = () => {
    const trimmed = text.trim();
    if (!(trimmed || file) || create.isPending) {
      return;
    }
    const sent: Sent = {
      file: file
        ? { name: file.name, type: file.type, url: URL.createObjectURL(file) }
        : undefined,
      id: crypto.randomUUID(),
      text: trimmed || undefined,
    };
    setTurns((current) => [...current, { sent }]);
    setText("");
    setFile(null);
    create.mutate(
      { file: file ?? undefined, text: trimmed || undefined },
      {
        onError: (error) => update(sent.id, { error: error.message }),
        onSuccess: (result) => update(sent.id, { result }),
      }
    );
  };

  return (
    <div
      className="bg-background data-[dragging=true]:ring-primary relative flex h-[calc(100svh-var(--header-height))] flex-col data-[dragging=true]:ring-2 data-[dragging=true]:ring-inset"
      data-dragging={dragging}
      data-slot="product-layout"
      onDragLeave={() => setDragging(false)}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        setFile(event.dataTransfer.files[0] ?? null);
      }}
    >
      {turns.length === 0 ? (
        <Empty className="flex-1">
          <EmptyHeader className="max-w-md">
            <EmptyMedia>
              <Image
                alt="Sherlock, Sniff's beagle detective"
                className="size-32 drop-shadow-md"
                height={128}
                priority
                src="/brand/sherlock.png"
                width={128}
              />
            </EmptyMedia>
            <EmptyTitle>Is this AI?</EmptyTitle>
            <EmptyDescription>
              Drop in a photo, a voice note or a video, or paste a paragraph of
              English text (about 110 words or more). Sherlock sniffs it and
              shows you the evidence. You can also forward things to Sniff on
              WhatsApp.
            </EmptyDescription>
          </EmptyHeader>
          <ul
            aria-label="Possible answers"
            className="flex flex-wrap justify-center gap-2"
          >
            {MOOD_ORDER.map((mood) => (
              <li
                className={`flex items-center gap-1 rounded-full py-0.5 pr-3 pl-1 text-xs font-semibold ${MOODS[mood].className}`}
                key={mood}
              >
                <SherlockFace className="size-6" mood={mood} size={24} />
                {MOODS[mood].label}
              </li>
            ))}
          </ul>
        </Empty>
      ) : (
        <MessageScrollerProvider autoScroll>
          <MessageScroller className="flex-1">
            <MessageScrollerViewport>
              <MessageScrollerContent
                className="mx-3 my-4 md:mx-6"
                density="compact"
              >
                {turns.map((turn, index) => (
                  <MessageScrollerItem
                    key={turn.sent.id}
                    messageId={turn.sent.id}
                    scrollAnchor={index === turns.length - 1}
                  >
                    <div className="flex flex-col gap-2">
                      <Message align="end">
                        <Bubble align="end" variant="outgoing">
                          <BubbleContent>
                            <div className="flex flex-col gap-2">
                              {turn.sent.file && (
                                <SentMedia file={turn.sent.file} />
                              )}
                              {turn.sent.text && (
                                <WhatsAppText text={turn.sent.text} />
                              )}
                            </div>
                          </BubbleContent>
                        </Bubble>
                      </Message>
                      <Message>
                        <MessageAvatar>
                          {turn.result ? (
                            <SherlockFace
                              className="size-8"
                              mood={moodOf(turn.result.verdict?.verdict)}
                            />
                          ) : (
                            <SherlockPose
                              className="size-8"
                              height={32}
                              pose={turn.error ? "lost-scent" : "sniffing"}
                              width={32}
                            />
                          )}
                        </MessageAvatar>
                        <Bubble
                          variant={turn.error ? "destructive" : "incoming"}
                        >
                          <BubbleContent>
                            {!(turn.result || turn.error) && (
                              <span className="text-muted-foreground flex items-center gap-2">
                                <SherlockPose
                                  className="h-6 w-auto motion-safe:animate-pulse"
                                  height={24}
                                  pose="sniffing"
                                  width={40}
                                />
                                Sherlock is sniffing…
                              </span>
                            )}
                            {turn.error && <span>{turn.error}</span>}
                            {turn.result && (
                              <div className="flex flex-col gap-2">
                                <WhatsAppText text={turn.result.reply} />
                                {turn.result.verdict && (
                                  <div className="flex items-center gap-2">
                                    <VerdictBadge
                                      pAi={turn.result.verdict.p_ai}
                                      verdict={turn.result.verdict.verdict}
                                    />
                                    <Button
                                      onClick={() =>
                                        setOpen(turn.result ?? null)
                                      }
                                      size="xs"
                                      variant="ghost"
                                    >
                                      Evidence
                                    </Button>
                                  </div>
                                )}
                              </div>
                            )}
                          </BubbleContent>
                        </Bubble>
                      </Message>
                    </div>
                  </MessageScrollerItem>
                ))}
              </MessageScrollerContent>
            </MessageScrollerViewport>
            <MessageScrollerButton />
          </MessageScroller>
        </MessageScrollerProvider>
      )}

      <div className="bg-background/80 border-t p-3 backdrop-blur md:px-6">
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-2">
          {file && (
            <AttachmentGroup>
              <Attachment size="sm">
                <AttachmentMedia>
                  <FileIcon />
                </AttachmentMedia>
                <AttachmentContent>
                  <AttachmentTitle>{file.name}</AttachmentTitle>
                </AttachmentContent>
                <AttachmentActions>
                  <AttachmentAction
                    aria-label="Remove file"
                    onClick={() => setFile(null)}
                  >
                    <XIcon />
                  </AttachmentAction>
                </AttachmentActions>
              </Attachment>
            </AttachmentGroup>
          )}
          <InputGroup>
            <InputGroupAddon align="inline-start">
              <input
                accept={ACCEPT}
                aria-label="Choose a file"
                className="hidden"
                onChange={(event) => {
                  setFile(event.target.files?.[0] ?? null);
                  event.target.value = "";
                }}
                ref={fileInput}
                type="file"
              />
              <InputGroupButton
                aria-label="Attach a file"
                onClick={() => fileInput.current?.click()}
                size="icon-xs"
              >
                <PaperclipIcon />
              </InputGroupButton>
            </InputGroupAddon>
            <InputGroupTextarea
              aria-label="Text to check"
              className="max-h-40 min-h-9"
              onChange={(event) => setText(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  send();
                }
              }}
              placeholder="Paste text, or attach a photo, voice note or video…"
              rows={1}
              value={text}
            />
            <InputGroupAddon align="inline-end">
              <InputGroupButton
                aria-label="Check"
                disabled={!(text.trim() || file) || create.isPending}
                onClick={send}
                size="icon-xs"
                variant="default"
              >
                <SendIcon />
              </InputGroupButton>
            </InputGroupAddon>
          </InputGroup>
        </div>
      </div>

      <EvidenceSheet
        onOpenChange={(next) => !next && setOpen(null)}
        open={open !== null}
        reply={open?.reply}
        verdict={open?.verdict ?? null}
      />
    </div>
  );
};
