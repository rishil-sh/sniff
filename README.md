# Sniff

**Is this AI?** Send Sniff a photo, a voice note, a video or a paragraph of text, on the web or over WhatsApp, and it answers in plain words, with the evidence behind the answer.

- **Evidence first.** Content Credentials (C2PA), invisible watermarks (Stable Diffusion, Meta AudioSeal) and generator metadata decide when present; otherwise calibrated detectors vote, with an honest _inconclusive_ band.
- **Runs on your machine.** Every detector model runs locally. No API key is needed for a verdict.
- **Friendly replies.** A [Mastra](https://mastra.ai) agent on Muse (Meta Model API) explains the verdict and answers follow-up questions. Without a key, Sniff uses built-in template replies.
- **Web and WhatsApp.** The same pipeline answers both, and every check lands in your history.

> Verdicts are automated estimates, not proof. Please don't use them as the only evidence against anyone.

## Quick start (Docker)

```bash
git clone <this repo> sniff && cd sniff
docker compose up
```

Open **http://localhost:3001** and create an account (it stays on your machine).

The first start downloads the detector models (~10 GB) into a Docker volume, so it takes a while; later starts take seconds. The images contain no model weights. To fetch models ahead of time, or after changing settings: `docker compose run --rm models`.

Optional settings go in a `.env` file next to `docker-compose.yml`. Copy [.env.example](.env.example):

| Setting | What it does |
| --- | --- |
| `MODEL_API_KEY` | Muse (Meta Model API) key for the friendly replies. Get one at https://dev.meta.ai |
| `AI_DETECTOR_TEXT_PAIR=1` | Adds the Llama-3.2-3B zero-shot text evidence (~13 GB more; give Docker ≥16 GB of memory) |
| `WHATSAPP_*` | Connects a WhatsApp number (see below) |
| `PUBLIC_SERVER_URL`, `PUBLIC_WEB_URL` | Your URLs when not on localhost |
| `ADMIN_EMAILS` | Emails that may see WhatsApp checks in the web History |

## How it works

```
web (Next.js :3001) ──oRPC──┐
                            ├─► server (Hono :3000) ──► detector (FastAPI :7860) → Verdict + evidence
WhatsApp ─► /webhooks/whatsapp┘       ├─ Mastra agent (Muse) → plain-words reply
                                      └─ SQLite: history, accounts, agent memory
```

| Input | Evidence |
| --- | --- |
| Image | C2PA / IPTC credentials, generator metadata, SD/SDXL watermark, Community Forensics + CLIP detectors |
| Voice | Meta AudioSeal watermark, w2v-BERT 2.0 probe (decides), DF Arena and XLSR-MamBo (supporting) |
| Text | Fakespot RoBERTa + Desklib DeBERTa classifiers (decide), Binoculars / Fast-DetectGPT / GLTR on Llama-3.2-3B (supporting) |
| Video | Not supported yet (provenance only) |

The detector writes the verdict headline itself. The agent only writes the explanation, so it can't change a verdict. Details, calibration and measured accuracy: [apps/detector/README.md](apps/detector/README.md).

## Repository layout

| Path | What it is |
| --- | --- |
| [apps/detector](apps/detector) | Python detector (FastAPI, PyTorch), uv project |
| [apps/server](apps/server) | API: Hono + oRPC + better-auth on Bun, Mastra agent, WhatsApp webhook |
| [apps/web](apps/web) | Next.js web app: Check (chat) and History |
| [packages/ui](packages/ui) | Shared shadcn/ui components and theme |
| [packages/api](packages/api) | oRPC routers and shared types |
| [packages/db](packages/db) | Drizzle schema and migrations (SQLite / libSQL) |

## Development

You need Node.js 24+, [pnpm](https://pnpm.io), [Bun](https://bun.sh) and [uv](https://docs.astral.sh/uv/).

```bash
pnpm bootstrap   # env files, dependencies, database
pnpm models      # optional: download models now instead of on first use
pnpm dev         # web :3001 · API :3000 · detector :7860
```

| Command |  |
| --- | --- |
| `pnpm check` / `pnpm fix` | Lint and format (Ultracite: oxlint + oxfmt) |
| `pnpm check-types` | TypeScript across all workspaces |
| `pnpm test` | Detector (pytest) and server (bun) tests |
| `pnpm db:generate` / `pnpm db:migrate` | New migration from the schema / apply migrations |
| `pnpm db:studio` | Browse the database |

## WhatsApp

1. Create a Meta app with the WhatsApp product. Put its access token, phone number ID and app secret, plus a verify token you choose, in `.env` (Docker) or `apps/server/.env` (local), as `WHATSAPP_*`.
2. Expose the server over HTTPS (e.g. `cloudflared tunnel --url http://localhost:3000`) and set the webhook URL to `https://<host>/webhooks/whatsapp` with the same verify token. Subscribe to `messages`.
3. Send the number a photo, a voice note or a paragraph of text.

Voice notes arrive as Ogg/Opus. The detector decodes them with the ffmpeg bundled in its image, so the API image does not include ffmpeg.

## Deploying the web app to Cloudflare (optional)

`packages/infra` deploys `apps/web` to Cloudflare Workers with [Alchemy](https://alchemy.run). Run `cd packages/infra && pnpm exec alchemy profile edit`, then `pnpm deploy`. The server and detector run anywhere Docker does.
