# Sniff

Is this AI? Send a photo, voice note, video or paragraph of text, on the web or over WhatsApp, and Sniff replies in
plain words with a verdict, backed by the detector evidence.

```
web (Next :3001) ──oRPC /rpc──┐
                              ├─► server (Hono :3000) ──► detector (FastAPI :7860) /v1/analyze → Verdict
WhatsApp ─► /webhooks/whatsapp┘        ├─ Mastra agent on Muse (Meta Model API) → friendly reply
                                       └─ SQLite: `check` history + Mastra memory
```

- `apps/detector`: the Python detector, an exact copy of `ai-detection-suite` (router → detectors → fusion). See
  its own README. Run `uv sync --extra dev --extra deploy` once; models download into `apps/detector/data/models`.
- `apps/server`: Hono + oRPC + better-auth. `src/check.ts` runs detector → agent → DB for both channels;
  `src/whatsapp.ts` is the WhatsApp Cloud API webhook; `src/mastra/` holds the Sniff agent.
- `apps/web`: Next.js + shadcn (`packages/ui`, logicsense theme). **Check** is a chat; **History** lists past checks.

The detector writes the bold headline and footer; the agent (Muse) only writes the explanation line, or chats when
the message is a question rather than something to check. Without `MODEL_API_KEY` the detector's template text is
used.

## Getting Started

```bash
pnpm install
(cd apps/detector && uv sync --extra dev --extra deploy)
pnpm db:migrate
pnpm dev          # detector :7860, server :3000, web :3001
```

Voice notes from WhatsApp are Ogg/Opus; the server converts them to WAV with `ffmpeg` (`brew install ffmpeg`, or set
`FFMPEG`).

## WhatsApp

1. In a Meta app with the WhatsApp product, set `WHATSAPP_ACCESS_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`,
   `WHATSAPP_APP_SECRET` and `WHATSAPP_VERIFY_TOKEN` in `apps/server/.env`.
2. Expose the server (e.g. `cloudflared tunnel --url http://localhost:3000`) and set the webhook to
   `https://<host>/webhooks/whatsapp` with the same verify token. Subscribe to `messages`.
3. Add your email to `ADMIN_EMAILS` to see WhatsApp checks in the web History.

## Database Setup

This project uses SQLite with Drizzle ORM.

1. SQLite stores data in a local file, so no database server is needed. Applying the schema creates the file `DATABASE_URL` points to.

2. Update your `.env` file in the `apps/server` directory with the appropriate connection details if needed.

3. Apply the schema to your database:

```bash
pnpm run db:push
```

Then, run the development server:

```bash
pnpm run dev
```

Open [http://localhost:3001](http://localhost:3001) in your browser to see the web application.
The API is running at [http://localhost:3000](http://localhost:3000).

## UI Customization

React web apps in this stack share shadcn/ui primitives through `packages/ui`.

- Change design tokens and global styles in `packages/ui/src/styles/globals.css`
- Update shared primitives in `packages/ui/src/components/*`
- Adjust shadcn aliases or style config in `packages/ui/components.json` and `apps/web/components.json`

### Add more shared components

Run this from the project root to add more primitives to the shared UI package:

```bash
npx shadcn@latest add accordion dialog popover sheet table -c packages/ui
```

Import shared components like this:

```tsx
import { Button } from "@sniff/ui/components/button";
```

### Add app-specific blocks

If you want to add app-specific blocks instead of shared primitives, run the shadcn CLI from `apps/web`.

## Environment Configuration

Each app owns its environment schema in `.env.schema`. Varlock generates `src/env.ts` during installation; run `pnpm run env:generate` after changing a schema. Commit schemas, and keep secrets in ignored env files or your deployment platform.

Import the generated `ENV` accessor in application code. Shared database and auth packages receive configuration or initialized clients from the application. See [Varlock's monorepo guide](https://varlock.dev/guides/monorepos/).

For Cloudflare, Alchemy loads and validates deployment inputs with `varlock/auto-load` in its Node/Bun deployment process. Worker code reads native bindings; web clients use the framework's public env API through `src/env.public.ts` where needed. Alchemy supplies resource URLs and managed database credentials. In-Worker Varlock protections are deferred until an official Alchemy integration is available; see [the non-Wrangler deployment guidance](https://varlock.dev/integrations/cloudflare/#non-wrangler-deploy-tools-alchemy-sst-pulumi).

Bun's automatic env loading is disabled in `bunfig.toml`; the framework integration or server bootstrap loads Varlock. Node deployments must include Varlock and its dependencies alongside the app schema.

Run standalone Node/Bun tools that use Varlock from the owning app directory so they load that app's schema and env files. `env:generate` only generates TypeScript files; it does not initialize environment values in a subsequent command.

## Deployment

### Alchemy

- Target: web on Cloudflare
- Configure provider accounts: `cd packages/infra && pnpm exec alchemy profile edit`
- Dev: pnpm run dev
- Deploy: pnpm run deploy
- Destroy: pnpm run destroy

`alchemy profile edit` stores the selected Axiom, Cloudflare, Neon, PlanetScale, and/or Prisma provider profiles under `~/.alchemy`; no provider-specific setup command is required by this scaffold.

Deploys are staged and default to a personal `dev_<username>` stage. For production, run the deploy with an explicit stage from `packages/infra`:

```bash
cd packages/infra && pnpm exec alchemy deploy --stage production
```

### Docker Compose

- Target: server
- Config: `docker-compose.yml` (app Dockerfiles live in `apps/*/Dockerfile`)
- Build images: pnpm run docker:build
- Start: pnpm run docker:up
- Logs: pnpm run docker:logs
- Stop: pnpm run docker:down

Environment variables are read from each app's `.env` file (baked into web builds for public variables) and overridden in `docker-compose.yml` for container networking.

Docker Compose uses the local `./.data/local.db` file. Run `pnpm run db:push` before starting the stack.

For more details, see the guide on [Deploying with Docker Compose](https://www.better-t-stack.dev/docs/guides/docker).

## Git Hooks and Formatting

- Run checks: `pnpm run check`

## Project Structure

```
sniff/
├── apps/
│   ├── web/         # Frontend application (Next.js)
│   └── server/      # Backend API (Hono, ORPC)
├── packages/
│   ├── ui/          # Shared shadcn/ui components and styles
│   ├── api/         # API layer / business logic
│   ├── auth/        # Authentication configuration & logic
│   └── db/          # Database schema & queries
```

## Available Scripts

- `pnpm run dev`: Start all applications in development mode
- `pnpm run build`: Build all applications
- `pnpm run dev:web`: Start only the web application
- `pnpm run dev:server`: Start only the server
- `pnpm run check-types`: Check TypeScript types across all apps
- `pnpm run db:push`: Push schema changes to database
- `pnpm run db:generate`: Generate database client/types
- `pnpm run db:migrate`: Run database migrations
- `pnpm run db:studio`: Open database studio UI
- `pnpm run check`: Run Oxlint and Oxfmt
- `cd apps/web && pnpm run generate-pwa-assets`: Generate PWA assets
- `pnpm run docker:build`: Build the Docker Compose images
- `pnpm run docker:up`: Build and start the Docker Compose stack
- `pnpm run docker:logs`: Tail logs from the Docker Compose stack
- `pnpm run docker:down`: Stop the Docker Compose stack

## Better Auth Schema Generation

After changing auth plugins or schema options, run `pnpm run auth:generate` from the project root. The script runs the Better Auth CLI through `varlock run` from the owning app directory, loading the auth instance from `src/services.ts`. Review the schema changes, then use your ORM's migration workflow to apply them.
