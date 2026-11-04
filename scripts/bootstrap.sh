#!/bin/sh
# One-time local setup: env files, JS + Python dependencies, database. Safe to re-run.
set -e
cd "$(dirname "$0")/.."

need() { command -v "$1" >/dev/null 2>&1 || { echo "✗ $1 is required: $2"; exit 1; }; }
need node "https://nodejs.org (v24+)"
need pnpm "npm install -g pnpm"
need bun "https://bun.sh"
need uv "https://docs.astral.sh/uv/getting-started/installation/"

for dir in apps/server apps/web apps/detector; do
  if [ ! -f "$dir/.env" ]; then
    cp "$dir/.env.example" "$dir/.env"
    echo "✓ created $dir/.env"
  fi
done
if grep -q '^BETTER_AUTH_SECRET=$' apps/server/.env; then
  secret=$(node -e 'console.log(require("node:crypto").randomBytes(32).toString("base64url"))')
  sed -i.bak "s|^BETTER_AUTH_SECRET=$|BETTER_AUTH_SECRET=$secret|" apps/server/.env && rm apps/server/.env.bak
  echo "✓ generated BETTER_AUTH_SECRET"
fi

pnpm install
(cd apps/detector && uv sync --extra dev)
mkdir -p .data
pnpm db:migrate

cat <<'DONE'

✓ Ready. Next:
  pnpm models   # download detector models (~10 GB, ~23 GB with the Llama pair); optional, else first use downloads them
  pnpm dev      # web http://localhost:3001 · API :3000 · detector :7860
DONE
