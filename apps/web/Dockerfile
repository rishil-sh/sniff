# Web app. `next build` compiles the app in the build stage; the final Alpine image only has
# the standalone server, static assets, and the node binary (npm is removed). Build context: repo root.

FROM node:24-alpine AS base
RUN apk add --no-cache libc6-compat \
    && npm install -g pnpm@12.3.4 turbo@2
WORKDIR /app

FROM base AS prune
COPY . .
RUN turbo prune web --docker

FROM base AS build
COPY --from=prune /app/out/json/ .
RUN --mount=type=cache,target=/pnpm-store pnpm install --frozen-lockfile --ignore-scripts --store-dir /pnpm-store
COPY --from=prune /app/out/full/ .
# Where browsers reach the API; baked into the client bundle at build time.
ARG NEXT_PUBLIC_SERVER_URL=http://localhost:3000
ENV NEXT_PUBLIC_SERVER_URL=$NEXT_PUBLIC_SERVER_URL NEXT_TELEMETRY_DISABLED=1
RUN pnpm --filter web run build \
    && find apps/web/.next -name '*.map' -delete

FROM node:24-alpine
RUN apk add --no-cache libc6-compat \
    && rm -rf /usr/local/lib/node_modules /usr/local/bin/npm /usr/local/bin/npx /usr/local/bin/corepack /usr/local/bin/yarn /usr/local/bin/yarnpkg
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3001 HOSTNAME=0.0.0.0
COPY --from=build --chown=node:node /app/apps/web/.next/standalone ./
COPY --from=build --chown=node:node /app/apps/web/.next/static ./apps/web/.next/static
COPY --from=build --chown=node:node /app/apps/web/public ./apps/web/public
USER node
EXPOSE 3001
CMD ["node", "apps/web/server.js"]
