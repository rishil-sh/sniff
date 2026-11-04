# API server, compiled to one Bun executable at image build time. Final stage is Alpine plus
# that binary, the libSQL native addon, and the SQL migrations. Build context: repo root.

FROM node:24-alpine AS base
RUN apk add --no-cache libc6-compat \
    && npm install -g pnpm@12.3.4 turbo@2
COPY --from=oven/bun:1-alpine /usr/local/bin/bun /usr/local/bin/bun
WORKDIR /app

# Keep only the server and the workspace packages it depends on.
FROM base AS prune
COPY . .
RUN turbo prune server --docker

FROM base AS build
COPY --from=prune /app/out/json/ .
RUN --mount=type=cache,target=/pnpm-store pnpm install --frozen-lockfile --ignore-scripts --store-dir /pnpm-store
COPY --from=prune /app/out/full/ .
WORKDIR /app/apps/server
# Bytecode is compiled now, so the container starts the binary directly.
RUN bun build ./src/index.ts --compile --bytecode --format=esm --minify --no-compile-autoload-dotenv --outfile /out/server \
    && mkdir -p /out/node_modules/@libsql \
    && cp -aL /app/node_modules/.pnpm/@libsql+linux-*-musl@*/node_modules/@libsql/linux-*-musl /out/node_modules/@libsql/

FROM alpine:3.22
# The Bun runtime inside the binary links against libstdc++. BusyBox wget is already in the base image.
RUN apk add --no-cache libstdc++ libgcc \
    && adduser -D -u 1000 sniff \
    && mkdir /data && chown sniff /data
WORKDIR /app
COPY --from=build --chown=sniff:sniff /out/ ./
COPY --chown=sniff:sniff packages/db/src/migrations ./migrations
COPY --chown=sniff:sniff apps/server/docker-entrypoint.sh ./docker-entrypoint.sh
USER sniff
ENV NODE_ENV=production PORT=3000 MIGRATIONS_DIR=/app/migrations
VOLUME /data
EXPOSE 3000
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD wget -qO- http://localhost:3000/ >/dev/null || exit 1
ENTRYPOINT ["sh", "docker-entrypoint.sh"]
