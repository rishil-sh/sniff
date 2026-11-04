/**
 * The API server's base URL. Browsers use the public URL; the Next.js server prefers SERVER_URL when set, because in
 * Docker "localhost" inside the web container is not the API (there it is http://server:3000).
 */
export const serverUrl = (): string => {
  const internal =
    typeof window === "undefined" ? process.env.SERVER_URL : undefined;
  const url = internal ?? process.env.NEXT_PUBLIC_SERVER_URL;
  if (!url) {
    throw new Error(
      "NEXT_PUBLIC_SERVER_URL is not set. Copy apps/web/.env.example to apps/web/.env."
    );
  }
  return url.replace(/\/$/u, "");
};
