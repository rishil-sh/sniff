// Bump this version when changing offline.html so installed apps refresh it.
const CACHE_NAME = "sniff-offline-v3";
const OFFLINE_URL = "/offline.html";

const install = async () => {
  const cache = await caches.open(CACHE_NAME);
  await cache.add(new Request(OFFLINE_URL, { cache: "reload" }));
  await self.skipWaiting();
};

const activate = async () => {
  const keys = await caches.keys();
  await Promise.all(
    keys
      .filter((key) => key.startsWith("sniff-offline-") && key !== CACHE_NAME)
      .map((key) => caches.delete(key))
  );
  await self.clients.claim();
};

// Never cache authenticated HTML or API responses. Server-rendered pages need
// a connection; a precached page keeps offline navigation understandable.
const navigate = async (request) => {
  try {
    return await fetch(request);
  } catch {
    return (await caches.match(OFFLINE_URL)) ?? Response.error();
  }
};

self.addEventListener("install", (event) => event.waitUntil(install()));
self.addEventListener("activate", (event) => event.waitUntil(activate()));
self.addEventListener("fetch", (event) => {
  if (event.request.mode === "navigate" && event.request.method === "GET") {
    event.respondWith(navigate(event.request));
  }
});
