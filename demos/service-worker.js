// Offline support for all demos under /demos/. Large files (the models on Hugging Face, videos) are
// cached on first use and then served from the cache, once for the whole site: a changed large file
// needs a new URL. Everything else is fetched from the network first, so changes to the pages reach
// visitors immediately, and served from the cache when offline.
const CACHE_NAME = "demos-v1";
const LARGE = /\.(onnx|mp4)$/;

self.addEventListener("install", () => self.skipWaiting());

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
      .then(() => self.clients.claim())
  );
});

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok) (await caches.open(CACHE_NAME)).put(request, response.clone());
  return response;
}

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) (await caches.open(CACHE_NAME)).put(request, response.clone());
    return response;
  } catch (err) {
    const cached = await caches.match(request);
    if (cached) return cached;
    throw err;
  }
}

const MODEL_HOST = "huggingface.co";   // the models (see models/models.js)

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET") return;
  if (LARGE.test(url.pathname) && (url.origin === location.origin || url.hostname === MODEL_HOST)) {
    event.respondWith(cacheFirst(event.request));
  } else if (url.origin === location.origin) {
    event.respondWith(networkFirst(event.request));
  }
});
