---
permalink: /examples/service-worker.js
---
// The demos moved from /examples/ to /demos/, and the old pages now redirect to /demos/. This
// replaces the service worker that earlier visits registered here: it unregisters itself and
// reloads open pages, so they get the redirect instead of a cached copy of the old demo.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => {
  event.waitUntil(
    self.registration.unregister()
      .then(() => self.clients.matchAll({ type: "window" }))
      .then((clients) => clients.forEach((client) => client.navigate(client.url)))
  );
});
