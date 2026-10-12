// Replaced by /examples/service-worker.js, which serves all demos: this removes the older registration
// that visitors of the LaFAM demo may still have.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", () => self.registration.unregister());
