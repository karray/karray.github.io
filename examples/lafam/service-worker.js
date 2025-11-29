const CACHE_NAME = "cache-v2.1";
// const urlsToCache = [];

// self.addEventListener("install", (event) => {
//   event.waitUntil(
//     caches.open(CACHE_NAME).then((cache) => {
//       // console.log("Opened cache");
//       return cache.addAll(urlsToCache);
//     })
//   );
// });

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keyList => {
      return Promise.all(keyList.map(key => {
        if (key !== CACHE_NAME) {
          return caches.delete(key);
        }
      }));
    })
  );
});

self.addEventListener('message', event => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

self.addEventListener("fetch", (event) => {
  if (!event.request.url.startsWith("http")) return;

  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) return cachedResponse;

      return fetch(event.request).then((response) => {
        if (!response || response.status !== 200 || response.type !== "basic") {
          return response;
        }

        const responseClone = response.clone();
        const contentLength = response.headers.get('Content-Length');

        if (response.body && contentLength) {
          const total = parseInt(contentLength, 10);
          let loaded = 0;

          const reader = response.body.getReader();
          const stream = new ReadableStream({
            async start(controller) {
              while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                loaded += value.byteLength;
                
                const rawPercentage = (loaded / total) * 100;
                // dut to compression, the progress may exceed 100%
                const progress = Math.min(100, Math.round(rawPercentage));
                
                const fileName = event.request.url.split('/').pop();
                reportProgress(fileName, progress);
                
                controller.enqueue(value);
              }
              controller.close();
            }
          });

          const newHeaders = new Headers(response.headers);
          newHeaders.delete('Content-Length');

          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, new Response(stream, { 
              headers: newHeaders 
            }));
          });

          return responseClone;
        } else {
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseClone);
          });
          return response;
        }
      });
    })
  );
});

function reportProgress(name, progress) {
  self.clients.matchAll().then(clients => {
    clients.forEach(client => {
      client.postMessage({
        type: 'progress',
        name: name,
        progress: progress
      });
    });
  });
}
