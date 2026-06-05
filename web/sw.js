// Minimal service worker — makes the app installable and caches the local shell.
// Cross-origin assets (three.js CDN, the avatar GLB) always go to the network.
const CACHE = "emily-avatar-v2";
const SHELL = [
  "./",
  "./index.html",
  "./style.css",
  "./main.js",
  "./lipsync.js",
  "./manifest.webmanifest",
  "./icon.svg",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Network-first for our own files: always serve the latest when online, fall
// back to cache when offline. (Cache-first would pin stale code across edits.)
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return; // let CDN/GLB hit the network
  if (e.request.method !== "GET") return;     // POST etc. go straight to network
  e.respondWith(
    fetch(e.request)
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy));
        return res;
      })
      .catch(() => caches.match(e.request))
  );
});
