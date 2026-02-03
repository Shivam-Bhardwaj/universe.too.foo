// Service worker: tile caching + simple eviction
//
// Scope: cache only /universe/index.json and /universe/cells/*.bin

const CACHE_NAME = 'universe-tiles-v1';
const MAX_ENTRIES = 2000;

self.addEventListener('install', (event) => {
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        (async () => {
            const keys = await caches.keys();
            await Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)));
            await self.clients.claim();
        })(),
    );
});

function isUniverseTileRequest(url) {
    return url.pathname === '/universe/index.json' || url.pathname.startsWith('/universe/cells/');
}

self.addEventListener('fetch', (event) => {
    const req = event.request;
    if (req.method !== 'GET') return;

    const url = new URL(req.url);
    if (!isUniverseTileRequest(url)) return;

    event.respondWith(cacheFirst(req));
});

async function cacheFirst(request) {
    const cache = await caches.open(CACHE_NAME);

    const cached = await cache.match(request);
    if (cached) return cached;

    const res = await fetch(request);
    if (res.ok) {
        await cache.put(request, res.clone());
        await evictIfNeeded(cache);
    }
    return res;
}

async function evictIfNeeded(cache) {
    const keys = await cache.keys();
    const extra = keys.length - MAX_ENTRIES;
    if (extra <= 0) return;

    // Evict oldest entries (cache.keys() order is insertion-ish; good enough baseline).
    await Promise.all(keys.slice(0, extra).map((k) => cache.delete(k)));
}




