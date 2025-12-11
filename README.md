# Universe (HELIOS)

Server-rendered, astronomical-scale visualization with a streamable “universe dataset” (HLG cells) and a browser client.

## Quickstart (local)

### 1) Build a non-empty universe dataset

The runtime expects a directory with:
- `index.json` (manifest)
- `cells/*.bin` (LZ4-compressed cell files)

If you have a Gaia-style CSV (columns: `source_id,ra,dec,parallax,phot_g_mean_mag,bp_rp`), build a small but visible dataset:

```bash
cd /home/curious/HELIOS

# Builds planets from DE440 (auto-download) + stars from the catalog
cargo run -p universe-cli -- build \
  --stars "/path/to/gaia_stars.csv" \
  --max-mag 8 \
  --limit 200000 \
  --ephemeris-dir data/ephemeris \
  --output universe
```

Note: this repo currently includes a sample Gaia CSV at `legacy assets/gaia_stars.csv` (may be removed later).

If you don’t have a catalog yet, you can build synthetic stars instead:

```bash
cargo run -p universe-cli -- build --synthetic 50000 --output universe
```

### 2) Run the server

```bash
cargo run -p universe-cli -- serve --universe universe --port 7878 --width 1280 --height 720 --fps 30
```

Endpoints:
- **UI**: `http://localhost:7878/` (serves `client/dist` if present)
- **Legacy MJPEG viewer**: `http://localhost:7878/mjpeg`
- **Video stream**:
  - H.264 (preferred): `ws://localhost:7878/stream?codec=h264`
  - MJPEG fallback: `ws://localhost:7878/stream`
- **Control**: `ws://localhost:7878/control` (use `?registered=1` to enable 5× jump budget)

### 3) Run the browser client (dev)

```bash
cd client
npm install
npm run dev
```

Open `http://localhost:3000` (Vite proxies `/stream` and `/control` to the server).

## Deployment (universe.too.foo)

```bash
./deploy.sh   # builds Rust (release) + builds client into client/dist
./run.sh      # runs server on :7878 and starts cloudflared tunnel if configured
```

Cloudflare tunnel routes are in `cloudflare/config.yml`.

## Controls (browser)

- **Click**: capture mouse
- **WASD**: move
- **Mouse**: look
- **Space / Shift**: up/down
- **Q / E**: slow/fast movement
- **0–6**: jump presets (teleport triggers “buffering” overlay)

## Jump budgeting (v1)

- **Guests**: 5 jumps max, refills 1 jump / 30 seconds
- **Registered scaffold**: 5× capacity + refill (enable with `?registered=1` or set `localStorage.universe_registered=1`)

## Architecture (high level)

- **Dataset** (`universe/`): HLG cell files + manifest (`index.json`) produced by `universe-cli build`
- **Server**: WGPU render → capture → **NVENC H.264** via ffmpeg (v1) → WebSocket broadcast
- **Client**: WebCodecs decode to canvas (fallback to MJPEG)
