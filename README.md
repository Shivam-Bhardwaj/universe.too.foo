# Universe (HELIOS)

Server-rendered, astronomical-scale visualization with a streamable “universe dataset” (HLG cells) and a browser client.

## Neural Planetarium (MPCORB real asteroid belt)

This repo also includes a standalone **desktop** renderer (crate `neural-planetarium`, run from repo root) that loads the Minor Planet Center orbit catalog (MPCORB) and renders:
- **Eros** (instance 0, neural deformation)
- **Main belt** (MPCORB elements)
- **Jupiter** (simple mean-elements approximation for resonance visualization)

### Build the real belt assets (local)

The large MPCORB inputs/outputs are **not committed** (they live under `assets/` which is gitignored).

1) Download MPCORB (~200–300MB):

```bash
curl -L --fail -o assets/MPCORB.DAT "https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT"
```

2) Compile a belt binary (defaults to 100k asteroids):

```bash
python3 compile_belt.py
```

This writes `assets/real_belt.bin` in the same binary layout as the Rust `KeplerParams` struct.

3) Run the desktop app:

```bash
cargo run --release
```

You should see the belt with **Kirkwood gaps** (resonances) and non-spherical “rocky” bodies (cheap per-instance deformation for the belt).

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

If you don't have a catalog yet, you can build synthetic stars instead:

```bash
cargo run -p universe-cli -- build --synthetic 50000 --output universe
```

### Real Gaia DR3 POC (with CUDA training)

For a real-data proof of concept using Gaia DR3:

```bash
# 1. Fetch bright stars from Gaia Archive
curl -G "https://gea.esac.esa.int/tap-server/tap/sync" \
  --data-urlencode "REQUEST=doQuery" \
  --data-urlencode "LANG=ADQL" \
  --data-urlencode "FORMAT=csv" \
  --data-urlencode "QUERY=SELECT TOP 2000 source_id,ra,dec,parallax,phot_g_mean_mag,bp_rp FROM gaiadr3.gaia_source WHERE phot_g_mean_mag < 10 AND parallax > 0" \
  -o data/gaia_poc.csv

# 2. Build universe from Gaia CSV
cargo run --release -p universe-cli -- build \
  --stars data/gaia_poc.csv \
  --max-mag 10 \
  --limit 2000 \
  --output universe_gaia_poc

# 3. Train with tch-rs CUDA backend (requires torch feature)
# Note: LD_PRELOAD forces CUDA library loading
export LIBTORCH_USE_PYTORCH=1
LD_PRELOAD="$(python3 -c 'import torch; print(torch.__path__[0])')/lib/libtorch_cuda.so" \
  cargo run --release --features torch -p universe-cli -- train-all \
    --input universe_gaia_poc \
    --output universe_gaia_poc_trained \
    --iterations 200 \
    --backend torch-cuda
```

### 2) Run the server

```bash
cargo run -p universe-cli -- serve --universe universe --port 7878 --width 1280 --height 720 --fps 30
```

Endpoints:
- **UI**: `http://localhost:7878/` (serves `client/dist` if present)
- **Dataset**: `http://localhost:7878/universe/` (serves preprocessed dataset files)

**Debug endpoints (dev only):**
- **Legacy MJPEG viewer**: `http://localhost:7878/mjpeg`
- **Video stream** (debug only, use `?mode=stream&debug=1` in client):
  - H.264: `ws://localhost:7878/stream?codec=h264`
  - MJPEG: `ws://localhost:7878/stream`
- **Control**: `ws://localhost:7878/control` (use `?registered=1` to enable 5× jump budget)

### 3) Run the browser client (dev)

```bash
cd client
npm install
npm run dev
```

Open `http://localhost:3000` (Vite proxies `/stream` and `/control` to the server).

## Runtime Requirements

**Production (recommended):**
- **WebGPU**: Chrome 113+, Edge 113+, Safari 18+, or Firefox 110+ (experimental flag)
- **WebGL2**: Fallback for older browsers (reduced performance, not recommended)

**Debug mode:**
- Pixel streaming (`?mode=stream&debug=1`) requires WebCodecs API support (Chrome/Edge)

The client automatically detects your browser and shows appropriate warnings if requirements aren't met.

## Deployment (universe.too.foo)

```bash
./deploy.sh   # builds Rust (release) + builds client into client/dist
./run.sh      # runs server on :7878 and starts cloudflared tunnel if configured
```

Cloudflare tunnel routes are in `cloudflare/config.yml`.

## Client query params (useful for debugging)

- **Renderer / loader selection**:
  - `?engine=auto` (default): try WASM cell loader, fall back to TypeScript if init fails
  - `?engine=wasm`: force WASM (fails fast if WASM can’t init)
  - `?engine=ts`: force TypeScript loader
- **Debug overlay**:
  - `?debugOverlay=1` (or `?debug=1`): shows renderer, loader, cell cache stats, FPS
- **Stream mode (debug-only)**:
  - `?mode=stream&debug=1`

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
- **Client** (production): Fetches dataset → LZ4 decompress → WebGPU/WebGL2 render (no server rendering)
- **Server** (debug only): WGPU render → capture → **NVENC H.264** via ffmpeg → WebSocket broadcast (for debugging)
