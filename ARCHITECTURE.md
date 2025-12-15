# HELIOS Architecture

## Phase 0: Single-Path Engine Foundation

### Phase 0.3: Engine Language Boundary Decision

**Decision:** Engine core in Rust, shared between native and WASM builds.

#### Current State

- **Server-side rendering:** Rust (`universe-render`, `universe-raster`) using WGPU
- **Client-side rendering:** TypeScript/JavaScript (`client/src/webgpu_splats.ts`, `client/src/webgl_splats.ts`)
- **Data processing:** Rust (`universe-data`, `universe-core`)
- **Simulation:** Rust (`universe-sim`)

#### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Engine Core (Rust)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Rendering   │  │   Camera     │  │   Streaming  │     │
│  │  Pipeline    │  │   Math       │  │   Policy     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  Compiles to:                                                │
│  • Native binary (reference viewer)                         │
│  • WASM module (web viewer)                                  │
└─────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌──────────────────┐
│  Native Viewer  │          │   Web Viewer     │
│  (Rust + WGPU)  │          │  (WASM + WebGPU)  │
│                 │          │                   │
│  • File I/O     │          │  • HTTP fetch    │
│  • Native window│          │  • Canvas API     │
└─────────────────┘          └──────────────────┘
```

#### Implementation Plan

1. **Extract shared engine crate** (`universe-engine`):
   - Camera math (floating origin, log depth)
   - Rendering pipeline (Gaussian splat rendering)
   - Tile streaming policy (LRU, selection)
   - Data structures (Cell, Splat, etc.)

2. **Platform adapters:**
   - **Native:** Direct WGPU device/queue, file I/O
   - **WASM:** WebGPU bindings via `wasm-bindgen`, HTTP fetch adapter

3. **Shared code constraints:**
   - No platform-specific APIs in core
   - Use trait abstractions for I/O (`Read`, `Write`, async streams)
   - Use `cfg(target_arch = "wasm32")` for platform-specific code paths

#### Benefits

- **Single source of truth:** Same rendering logic in native and web
- **Performance:** Rust performance in both contexts
- **Maintainability:** One codebase for rendering engine
- **Testing:** Native viewer can test engine logic without browser

#### Migration Path

1. **Phase 1:** Build native reference viewer using existing Rust crates
2. **Phase 2:** Extract shared engine crate, compile to WASM
3. **Phase 3:** Replace TypeScript renderer with WASM engine

---

## Runtime Requirements (Phase 0.2)

**Production:**
- WebGPU: Chrome 113+, Edge 113+, Safari 18+, Firefox 110+ (experimental)
- WebGL2: Fallback (not recommended for production)

**Debug:**
- Pixel streaming: WebCodecs API (Chrome/Edge)

---

## Current Module Structure

```
crates/
├── universe-core/      # Core data structures (CellId, HLG, coordinates)
├── universe-data/      # Data pipeline (CSV → cells, LZ4 compression)
├── universe-render/    # Server-side WGPU renderer
├── universe-raster/    # Tile-based rasterization (shared shaders)
├── universe-sim/       # Orbital mechanics simulation
├── universe-stream/    # WebSocket server (debug only)
└── universe-train/     # ML training (Burn framework)

client/
└── src/
    ├── webgpu_splats.ts    # WebGPU renderer (to be replaced by WASM)
    ├── webgl_splats.ts     # WebGL2 fallback (to be replaced by WASM)
    ├── camera.ts            # Camera math (to be replaced by WASM)
    ├── dataset.ts          # Dataset loader (to be replaced by WASM)
    └── runtime.ts           # Runtime detection (stays in TS)
```
