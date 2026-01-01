# HELIOS Universe Planetarium - Browser Client

Interactive heliocentric universe visualization with neural-compressed star data, time travel (±100,000 years), and multi-scale navigation from planetary surfaces to intergalactic distances.

**Author:** Shivam Bhardwaj
**Version:** 1.1 (January 2025)

## Quick Start

### Development (with server running locally)

```bash
# Terminal 1: Start the server
cargo run -p universe-cli -- serve

# Terminal 2: Start the client dev server
cd client
npm install
npm run dev

# Open http://localhost:3000
```

### Production Build

```bash
cd client
npm install
npm run build

# Output will be in client/dist/
```

## Features

### ✅ NASA Eyes-Style Anchor Navigation
- **Per-anchor local bubbles**: Each landmark (Sun, Earth, Andromeda) is a local reference frame
- **Zoom-out limit**: Can't zoom out infinitely - max distance keeps anchor's system ≥10% of screen
- **Warp between anchors**: Search → select → jump → anchor switches on arrival
- **Smart prefetch**: Target cells load during jump animation (reduced buffering)
- **Breadcrumb HUD**: Shows `Anchor Name • Distance from Anchor • Distance to Sun`

### ✅ Expanded Landmark Catalog (100+ objects)
- **9 Planets**: Sun through Neptune
- **7 Dwarf Planets/KBOs**: Pluto, Eris, Makemake, Haumea, Sedna, Gonggong, Quaoar
- **5 Spacecraft**: Voyager 1/2, New Horizons, JWST, Parker Solar Probe
- **~30 Bright Stars**: Sirius, Betelgeuse, Vega, Arcturus, Rigel, Deneb, Canopus, Alpha Centauri, Proxima Centauri, and more
- **~40 Messier Objects**:
  - **Galaxies**: Andromeda (M31), Whirlpool (M51), Sombrero (M104), Triangulum (M33), and more
  - **Nebulae**: Orion (M42), Eagle (M16), Lagoon (M8), Ring (M57), Crab (M1), and more
  - **Clusters**: Pleiades (M45), Hercules (M13), Beehive (M44), and more
- **Infrastructure**: Supports loading additional landmarks from `/universe/landmarks.json` (500+ possible)

### ✅ Phase 3: Time Travel (±100,000 years)
- **Time slider**: Scrub through 200,000 years instantly
- **Jump buttons**: ±100 years, ±1,000 years
- **Playback controls**: Play/Pause, speed multipliers (0.5×, 2×)
- **Date range**: 98,000 BC to 102,000 AD
- **Live display**: Shows offset from J2000 epoch

### 🔮 Coming Soon
- **Proper motion**: Stars move over millennia
- **Spacecraft orbits**: Time-dependent trajectories
- **Oort Cloud**: Procedural 1000+ objects beyond 2000 AU
- **Milky Way shader**: Spiral arms visible when zoomed out
- **Multi-spectrum**: Visible, IR, X-ray, Radio views

## Controls

### Movement (Anchor-Based Navigation)
| Input | Action |
|-------|--------|
| **WASD** | Move camera (clamped to current anchor's local bubble) |
| **Scroll wheel** | Forward/backward thrust |
| **Mouse drag** | Look around |
| **Space** | Move up |
| **Shift** | Move down |
| **Q** | Decrease speed multiplier |
| **E** | Increase speed multiplier |
| **H** | Jump to Sun (resets anchor to Sun) |

**Navigation Model:**
- You explore within a **local bubble** around the current anchor (Sun, Earth, Andromeda, etc.)
- Max distance is computed so the anchor's system stays visible (≥10% of screen height)
- To explore distant regions, **warp to a landmark** (search → click → auto-switch anchor)

### Landmark Search & Warping
| Input | Action |
|-------|--------|
| **Search panel** (top-right) | Type landmark name (e.g., "Andromeda", "Orion", "M13") |
| **Click result** | Initiate warp to landmark (2-10s animation) |
| **Click object** (desktop) | Warp to clicked planet/spacecraft |
| **Tap object** (mobile) | Warp to tapped object |
| **Long-press** (mobile) | Instant teleport to object |

**During warp:**
- Camera auto-rotates to face target
- Target cells prefetch (reduces buffering on arrival)
- Anchor automatically switches when you arrive

### Quick Navigation
| Key | Destination |
|-----|-------------|
| **0** | Solar system overview (1.5 AU from Sun) |
| **1** | Earth (1 AU from Sun) |
| **2** | Mars (1.5 AU from Sun) |
| **3** | Jupiter (5.2 AU from Sun) |
| **4** | Saturn (9.5 AU from Sun) |
| **5** | Uranus (19.2 AU from Sun) |
| **6** | Neptune (30 AU from Sun) |

**Note:** Pressing these keys updates position but does **not** change the anchor (anchor switches only on warp completion).

### Time Controls (UI Panel)
- **Slider**: Scrub -100,000 to +100,000 years
- **Buttons**: -1000y, -100y, RESET, +100y, +1000y
- **Playback**: PLAY/PAUSE, 0.5×, 2× speed

## Architecture

```
Browser Client (TypeScript/Vite)
    │
    ├─ /stream  (WebSocket) ──> MJPEG frames
    │
    └─ /control (WebSocket) ──> Input events
                          <──── Server state
```

## Files

- `index.html` - Main HTML with UI overlay
- `src/main.ts` - Application entry point
- `src/client.ts` - WebSocket client for streaming
- `src/input.ts` - Keyboard/mouse input handling
- `src/hud.ts` - HUD overlay with time/position display
- `vite.config.ts` - Vite configuration with proxy
- `package.json` - Dependencies and scripts
- `tsconfig.json` - TypeScript configuration

## Useful query params

- `?engine=auto|wasm|ts`: cell loader selection (auto tries WASM and falls back to TS)
- `?debugOverlay=1` (or `?debug=1`): show debug overlay (renderer/loader/cell stats/FPS)
