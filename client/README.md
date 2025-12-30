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

### ✅ Phase 1: Navigation & Scale System
- **Regime-aware speed scaling**: Auto-adjusts from planetary (km/s) to intergalactic (Mpc/s)
- **5 scale regimes**: Planetary, Solar System, Interstellar, Galactic, Intergalactic
- **Zoom limits**: 1 km minimum to ~300 Mpc maximum
- **Breadcrumb display**: Always know your location context
- **H key**: Instant jump to Sun

### ✅ Phase 2: Object Catalog (49 objects)
- **9 Planets**: Sun through Neptune
- **5 Spacecraft**: Voyager 1 (~164 AU), Voyager 2 (~137 AU), New Horizons (~58 AU), JWST (L2), Parker Solar Probe
- **7 Dwarf Planets/KBOs**: Pluto, Eris, Makemake, Haumea, Sedna, Gonggong, Quaoar
- **13 Messier Objects**: M8, M13, M20, M27, M33, M44, M51, M57, M64, M81, M87, M101, M104
- **Famous objects**: Andromeda, Magellanic Clouds, Orion Nebula, Crab Nebula, Pleiades, Sirius, Betelgeuse

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

### Movement
| Input | Action |
|-------|--------|
| **WASD** | Move camera (speed auto-scales by distance) |
| **Mouse drag** | Look around |
| **Space** | Move up |
| **Shift** | Move down |
| **Q** | Decrease speed multiplier |
| **E** | Increase speed multiplier |
| **H** | Jump to Sun (Home) |

### Quick Navigation
| Key | Destination |
|-----|-------------|
| **0** | Solar system overview (1.5 AU) |
| **1** | Earth (1 AU) |
| **2** | Mars (1.5 AU) |
| **3** | Jupiter (5.2 AU) |
| **4** | Saturn (9.5 AU) |
| **5** | Uranus (19.2 AU) |
| **6** | Neptune (30 AU) |

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
