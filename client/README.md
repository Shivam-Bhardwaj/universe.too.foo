# Universe Browser Client

TypeScript/Vite browser client for the Universe Universe Visualization system.

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

## Controls

| Input | Action |
|-------|--------|
| **Click** | Capture mouse for controls |
| **WASD** | Move camera |
| **Mouse** | Look around |
| **Space** | Move up |
| **Shift** | Move down |
| **Q** | Slow movement (0.1x) |
| **E** | Fast movement (10x) |
| **P** | Pause/Resume time |
| **,** | Slow down time (0.5x) |
| **.** | Speed up time (2x) |
| **1** | Jump to Earth |
| **2** | Jump to Mars |
| **3** | Jump to Jupiter |
| **4** | Jump to Saturn |
| **5** | Jump to Uranus |
| **6** | Jump to Neptune |
| **0** | Solar system overview |
| **Esc** | Release mouse |

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
