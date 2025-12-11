#!/bin/bash

# Start Universe server and Cloudflare tunnel

set -e

echo "=== Starting Universe ==="

# Defaults
PORT="${PORT:-7878}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FPS="${FPS:-30}"
UNIVERSE_DIR="${UNIVERSE_DIR:-universe}"

# Check if release build exists
if [ ! -f "target/release/universe" ]; then
    echo "Release build not found. Run ./deploy.sh first."
    exit 1
fi

# Start server in background
echo "Starting Universe server on port ${PORT}..."
./target/release/universe serve \
    --port "${PORT}" \
    --universe "${UNIVERSE_DIR}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --fps "${FPS}" &

SERVER_PID=$!

# Wait for server to start
sleep 3

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Error: Server failed to start"
    exit 1
fi

echo "✓ Server started (PID: $SERVER_PID)"

# Check if tunnel is configured
CFG="${HOME}/.cloudflared/config.yml"
if ! command -v cloudflared >/dev/null 2>&1; then
    echo ""
    echo "Warning: cloudflared is not installed."
    echo "Install it (or run ./cloudflare/setup.sh) to enable universe.too.foo."
    echo ""
    echo "Server is running locally at: http://localhost:${PORT}"
    echo ""
    echo "Press Ctrl+C to stop"
    trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM
    wait $SERVER_PID
    exit 0
fi

if [ ! -f "${CFG}" ]; then
    echo ""
    echo "Warning: Cloudflare tunnel not configured."
    echo "Missing: ${CFG}"
    echo "Run ./cloudflare/setup.sh to set up the tunnel (interactive), or install a config+credentials/token."
    echo ""
    echo "Server is running locally at: http://localhost:${PORT}"
    echo ""
    echo "Press Ctrl+C to stop"

    # Wait for server only
    trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM
    wait $SERVER_PID
    exit 0
fi

# Start tunnel
echo "Starting Cloudflare tunnel..."
cloudflared --config "${CFG}" --no-autoupdate tunnel run &

TUNNEL_PID=$!

echo "✓ Tunnel started (PID: $TUNNEL_PID)"

echo ""
echo "=== Universe is Running ==="
echo ""
echo "  Local:  http://localhost:${PORT}"
echo "  Public: https://universe.too.foo"
echo ""
echo "  Server PID: $SERVER_PID"
echo "  Tunnel PID: $TUNNEL_PID"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Handle shutdown
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $SERVER_PID 2>/dev/null || true
    kill $TUNNEL_PID 2>/dev/null || true
    echo "Stopped."
    exit 0
}

trap cleanup INT TERM

# Wait for processes
wait
