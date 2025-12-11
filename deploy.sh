#!/bin/bash

# Universe Deployment Script

set -e

echo "=== Universe Deployment ==="

# Build Rust server (release mode)
echo ""
echo "[1/3] Building server (release mode)..."
cargo build --release -p universe-cli

# Build TypeScript client
echo ""
echo "[2/3] Building client..."
cd client

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing client dependencies..."
    npm install
fi

# Build client
npm run build

cd ..

# Verify builds
echo ""
echo "[3/3] Verifying builds..."

if [ ! -f "target/release/universe" ]; then
    echo "Error: Server binary not found"
    exit 1
fi

if [ ! -d "client/dist" ]; then
    echo "Error: Client dist directory not found"
    exit 1
fi

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Server binary: ./target/release/universe"
echo "Client files: ./client/dist/"
echo ""
echo "To run locally:"
echo "  ./target/release/universe serve"
echo "  Then open http://localhost:7878"
echo ""
echo "To deploy with Cloudflare Tunnel:"
echo "  1. Run: ./cloudflare/setup.sh (first time only)"
echo "  2. Run: ./run.sh"
echo "  3. Access at: https://universe.too.foo"
echo ""
