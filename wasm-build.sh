#!/bin/bash
# Phase 2.1: Build WASM target for universe-engine

set -e

echo "Building WASM target for universe-engine..."

# Install wasm-pack if not present
if ! command -v wasm-pack &> /dev/null; then
    echo "Installing wasm-pack..."
    cargo install wasm-pack
fi

cd crates/universe-engine
wasm-pack build --target web --out-dir ../../client/wasm

echo "WASM build complete! Output: client/wasm/"
