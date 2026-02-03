#!/bin/bash
# Build complete HELIOS universe dataset
#
# Usage: ./scripts/build_universe.sh [STAR_COUNT] [--train] [--deploy]
#
# Examples:
#   ./scripts/build_universe.sh 100000              # Build 100k star universe
#   ./scripts/build_universe.sh 100000 --deploy     # Build and deploy to client
#   ./scripts/build_universe.sh 100000 --train      # Build and train (takes hours)
#
# Note: Training is optional. Untrained data has correct 3D positions
# and looks good without training. Training optimizes splat sizes.

set -e

STAR_COUNT=${1:-100000}
DO_TRAIN=false
DO_DEPLOY=false

# Parse flags
for arg in "$@"; do
    case $arg in
        --train) DO_TRAIN=true ;;
        --deploy) DO_DEPLOY=true ;;
    esac
done

DATA_DIR="data"
UNIVERSE_NAME="universe_gaia_${STAR_COUNT}"
CSV_FILE="${DATA_DIR}/gaia_${STAR_COUNT}.csv"
DEEP_SKY_FILE="${DATA_DIR}/deep_sky_objects.json"

echo "========================================"
echo "HELIOS Universe Builder"
echo "========================================"
echo "Stars:    $STAR_COUNT"
echo "Train:    $DO_TRAIN"
echo "Deploy:   $DO_DEPLOY"
echo "========================================"

# Step 1: Download Gaia data if needed
if [ ! -f "$CSV_FILE" ]; then
    echo ""
    echo "[1/5] Downloading Gaia DR3 data..."
    ./scripts/download_gaia.sh "$STAR_COUNT"
else
    echo ""
    echo "[1/5] Using existing Gaia data: $CSV_FILE"
fi

# Step 2: Build universe from stars
echo ""
echo "[2/5] Building universe dataset..."
cargo run --release -p universe-cli -- build \
  --stars "$CSV_FILE" \
  --limit "$STAR_COUNT" \
  --max-mag 15.0 \
  --output "$UNIVERSE_NAME"

# Step 3: Generate star landmarks
echo ""
echo "[3/5] Generating star landmarks..."
cargo run --release -p universe-cli -- generate-landmarks \
  --universe "$UNIVERSE_NAME" \
  --stars "$CSV_FILE" \
  --count 50 \
  --output "${UNIVERSE_NAME}/landmarks.json"

# Step 4: Add deep sky objects to landmarks
if [ -f "$DEEP_SKY_FILE" ]; then
    echo ""
    echo "[4/5] Adding deep sky objects..."
    python3 scripts/build_deep_sky.py
else
    echo ""
    echo "[4/5] Skipping deep sky objects (file not found: $DEEP_SKY_FILE)"
fi

# Step 5: Optional training
if [ "$DO_TRAIN" = true ]; then
    echo ""
    echo "[5/5] Training (this will take a long time)..."
    TRAINED_NAME="${UNIVERSE_NAME}_trained"

    cargo run --release -p universe-cli -- train \
      --input "$UNIVERSE_NAME" \
      --output "$TRAINED_NAME" \
      --iterations 300 \
      --backend wgpu \
      --lambda-isotropy 0.1 \
      --lambda-collapse 0.2 \
      --min-scale-ratio 0.2

    FINAL_UNIVERSE="$TRAINED_NAME"
else
    echo ""
    echo "[5/5] Skipping training (use --train to enable)"
    FINAL_UNIVERSE="$UNIVERSE_NAME"
fi

# Deploy to client if requested
if [ "$DO_DEPLOY" = true ]; then
    echo ""
    echo "Deploying to client..."
    rm -rf client/public/universe
    cp -r "$FINAL_UNIVERSE" client/public/universe

    # Ensure deep sky landmarks are included
    if [ -f "${UNIVERSE_NAME}/landmarks.json" ]; then
        cp "${UNIVERSE_NAME}/landmarks.json" client/public/universe/
    fi

    echo "Deployed to client/public/universe"
fi

echo ""
echo "========================================"
echo "Build complete!"
echo "========================================"
echo ""
echo "Universe: $FINAL_UNIVERSE"
echo ""
if [ "$DO_DEPLOY" = false ]; then
    echo "To deploy: ./scripts/build_universe.sh $STAR_COUNT --deploy"
fi
echo "To view:   cd client && npm run dev"
echo "           Open http://localhost:3000"
