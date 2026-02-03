#!/bin/bash
# Fast Gaia training - trains only landmark neighborhoods
# Much faster than full training, good for testing
#
# Usage: ./scripts/train_gaia_fast.sh [STAR_COUNT] [ITERATIONS] [NEIGHBORS]
#
# Examples:
#   ./scripts/train_gaia_fast.sh 50000 200 1    # Quick test (~2-5 min)
#   ./scripts/train_gaia_fast.sh 100000 500 2   # Production (~15-30 min)

set -e

STAR_COUNT=${1:-50000}
ITERATIONS=${2:-200}
NEIGHBORS=${3:-1}
BACKEND=${4:-wgpu}

DATA_DIR="data"
UNIVERSE_NAME="universe_gaia_${STAR_COUNT}"
TRAINED_NAME="${UNIVERSE_NAME}_trained"
CSV_FILE="${DATA_DIR}/gaia_${STAR_COUNT}.csv"

echo "========================================"
echo "HELIOS Fast Gaia Training (Landmarks)"
echo "========================================"
echo "Stars:      $STAR_COUNT"
echo "Iterations: $ITERATIONS"
echo "Neighbors:  $NEIGHBORS"
echo "Backend:    $BACKEND"
echo "========================================"

# Step 1: Download Gaia data if not exists
if [ ! -f "$CSV_FILE" ]; then
    echo ""
    echo "[1/5] Downloading Gaia DR3 data ($STAR_COUNT stars)..."
    mkdir -p "$DATA_DIR"

    curl -G "https://gea.esac.esa.int/tap-server/tap/sync" \
      --data-urlencode "REQUEST=doQuery" \
      --data-urlencode "LANG=ADQL" \
      --data-urlencode "FORMAT=csv" \
      --data-urlencode "QUERY=SELECT TOP $STAR_COUNT source_id,ra,dec,parallax,phot_g_mean_mag,bp_rp FROM gaiadr3.gaia_source WHERE phot_g_mean_mag < 12 AND parallax > 0.5 AND parallax_over_error > 10 ORDER BY RANDOM_INDEX" \
      -o "$CSV_FILE"

    LINE_COUNT=$(wc -l < "$CSV_FILE")
    echo "Downloaded $((LINE_COUNT - 1)) stars"
else
    echo ""
    echo "[1/5] Using existing Gaia data: $CSV_FILE"
fi

# Step 2: Build universe
echo ""
echo "[2/5] Building universe dataset..."
cargo run --release -p universe-cli -- build \
  --stars "$CSV_FILE" \
  --limit "$STAR_COUNT" \
  --max-mag 12.0 \
  --output "$UNIVERSE_NAME"

# Step 3: Generate landmarks
echo ""
echo "[3/5] Generating landmarks..."
cargo run --release -p universe-cli -- generate-landmarks \
  --universe "$UNIVERSE_NAME" \
  --stars "$CSV_FILE" \
  --count 50 \
  --output "$UNIVERSE_NAME/landmarks.json"

# Step 4: Train landmark neighborhoods only
echo ""
echo "[4/5] Training landmark neighborhoods (fast)..."
cargo run --release -p universe-cli -- train-landmarks \
  --input "$UNIVERSE_NAME" \
  --output "$TRAINED_NAME" \
  --landmarks "$UNIVERSE_NAME/landmarks.json" \
  --neighbors "$NEIGHBORS" \
  --iterations "$ITERATIONS" \
  --backend "$BACKEND"

# Step 5: Copy to client
echo ""
echo "[5/5] Copying to client..."
rm -rf client/public/universe
cp -r "$TRAINED_NAME" client/public/universe
cp "$UNIVERSE_NAME/landmarks.json" client/public/universe/ 2>/dev/null || true

echo ""
echo "========================================"
echo "Fast training complete!"
echo "========================================"
echo ""
echo "To view:"
echo "  cd client && npm run dev"
echo "  Open http://localhost:3000"
echo ""
