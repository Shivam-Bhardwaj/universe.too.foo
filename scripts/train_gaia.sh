#!/bin/bash
# Full Gaia training pipeline
# Usage: ./scripts/train_gaia.sh [STAR_COUNT] [ITERATIONS]
#
# Examples:
#   ./scripts/train_gaia.sh 50000 500      # 50k stars, 500 iterations (quick test)
#   ./scripts/train_gaia.sh 100000 1000    # 100k stars, 1000 iterations (production)
#   ./scripts/train_gaia.sh 200000 1000    # 200k stars, 1000 iterations (high quality)

set -e

STAR_COUNT=${1:-100000}
ITERATIONS=${2:-500}
BACKEND=${3:-wgpu}

DATA_DIR="data"
UNIVERSE_NAME="universe_gaia_${STAR_COUNT}"
TRAINED_NAME="${UNIVERSE_NAME}_trained"
CSV_FILE="${DATA_DIR}/gaia_${STAR_COUNT}.csv"

echo "========================================"
echo "HELIOS Gaia Training Pipeline"
echo "========================================"
echo "Stars:      $STAR_COUNT"
echo "Iterations: $ITERATIONS"
echo "Backend:    $BACKEND"
echo "Output:     $TRAINED_NAME"
echo "========================================"

# Step 1: Download Gaia data if not exists
if [ ! -f "$CSV_FILE" ]; then
    echo ""
    echo "[1/5] Downloading Gaia DR3 data ($STAR_COUNT stars)..."
    mkdir -p "$DATA_DIR"

    # Query with quality filters:
    # - parallax > 0.5 mas (distance < 2000 pc for reasonable 3D)
    # - parallax_over_error > 10 (good distance precision)
    # - Random sampling for sky coverage
    curl -G "https://gea.esac.esa.int/tap-server/tap/sync" \
      --data-urlencode "REQUEST=doQuery" \
      --data-urlencode "LANG=ADQL" \
      --data-urlencode "FORMAT=csv" \
      --data-urlencode "QUERY=SELECT TOP $STAR_COUNT source_id,ra,dec,parallax,phot_g_mean_mag,bp_rp FROM gaiadr3.gaia_source WHERE phot_g_mean_mag < 12 AND parallax > 0.5 AND parallax_over_error > 10 ORDER BY RANDOM_INDEX" \
      -o "${CSV_FILE}.tmp"

    # Verify download
    LINE_COUNT=$(wc -l < "${CSV_FILE}.tmp")
    if [ "$LINE_COUNT" -lt 100 ]; then
        echo "ERROR: Download failed or returned too few stars"
        cat "${CSV_FILE}.tmp"
        rm "${CSV_FILE}.tmp"
        exit 1
    fi

    mv "${CSV_FILE}.tmp" "$CSV_FILE"
    echo "Downloaded $((LINE_COUNT - 1)) stars to $CSV_FILE"
else
    echo ""
    echo "[1/5] Using existing Gaia data: $CSV_FILE"
    LINE_COUNT=$(wc -l < "$CSV_FILE")
    echo "Found $((LINE_COUNT - 1)) stars"
fi

# Step 2: Build universe
echo ""
echo "[2/5] Building universe dataset..."
cargo run --release -p universe-cli -- build \
  --stars "$CSV_FILE" \
  --limit "$STAR_COUNT" \
  --max-mag 12.0 \
  --output "$UNIVERSE_NAME"

# Step 3: Generate landmarks from brightest stars
echo ""
echo "[3/5] Generating landmarks..."
cargo run --release -p universe-cli -- generate-landmarks \
  --universe "$UNIVERSE_NAME" \
  --stars "$CSV_FILE" \
  --count 100 \
  --output "$UNIVERSE_NAME/landmarks.json"

# Step 4: Train all cells
echo ""
echo "[4/5] Training all cells (this may take a while)..."
echo "      Estimated time: ~$(echo "$STAR_COUNT / 1000 * $ITERATIONS / 500" | bc) minutes"
cargo run --release -p universe-cli -- train-all \
  --input "$UNIVERSE_NAME" \
  --output "$TRAINED_NAME" \
  --iterations "$ITERATIONS" \
  --backend "$BACKEND"

# Step 5: Copy to client
echo ""
echo "[5/5] Copying to client..."
rm -rf client/public/universe
cp -r "$TRAINED_NAME" client/public/universe

# Also copy landmarks
if [ -f "$UNIVERSE_NAME/landmarks.json" ]; then
    cp "$UNIVERSE_NAME/landmarks.json" client/public/universe/
fi

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"
echo ""
echo "Results:"
CELL_COUNT=$(jq '.cells | length' "$TRAINED_NAME/index.json")
SPLAT_COUNT=$(jq '[.cells[].splat_count] | add' "$TRAINED_NAME/index.json")
echo "  Cells:  $CELL_COUNT"
echo "  Splats: $SPLAT_COUNT"
echo ""
echo "To view:"
echo "  cd client && npm run dev"
echo "  Open http://localhost:3000"
echo ""
