#!/bin/bash
# Download Gaia DR3 star data from ESA TAP service
#
# Usage: ./scripts/download_gaia.sh [STAR_COUNT]
#
# Examples:
#   ./scripts/download_gaia.sh 50000    # 50k stars (quick test)
#   ./scripts/download_gaia.sh 100000   # 100k stars
#   ./scripts/download_gaia.sh 500000   # 500k stars (large, slower)

set -e

STAR_COUNT=${1:-100000}
DATA_DIR="data"
CSV_FILE="${DATA_DIR}/gaia_${STAR_COUNT}.csv"

echo "========================================"
echo "HELIOS Gaia Data Downloader"
echo "========================================"
echo "Requesting: $STAR_COUNT stars"
echo "Output: $CSV_FILE"
echo "========================================"

mkdir -p "$DATA_DIR"

if [ -f "$CSV_FILE" ]; then
    LINE_COUNT=$(wc -l < "$CSV_FILE")
    echo "File already exists with $((LINE_COUNT - 1)) stars"
    echo "Delete it to re-download: rm $CSV_FILE"
    exit 0
fi

echo ""
echo "Downloading from ESA Gaia Archive..."
echo "(This may take a few minutes for large counts)"

# Query for high-quality stars:
# - parallax > 0.5 mas (within ~2000 pc)
# - parallax_over_error > 10 (10% precision or better)
# - phot_g_mean_mag < 15 (reasonably bright)
# - Random sampling for uniform distribution
curl -G "https://gea.esac.esa.int/tap-server/tap/sync" \
  --data-urlencode "REQUEST=doQuery" \
  --data-urlencode "LANG=ADQL" \
  --data-urlencode "FORMAT=csv" \
  --data-urlencode "QUERY=SELECT TOP $STAR_COUNT source_id,ra,dec,parallax,phot_g_mean_mag,bp_rp FROM gaiadr3.gaia_source WHERE phot_g_mean_mag < 15 AND parallax > 0.5 AND parallax_over_error > 10 ORDER BY RANDOM_INDEX" \
  -o "$CSV_FILE"

if [ -f "$CSV_FILE" ]; then
    LINE_COUNT=$(wc -l < "$CSV_FILE")
    echo ""
    echo "Downloaded $((LINE_COUNT - 1)) stars to $CSV_FILE"
else
    echo "Error: Download failed"
    exit 1
fi
