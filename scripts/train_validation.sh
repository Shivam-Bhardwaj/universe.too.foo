#!/bin/bash
set -e

echo "=== HELIOS Training Validation Pipeline ==="
echo ""

# Configuration
DATASET_DIR="universe_train_test"
NUM_STARS=1000

# Clean previous runs
echo "Cleaning previous test data..."
rm -rf "$DATASET_DIR"

# Build synthetic dataset
echo ""
echo "Building synthetic dataset ($NUM_STARS stars)..."
cargo run --release -p universe-cli -- build \
    --output "$DATASET_DIR" \
    --synthetic "$NUM_STARS"

# Verify dataset
echo ""
echo "Dataset created:"
find "$DATASET_DIR" -type f -name "*.bin" 2>/dev/null | wc -l | xargs echo "  Cells:"
du -sh "$DATASET_DIR" 2>/dev/null | awk '{print "  Size: " $1}'

# Run training
echo ""
echo "Starting training..."
echo "  (This will take approximately 1 hour for 1000 stars)"
echo ""

RUST_LOG=info cargo run --release -p universe-cli -- train-all \
    --input "$DATASET_DIR" \
    --output "${DATASET_DIR}_trained" \
    --iterations 1000 \
    --learning-rate 0.001 \
    --views-per-iter 4 \
    2>&1 | tee training.log

echo ""
echo "===================================================="
echo "Training complete! Output: ${DATASET_DIR}_trained"
echo "===================================================="
echo ""
echo "Next steps:"
echo "  1. Run validation: bash scripts/validate_training.sh"
echo "  2. Generate report: python3 scripts/generate_training_report.py"
