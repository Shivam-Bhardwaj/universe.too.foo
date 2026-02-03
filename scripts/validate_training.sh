#!/bin/bash
set -e

TRAINED_DIR="universe_train_test_trained"

echo "=== Training Validation ==="
echo ""

# Check output exists
if [ ! -d "$TRAINED_DIR" ]; then
    echo "ERROR: Trained dataset not found at $TRAINED_DIR"
    echo "Please run 'bash scripts/train_validation.sh' first."
    exit 1
fi

# Compare cell counts
ORIGINAL_CELLS=$(find universe_train_test/cells -name "*.bin" 2>/dev/null | wc -l)
TRAINED_CELLS=$(find "$TRAINED_DIR/cells" -name "*.bin" 2>/dev/null | wc -l)

echo "Cell count:"
echo "  Original: $ORIGINAL_CELLS"
echo "  Trained:  $TRAINED_CELLS"

if [ "$ORIGINAL_CELLS" -ne "$TRAINED_CELLS" ]; then
    echo "  ⚠️  WARNING: Cell count mismatch!"
else
    echo "  ✅ Cell count matches"
fi

# Extract training metrics from logs (if available)
echo ""
echo "Training metrics:"
if [ -f "training.log" ]; then
    # Extract first and last loss values
    FIRST_LOSS=$(grep -m1 'loss=' training.log | sed 's/.*loss=\([0-9.]*\).*/\1/' | head -1)
    LAST_LOSS=$(grep 'final=' training.log | tail -1 | sed 's/.*final=\([0-9.]*\).*/\1/')

    if [ -n "$FIRST_LOSS" ] && [ -n "$LAST_LOSS" ]; then
        echo "  Initial loss: $FIRST_LOSS"
        echo "  Final loss:   $LAST_LOSS"

        # Calculate reduction percentage using bc if available
        if command -v bc &> /dev/null; then
            REDUCTION=$(echo "scale=1; (1 - $LAST_LOSS / $FIRST_LOSS) * 100" | bc)
            echo "  Loss reduction: ${REDUCTION}%"
        fi
    else
        echo "  (Could not parse loss values from training.log)"
    fi
else
    echo "  (No training.log found - check console output)"
fi

# Size comparison
echo ""
echo "Dataset sizes:"
du -sh universe_train_test 2>/dev/null | awk '{print "  Original: " $1}'
du -sh "$TRAINED_DIR" 2>/dev/null | awk '{print "  Trained:  " $1}'

# Count total splats
if command -v jq &> /dev/null; then
    echo ""
    echo "Total splats:"
    ORIG_SPLATS=$(jq '.cells | map(.splat_count) | add' universe_train_test/index.json 2>/dev/null)
    TRAINED_SPLATS=$(jq '.cells | map(.splat_count) | add' "$TRAINED_DIR/index.json" 2>/dev/null)

    if [ -n "$ORIG_SPLATS" ] && [ -n "$TRAINED_SPLATS" ]; then
        echo "  Original: $ORIG_SPLATS"
        echo "  Trained:  $TRAINED_SPLATS"
    fi
fi

echo ""
echo "===================================================="
echo "Validation complete!"
echo "===================================================="
echo ""
echo "Trained dataset location: $TRAINED_DIR"
echo ""
echo "Next steps:"
echo "  1. Generate detailed report: python3 scripts/generate_training_report.py"
echo "  2. Deploy to client: cp -r $TRAINED_DIR client/public/universe"
