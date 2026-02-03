#!/bin/bash
set -e

echo "=== Testing Landmarks + Navigation Overhaul ==="

# Clean previous test
rm -rf universe_test* 2>/dev/null || true

# Step 1: Build synthetic dataset
echo "Step 1: Building synthetic dataset..."
cargo run --release -p universe-cli -- build \
  --synthetic 1000 \
  --output universe_test

# Step 2: Create minimal landmarks
echo "Step 2: Creating test landmarks..."
mkdir -p universe_test
cat > universe_test/landmarks.json << 'LANDMARKS'
[
  {"id": "earth", "name": "Earth", "kind": "planet", "pos_meters": {"x": 1.496e11, "y": 0, "z": 0}, "radius_hint": 6.371e6}
]
LANDMARKS

# Step 3: Rebuild with landmarks
echo "Step 3: Rebuilding with landmarks..."
cargo run --release -p universe-cli -- build \
  --synthetic 1000 \
  --output universe_test

# Step 4: Test training
echo "Step 4: Testing landmark-focused training..."
cargo run --release -p universe-cli -- train-landmarks \
  --input universe_test \
  --output universe_test_trained \
  --landmarks universe_test/landmarks.json \
  --neighbors 1 \
  --iterations 50 \
  --backend wgpu

# Step 5: Verify
echo "Step 5: Verifying results..."
ORIGINAL=$(jq '.cells | length' universe_test/index.json)
TRAINED=$(jq '.cells | length' universe_test_trained/index.json)

echo "Original cells: $ORIGINAL"
echo "Trained cells: $TRAINED"

if [ "$TRAINED" -lt "$ORIGINAL" ]; then
  echo "✅ SUCCESS: Selective training worked ($TRAINED < $ORIGINAL)"
else
  echo "❌ FAILED: Expected trained < original"
  exit 1
fi

echo "=== All tests passed! ==="


