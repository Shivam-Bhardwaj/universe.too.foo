# Testing Guide - Landmarks + Navigation Overhaul

This guide walks you through testing all the new features: anchor navigation, landmark ingestion, and landmark-focused training.

## Quick Test (5-10 minutes) - Synthetic Dataset

### Step 1: Build a Small Test Dataset

```bash
cd /home/curious/HELIOS

# Build with synthetic stars (fast, no external data needed)
cargo run --release -p universe-cli -- build \
  --synthetic 5000 \
  --output universe_test
```

**Expected output:**
```
Universe built: X cells, Y splats, Z MB
```

### Step 2: Export Landmarks (One-Time Setup)

Create `universe_test/landmarks.json` with a few test landmarks. You can use this minimal example:

```bash
cat > universe_test/landmarks.json << 'EOF'
[
  {
    "id": "sun",
    "name": "Sun",
    "kind": "star",
    "pos_meters": {"x": 0, "y": 0, "z": 0},
    "radius_hint": 6.96e8,
    "description": "Our star"
  },
  {
    "id": "earth",
    "name": "Earth",
    "kind": "planet",
    "pos_meters": {"x": 1.496e11, "y": 0, "z": 0},
    "radius_hint": 6.371e6,
    "description": "Home"
  },
  {
    "id": "andromeda",
    "name": "Andromeda Galaxy (M31)",
    "kind": "galaxy",
    "pos_meters": {"x": 2.4e22, "y": 0, "z": 0},
    "radius_hint": 1.1e21,
    "description": "Nearest large galaxy"
  }
]
EOF
```

### Step 3: Rebuild with Landmarks

```bash
# Rebuild to ingest landmarks
cargo run --release -p universe-cli -- build \
  --synthetic 5000 \
  --output universe_test
```

**Expected output:**
```
Loading landmarks from universe_test/landmarks.json
Ingesting 3 landmarks
Universe built: X cells, Y splats, Z MB
```

### Step 4: Test Landmark-Focused Training

```bash
# Train only cells containing landmarks (+ 1 neighbor shell)
cargo run --release -p universe-cli -- train-landmarks \
  --input universe_test \
  --output universe_test_trained \
  --landmarks universe_test/landmarks.json \
  --neighbors 1 \
  --iterations 100 \
  --backend wgpu
```

**Expected output:**
```
Loading landmarks from universe_test/landmarks.json
Loaded 3 landmarks
Universe has X cells total
Found Y cells containing landmarks
Expanded to Z cells (neighbors=1)
Training Z cells (out of X total)
Training selected cells...
Selective training complete. Output: universe_test_trained
```

**Verify cell counts:**
```bash
echo "Original cells: $(jq '.cells | length' universe_test/index.json)"
echo "Trained cells: $(jq '.cells | length' universe_test_trained/index.json)"
```

You should see trained cells << original cells (e.g., 5-20 trained vs 50-200 original).

### Step 5: Pack for Fast Loading

```bash
cargo run --release -p universe-cli -- pack \
  --universe universe_test_trained
```

### Step 6: Test Client Navigation

```bash
# Copy trained dataset to client public directory
cp -r universe_test_trained client/public/universe

# Start client dev server
cd client
npm install  # if not already done
npm run dev
```

**Open browser:** `http://localhost:3000`

**Test checklist:**
- [ ] Search for "Earth" → click → verify warp animation
- [ ] Verify breadcrumb shows "Earth • X AU • Y AU from Sun"
- [ ] Try zooming out (scroll/pinch) → should hit max distance limit
- [ ] Search for "Andromeda" → warp → verify anchor switches to Andromeda
- [ ] Verify you see a visible galaxy splat (pale purple, transparent)
- [ ] Check console for `[PREFETCH]` logs during jumps

---

## Full Test (30-60 minutes) - Real Gaia Data

### Step 1: Download Gaia Stars

```bash
# Download top 10k brightest stars (smaller than 100k for faster testing)
curl -G "https://gea.esac.esa.int/tap-server/tap/sync" \
  --data-urlencode "REQUEST=doQuery" \
  --data-urlencode "LANG=ADQL" \
  --data-urlencode "FORMAT=csv" \
  --data-urlencode "QUERY=SELECT TOP 10000 source_id,ra,dec,parallax,phot_g_mean_mag,bp_rp FROM gaiadr3.gaia_source WHERE phot_g_mean_mag < 8 AND parallax > 0 ORDER BY phot_g_mean_mag" \
  -o data/gaia_10k.csv
```

### Step 2: Export Full Landmark Catalog

Create a script to export landmarks from TypeScript:

```bash
cat > scripts/export-landmarks.js << 'EOF'
// Quick script to export landmarks (requires Node.js)
const fs = require('fs');
const path = require('path');

// Read TypeScript landmarks file
const landmarksTs = fs.readFileSync('client/src/landmarks.ts', 'utf8');

// Extract BUILTIN_LANDMARKS array (simplified - you may need to adjust)
// For now, manually copy the JSON structure from client/src/landmarks.ts
// Or use a proper TypeScript parser

console.log('Manual export required:');
console.log('1. Open client/src/landmarks.ts');
console.log('2. Copy BUILTIN_LANDMARKS array');
console.log('3. Convert to JSON format');
console.log('4. Save as universe/landmarks.json');
EOF

node scripts/export-landmarks.js
```

**Or manually:** Copy the `BUILTIN_LANDMARKS` array from `client/src/landmarks.ts` and convert to JSON format.

### Step 3: Build with Real Stars + Landmarks

```bash
cargo run --release -p universe-cli -- build \
  --stars data/gaia_10k.csv \
  --limit 10000 \
  --max-mag 8.0 \
  --output universe_real
```

**Verify landmarks were ingested:**
```bash
# Check if landmark cells exist
ls universe_real/cells/ | wc -l
jq '.cells | length' universe_real/index.json
```

### Step 4: Train Landmark Neighborhoods

```bash
cargo run --release -p universe-cli -- train-landmarks \
  --input universe_real \
  --output universe_real_trained \
  --landmarks universe_real/landmarks.json \
  --neighbors 2 \
  --iterations 500 \
  --backend wgpu
```

**Expected:** Training 100-300 cells (out of 500-2000 total), taking 30-90 minutes.

### Step 5: Test in Client

```bash
# Pack trained dataset
cargo run --release -p universe-cli -- pack --universe universe_real_trained

# Copy to client
cp -r universe_real_trained client/public/universe

# Start client
cd client && npm run dev
```

**Test deep-sky objects:**
- [ ] Search "Andromeda" → warp → see galaxy splat
- [ ] Search "Orion Nebula" → warp → see pink nebula splat
- [ ] Search "M13" → warp → see cluster splat
- [ ] Search "Vega" → warp → see star splat

---

## Unit Tests (Rust)

### Test Landmark Loading

```bash
cargo test -p universe-data --lib landmarks
```

### Test Training Functions

```bash
cargo test -p universe-train
```

### Test CLI Commands

```bash
# Test help output
cargo run --release -p universe-cli -- train-landmarks --help
```

---

## Integration Test Script

Create a complete end-to-end test:

```bash
cat > scripts/test_landmarks.sh << 'EOF'
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
EOF

chmod +x scripts/test_landmarks.sh
./scripts/test_landmarks.sh
```

---

## Troubleshooting

### "No landmarks.json found"
- Ensure `landmarks.json` exists in the output directory before running `build`
- Check file path: `universe_test/landmarks.json` (not `universe_test/cells/landmarks.json`)

### "No cells containing landmarks"
- Landmarks might be inside `R_MIN` (Mercury perihelion) - these are skipped
- Check landmark positions are > 4.6e10 meters from origin
- Verify HLG grid can map the positions (check logs)

### Training fails with "cell not found"
- Ensure input directory has `index.json` and `cells/*.bin`
- Check that selected cells exist in the manifest

### Client shows no landmarks
- Verify `universe/landmarks.json` exists in `client/public/universe/`
- Check browser console for loading errors
- Ensure dataset was built with landmarks ingested

### Anchor doesn't switch on warp
- Check browser console for errors
- Verify `flightControls.startJump()` is called with `targetId` parameter
- Check that jump completion callback is set up in `main.ts`

---

## Performance Benchmarks

### Expected Training Times (wgpu backend, 500 iterations)

| Dataset Size | Total Cells | Selected (N=2) | Training Time |
|-------------|-------------|----------------|---------------|
| 1k stars | ~50 | ~5-10 | 5-10 min |
| 10k stars | ~500 | ~50-100 | 30-60 min |
| 100k stars | ~5000 | ~150-300 | 2-5 hours |

### Expected Cell Counts

| Neighbors | Typical Selection | Use Case |
|-----------|-------------------|----------|
| N=0 | ~50-100 cells | Minimal (landmarks only) |
| N=1 | ~150-300 cells | Recommended for demos |
| N=2 | ~300-600 cells | Production quality |
| N=3 | ~600-1200 cells | Maximum coverage |

---

## Next Steps After Testing

1. **Scale up**: Use 100k stars + full landmark catalog
2. **Deploy**: Copy trained dataset to production server
3. **Iterate**: Adjust training parameters based on visual quality
4. **Document**: Add any workflow improvements to README


