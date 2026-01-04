# HELIOS Training Guide

Complete guide for training Gaussian Splatting representations on the HELIOS Neural Planetarium dataset.

## Quick Start

```bash
# 1. Prepare synthetic dataset (1000 stars for testing)
bash scripts/train_validation.sh

# 2. Validate results
bash scripts/validate_training.sh

# 3. Generate report and plots
python3 scripts/generate_training_report.py

# 4. Deploy to client
cp -r universe_train_test_trained client/public/universe
```

## Manual Training Commands

If you prefer manual control:

```bash
# Build dataset
cargo run --release -p universe-cli -- build \
    --output universe \
    --synthetic 1000

# Train all cells
cargo run --release -p universe-cli -- train-all \
    --input universe \
    --output universe_trained \
    --iterations 1000 \
    --learning-rate 0.001 \
    --views-per-iter 4

# Train single cell
cargo run --release -p universe-cli -- train-cell \
    --cell universe/cells/0_0_0.bin \
    --output trained_cell.bin
```

## Selective Training (Landmark-Focused) ⭐ Recommended for Production

For realistic datasets with 100k+ stars, training **all cells** is impractical (thousands of cells, 10-50 GPU-hours). Instead, use **landmark-focused training** to train only cells containing notable objects (galaxies, nebulae, star clusters, planets, spacecraft).

### Why Selective Training?

- **Efficiency**: Train 50-300 cells instead of 5000+ (100× time savings)
- **Quality where it matters**: Landmarks are where users actually visit
- **Practical demos**: 1-5 hours instead of 50+ hours

### Workflow

#### 1. Build Dataset with Landmarks
```bash
# Build universe with 100k stars
cargo run --release -p universe-cli -- build \
  --stars data/gaia_100k.csv \
  --limit 100000 \
  --max-mag 8.0 \
  --output universe

# Ensure universe/landmarks.json exists (see README "Realistic Sky Recipe")
# Built-in landmarks from client/src/landmarks.ts should be exported
```

#### 2. Train Only Landmark Cells
```bash
cargo run --release -p universe-cli -- train-landmarks \
  --input universe \
  --output universe_trained \
  --landmarks universe/landmarks.json \
  --neighbors 2 \
  --iterations 500 \
  --backend wgpu
```

**Parameters:**
- `--neighbors N`: Expand selection by N cells in each direction (l, theta, phi)
  - `N=0`: Train only cells containing landmarks (~50-100 cells)
  - `N=1`: Include immediate neighbors (~200-400 cells)
  - `N=2`: Include 2-shell neighborhood (~500-800 cells) **← Recommended**
  - `N=3`: Large neighborhood (~1000-1500 cells)

- `--backend`: Training backend
  - `wgpu`: GPU via Burn WGPU backend (default, works everywhere)
  - `torch-cuda`: PyTorch CUDA (faster if available, requires `--features torch`)
  - `torch-cpu`: PyTorch CPU (slower than wgpu)

#### 3. Compare Cell Counts
```bash
# Check what was selected
echo "Total cells: $(jq '.cells | length' universe/index.json)"
echo "Trained cells: $(jq '.cells | length' universe_trained/index.json)"

# Example output:
# Total cells: 5432
# Trained cells: 287 (5.3% of total)
```

#### 4. Use Trained Dataset
```bash
# Copy to client
cp -r universe_trained client/public/universe

# Or pack first for faster loading
cargo run --release -p universe-cli -- pack \
  --universe universe_trained \
  --output universe_trained/cells.pack.bin
```

### Expected Performance

| Dataset | Total Cells | Landmarks | Selected (N=2) | Training Time (wgpu) |
|---------|-------------|-----------|----------------|---------------------|
| 10k stars | ~500 | 100 | ~50-80 | 30 min - 1 hour |
| 100k stars | ~5000 | 100 | ~150-300 | 1-3 hours |
| 500k stars | ~15000 | 100 | ~200-400 | 2-5 hours |

### Cell Selection Algorithm

The `train-landmarks` command:
1. Loads landmarks.json (100+ objects)
2. For each landmark, computes containing CellId using HLG grid
3. Expands selection by N neighbors in all directions:
   - **l (radial)**: ±N shells
   - **theta (azimuth)**: ±N divisions (wraps around)
   - **phi (polar)**: ±N divisions (clamped to [0, n_phi-1])
4. Filters to cells that exist in the manifest
5. Trains only selected cells using standard training loop

### When to Use Full Training vs. Selective

**Use `train-all`:**
- Small datasets (< 100 cells)
- Testing/validation
- You have lots of GPU time

**Use `train-landmarks`:**
- Large datasets (100k+ stars → 1000+ cells)
- Production demos (user only visits landmarks)
- Limited training budget

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iterations` | 1000 | Gradient descent steps per cell |
| `--learning-rate` | 0.001 | Adam optimizer learning rate |
| `--views-per-iter` | 4 | Camera views per iteration |
| `--image-size` | 256×256 | Render resolution for training |
| `--lambda-dssim` | 0.2 | D-SSIM loss weight (0.0 = pure L1) |
| `--densify-interval` | 100 | Steps between densification (not yet implemented) |
| `--densify-until` | 800 | Stop densification after this iteration |
| `--prune-opacity-threshold` | 0.01 | Minimum opacity to keep splat (not yet implemented) |

### 3D Geometry Regularization (Prevents Z-Collapse)

These parameters prevent the "flat sheet" problem where splats collapse to 2D billboards during training:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lambda-isotropy` | 0.05 | Weight for isotropy loss (encourages spherical splats) |
| `--lambda-collapse` | 0.1 | Weight for collapse prevention loss |
| `--min-scale-ratio` | 0.1 | Minimum ratio of smallest/largest scale axis |

**Why these matter:**
- For astronomical point sources (stars), splats should be spherical
- Without regularization, the 2D image loss allows Z-axis to collapse
- `lambda-isotropy` penalizes variance between scale axes (encourages equal scales)
- `lambda-collapse` penalizes any axis shrinking below 10% of the largest axis

**Tuning tips:**
- Increase `lambda-isotropy` (0.1-0.2) if splats appear flat/elliptical
- Increase `lambda-collapse` (0.2-0.5) if Z-collapse still occurs
- Decrease `min-scale-ratio` (0.05) for extended objects like galaxies

## Expected Results

### Small Test (1k stars, ~10 cells)
- **Training time**: ~1 hour on CPU
- **Loss reduction**: 60-90% (typical)
- **Output size**: ~5-20 MB compressed
- **Splat count**: Similar to input (densification not implemented)

### Medium Scale (10k stars, ~50 cells)
- **Training time**: ~5-10 hours
- **Loss reduction**: 60-90%
- **Output size**: ~50-100 MB compressed

### Full Scale (100k stars, ~100 cells)
- **Training time**: ~10-20 hours
- **Loss reduction**: 60-90%
- **Output size**: ~50-200 MB compressed

## Understanding the Training Process

### Loss Function

The training uses a combined L1 + D-SSIM loss:

```
loss = (1 - λ) × L1(rendered, gt) + λ × D-SSIM(rendered, gt)
     = 0.8 × L1 + 0.2 × D-SSIM
```

- **L1 loss**: Pixel-level accuracy
- **D-SSIM**: Perceptual similarity (structural information)
- **λ = 0.2**: Default weight (adjustable via `--lambda-dssim`)

### Training Loop (per cell)

For each cell:
1. Extract initial Gaussian splat parameters from dataset
2. Initialize learnable parameters (positions, scales, rotations, colors, opacities)
3. For each iteration:
   - Sample random camera views around cell
   - Render using differentiable rasterizer
   - Compare to ground truth (simple splatting)
   - Compute gradients via backpropagation
   - Update parameters with Adam optimizer
4. Export trained splats to output dataset

### Ground Truth Renderer

The training compares against a simple CPU-based ground truth renderer that:
- Projects Gaussians to screen space
- Rasterizes each Gaussian as a 2D ellipse
- Alpha-blends in depth order

This provides a reference target for the differentiable rasterizer to match.

## Validation

### Check Training Success

```bash
bash scripts/validate_training.sh
```

Expected output:
```
Cell count:
  Original: 10
  Trained:  10
  ✅ Cell count matches

Training metrics:
  Initial loss: 0.234567
  Final loss:   0.045678
  Loss reduction: 80.5%

Dataset sizes:
  Original: 12M
  Trained:  15M
```

### Generate Convergence Plot

```bash
python3 scripts/generate_training_report.py
```

This creates:
- `training_convergence.png` - Loss vs iteration plot
- `client/public/papers/data/training_metrics.json` - Metrics for website
- `client/public/papers/figures/convergence.png` - Plot for website

## Current Limitations

### Not Yet Implemented

1. **Adaptive Density Control**:
   - Densification (splitting/cloning high-gradient splats)
   - Pruning (removing low-opacity or small splats)
   - See TODO at `crates/universe-train/src/trainer.rs:179-180`

2. **GPU Acceleration**:
   - Training uses CPU backend (NdArray)
   - WGPU backend available but not critical for current scale
   - Sufficient for per-cell training (50-500 splats)

3. **Parallelization**:
   - Cells trained sequentially
   - Easy to parallelize with rayon if needed

### Working As Designed

- **Fixed Gaussian count**: Number of splats stays constant (no densification/pruning)
- **CPU-only training**: Fast enough for validation, GPU not required
- **Synthetic data**: Using generated stars for testing before real catalogs

## Troubleshooting

### Training Slow or Out of Memory

**Reduce image resolution**:
```bash
cargo run --release -p universe-cli -- train-all \
    --image-size 128
```

**Reduce views per iteration**:
```bash
cargo run --release -p universe-cli -- train-all \
    --views-per-iter 2
```

**Use release build** (10-100× faster):
```bash
cargo run --release -p universe-cli -- ...  # Always use --release!
```

### Loss Not Decreasing

**Try different learning rates**:
```bash
# Faster convergence (may overshoot)
--learning-rate 0.01

# Slower but more stable
--learning-rate 0.0001
```

**Check ground truth**:
- Verify ground truth renderer produces non-uniform images
- Check `RUST_LOG=debug` for diagnostic info

**Verify gradient flow**:
```bash
cargo test -p universe-train test_gradient_flow
```

### Gradient Errors

If you see errors like "unsqueezed rank must be greater than input rank":
- This was fixed in `model.rs` (removed incorrect `unsqueeze_dim` call)
- Make sure you have the latest code

### Dataset Issues

**No cells generated**:
- Check that input data has valid positions
- Verify HLG grid is not too coarse/fine
- Look at `universe_train_test/index.json` for cell list

**Cells too large**:
- Reduce grid level or increase angular sectors
- See `crates/universe-core/src/grid.rs` for HLG config

## Performance Tuning

### Training Speed

| Configuration | Speed (1000 iters) | Quality |
|---------------|-------------------|---------|
| 256×256, 4 views | ~60 min | Best |
| 128×128, 4 views | ~15 min | Good |
| 256×256, 2 views | ~30 min | Good |
| 128×128, 2 views | ~8 min | Acceptable |

### Memory Usage

Approximate memory per cell training:
- 256×256, 1000 splats: ~256 MB
- 128×128, 1000 splats: ~64 MB
- Scales roughly as: `H × W × N × 4 bytes`

## Data Sources

### Synthetic Stars (Testing)

```bash
cargo run --release -p universe-cli -- build --synthetic 1000
```

Generates random stars with:
- Uniform distribution in HLG grid
- Random colors
- Fixed sizes and opacities
- Useful for pipeline validation

### Real Data (Future)

Planned support for:
- **MPCORB**: Asteroid orbits (~1.3M objects)
- **Gaia DR3**: Star catalog (~1.8B stars)
- **Custom CSV**: Import your own data

## Next Steps

### After Successful Validation

1. **Scale up**: Train on larger datasets (10k, 100k stars)
2. **Deploy**: Copy trained dataset to `client/public/universe`
3. **Visualize**: Open browser client to see results
4. **Iterate**: Adjust parameters and retrain

### Future Improvements

1. **Implement densification/pruning** (lines 179-180 in trainer.rs)
2. **Add Gaia star ingestion pipeline**
3. **Optimize with WGPU backend** (if needed)
4. **Parallelize cell training** (rayon)
5. **Add temporal dynamics** (orbit propagation)

## Additional Resources

- **Differentiable Rasterizer**: `crates/universe-train/src/rasterizer.rs`
- **Training Loop**: `crates/universe-train/src/trainer.rs`
- **Ground Truth**: `crates/universe-train/src/ground_truth.rs`
- **Tests**: `cargo test -p universe-train`

## Support

For issues or questions:
- Check test results: `cargo test -p universe-train`
- Enable debug logging: `RUST_LOG=debug`
- Review training log: `less training.log`
