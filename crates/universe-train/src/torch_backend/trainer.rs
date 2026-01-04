//! TorchTrainer - CUDA-accelerated training for Gaussian splatting
//!
//! Implements 3D Gaussian Splatting training with:
//! - CUDA acceleration via tch-rs (LibTorch bindings)
//! - Adaptive density control: densification and pruning
//! - Gradient accumulation for density decisions

use anyhow::{bail, Result};
use indicatif::{ProgressBar, ProgressStyle};
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::camera::{generate_training_cameras, Camera};
use crate::ground_truth::GroundTruthRenderer;
use crate::TrainConfig;

use super::loss::{combined_loss, full_loss_with_regularization, LossComponents};
use super::rasterizer::render_gaussians;

use universe_core::grid::CellBounds;
use universe_data::{CellData, CellEntry, CellManifest, GaussianSplat};

/// Accumulates position gradients for densification decisions
struct GradientAccumulator {
    /// Sum of gradient norms per splat [N]
    position_grad_sum: Tensor,
    /// Number of accumulated samples
    count: i64,
    device: Device,
}

impl GradientAccumulator {
    fn new(num_splats: i64, device: Device) -> Self {
        Self {
            position_grad_sum: Tensor::zeros([num_splats], (Kind::Float, device)),
            count: 0,
            device,
        }
    }

    /// Accumulate gradient norms from positions gradient
    fn accumulate(&mut self, positions_grad: &Tensor) {
        // positions_grad is [N, 3], compute per-splat L2 norm
        let grad_norm = positions_grad
            .pow_tensor_scalar(2)
            .sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
            .sqrt();
        self.position_grad_sum = &self.position_grad_sum + grad_norm;
        self.count += 1;
    }

    /// Get average gradient norms
    fn average(&self) -> Tensor {
        if self.count == 0 {
            return self.position_grad_sum.shallow_clone();
        }
        &self.position_grad_sum / (self.count as f64)
    }

    /// Reset accumulator (after densification)
    fn reset(&mut self, new_num_splats: i64) {
        self.position_grad_sum = Tensor::zeros([new_num_splats], (Kind::Float, self.device));
        self.count = 0;
    }
}

/// Device selection for torch backend
#[derive(Clone, Copy, Debug)]
pub enum TorchDevice {
    Cpu,
    Cuda(usize),
}

impl TorchDevice {
    pub fn to_tch_device(self) -> Device {
        match self {
            TorchDevice::Cpu => Device::Cpu,
            TorchDevice::Cuda(idx) => Device::Cuda(idx),
        }
    }

    /// Check if CUDA is available and return appropriate device
    pub fn cuda_if_available() -> Self {
        if tch::Cuda::is_available() {
            TorchDevice::Cuda(0)
        } else {
            tracing::warn!("CUDA not available, falling back to CPU");
            TorchDevice::Cpu
        }
    }
}

/// Torch-based trainer for Gaussian splatting
pub struct TorchTrainer {
    config: TrainConfig,
    device: Device,
}

impl TorchTrainer {
    pub fn new(config: TrainConfig, device: TorchDevice) -> Result<Self> {
        let tch_device = device.to_tch_device();

        // Validate CUDA availability if requested
        if let TorchDevice::Cuda(_) = device {
            if !tch::Cuda::is_available() {
                bail!(
                    "CUDA requested but not available. \
                     Ensure libtorch is built with CUDA support."
                );
            }
            let cuda_count = tch::Cuda::device_count();
            tracing::info!("CUDA available with {} device(s)", cuda_count);
        }

        Ok(Self {
            config,
            device: tch_device,
        })
    }

    /// Train a single cell
    pub fn train_cell(&self, cell: &CellData) -> Result<Vec<GaussianSplat>> {
        tracing::info!(
            "TorchTrainer: Training cell ({},{},{}) with {} splats on {:?}",
            cell.metadata.id.l,
            cell.metadata.id.theta,
            cell.metadata.id.phi,
            cell.splats.len(),
            self.device
        );

        if cell.splats.is_empty() {
            return Ok(vec![]);
        }

        // Create model on device with VarStore for proper optimizer integration
        let vs = nn::VarStore::new(self.device);
        let root = vs.root();

        // Initialize model parameters in VarStore
        let n = cell.splats.len() as i64;
        let pos_data: Vec<f32> = cell.splats.iter().flat_map(|s| s.pos.iter().copied()).collect();
        let scale_data: Vec<f32> = cell.splats.iter().flat_map(|s| s.scale.iter().map(|x| x.max(1e-6).ln())).collect();
        let rot_data: Vec<f32> = cell.splats.iter().flat_map(|s| s.rotation.iter().copied()).collect();
        let color_data: Vec<f32> = cell.splats.iter().flat_map(|s| s.color.iter().copied()).collect();
        let opacity_data: Vec<f32> = cell.splats.iter().map(|s| inverse_sigmoid(s.opacity.clamp(0.01, 0.99))).collect();

        let positions = root.var_copy("positions", &Tensor::from_slice(&pos_data).view([n, 3]));
        let log_scales = root.var_copy("log_scales", &Tensor::from_slice(&scale_data).view([n, 3]));
        let rotations = root.var_copy("rotations", &Tensor::from_slice(&rot_data).view([n, 4]));
        let colors = root.var_copy("colors", &Tensor::from_slice(&color_data).view([n, 3]));
        let logit_opacities = root.var_copy("logit_opacities", &Tensor::from_slice(&opacity_data));

        // Create Adam optimizer
        let mut optimizer = nn::Adam::default().build(&vs, self.config.learning_rate)?;

        // Generate camera views - use actual splat positions, not cell bounds
        let avg_pos: glam::Vec3 = cell.splats.iter()
            .map(|s| glam::Vec3::new(s.pos[0], s.pos[1], s.pos[2]))
            .fold(glam::Vec3::ZERO, |acc, p| acc + p) / cell.splats.len() as f32;

        // Compute radius from splat positions
        let max_dist = cell.splats.iter()
            .map(|s| {
                let p = glam::Vec3::new(s.pos[0], s.pos[1], s.pos[2]);
                (p - avg_pos).length()
            })
            .fold(0.0f32, |a, b| a.max(b));
        let radius = max_dist.max(1e15); // Minimum radius for single splats

        let cameras = generate_training_cameras(
            avg_pos,
            radius,
            self.config.views_per_iter * 10,
            self.config.image_size,
        );

        // Ground truth renderer (CPU-based)
        let gt_renderer =
            GroundTruthRenderer::new(self.config.image_size.0, self.config.image_size.1);

        // Progress bar
        let pb = ProgressBar::new(self.config.iterations as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed}] [{bar:30}] {pos}/{len} loss={msg}")
                .expect("template"),
        );

        let mut best_loss = f32::INFINITY;
        let mut first_loss = 0.0f32;

        // Initialize gradient accumulator for densification
        let mut grad_accum = GradientAccumulator::new(n, self.device);

        // Track current number of splats (changes with densify/prune)
        let mut current_n = n;

        // Training loop
        for iter in 0..self.config.iterations {
            let mut total_loss = 0.0;

            for view_idx in 0..self.config.views_per_iter {
                let cam_idx = (iter * self.config.views_per_iter + view_idx) % cameras.len();
                let camera = &cameras[cam_idx];

                // Render ground truth and convert to tensor
                let gt_image = gt_renderer.render(&cell.splats, camera);
                let gt_tensor = image_to_torch_tensor(&gt_image, self.device);

                // Compute derived quantities
                let scales = log_scales.exp();
                let opacities = logit_opacities.sigmoid();
                let rot_norm = rotations.pow_tensor_scalar(2).sum_dim_intlist([1i64].as_slice(), true, Kind::Float).sqrt();
                let normalized_rotations = &rotations / &rot_norm;

                // Forward pass (differentiable render)
                let rendered = render_gaussians(
                    &positions,
                    &scales,
                    &normalized_rotations,
                    &colors.clamp(0.0, 1.0),
                    &opacities,
                    camera,
                    self.config.image_size,
                );

                // Compute loss with 3D geometry regularization to prevent Z-collapse
                let (loss, loss_components) = full_loss_with_regularization(
                    &rendered,
                    &gt_tensor,
                    &scales,
                    &log_scales,
                    self.config.lambda_dssim,
                    self.config.lambda_isotropy,
                    self.config.lambda_collapse,
                );

                let loss_val: f32 = loss_components.total;
                total_loss += loss_val;

                // Backward pass
                optimizer.zero_grad();
                loss.backward();

                // Accumulate position gradients for densification decisions
                if iter < self.config.densify_until {
                    if let Some(grad) = positions.grad() {
                        grad_accum.accumulate(&grad);
                    }
                }

                optimizer.step();
            }

            let avg_loss = total_loss / self.config.views_per_iter as f32;
            if iter == 0 {
                first_loss = avg_loss;
            }
            best_loss = best_loss.min(avg_loss);

            // Adaptive density control: densification and pruning
            if iter > 0
                && iter < self.config.densify_until
                && iter % self.config.densify_interval == 0
            {
                let grad_avg = grad_accum.average();

                // Densify (clone/split high-gradient splats)
                let (new_pos, new_scales, new_rots, new_colors, new_opacities) =
                    self.densify_gaussians(&positions, &log_scales, &rotations, &colors, &logit_opacities, &grad_avg);

                // Prune (remove low-opacity splats)
                let (final_pos, final_scales, final_rots, final_colors, final_opacities) =
                    self.prune_gaussians(&new_pos, &new_scales, &new_rots, &new_colors, &new_opacities);

                let new_n = final_pos.size()[0];

                if new_n != current_n {
                    tracing::info!(
                        "Densify+Prune at iter {}: {} -> {} splats",
                        iter, current_n, new_n
                    );
                    current_n = new_n;

                    // Recreate VarStore with new parameter count
                    // Note: This creates new optimizer state, which is standard for 3DGS
                    let vs_new = nn::VarStore::new(self.device);
                    let root_new = vs_new.root();

                    // Copy tensors to new VarStore
                    positions = root_new.var_copy("positions", &final_pos);
                    log_scales = root_new.var_copy("log_scales", &final_scales);
                    rotations = root_new.var_copy("rotations", &final_rots);
                    colors = root_new.var_copy("colors", &final_colors);
                    logit_opacities = root_new.var_copy("logit_opacities", &final_opacities.view([new_n]));

                    // Recreate optimizer with new VarStore
                    optimizer = nn::Adam::default().build(&vs_new, self.config.learning_rate)?;

                    // Reset gradient accumulator for new splat count
                    grad_accum.reset(new_n);
                }
            }

            if iter % self.config.log_interval == 0 {
                pb.set_message(format!("{:.4} ({})", avg_loss, current_n));
            }

            pb.inc(1);
        }

        pb.finish_and_clear();
        tracing::info!(
            "Cell trained: start={:.4} -> final={:.4}, splats: {} -> {}",
            first_loss, best_loss, n, current_n
        );

        // Export trained splats - flatten tensors for Vec conversion
        let final_n = current_n as usize;
        let pos: Vec<f32> = Vec::try_from(positions.detach().to_device(Device::Cpu).view([-1])).unwrap();
        let final_scales: Vec<f32> = Vec::try_from(log_scales.exp().detach().to_device(Device::Cpu).view([-1])).unwrap();
        let rot_norm = rotations.pow_tensor_scalar(2).sum_dim_intlist([1i64].as_slice(), true, Kind::Float).sqrt();
        let rots: Vec<f32> = Vec::try_from((&rotations / &rot_norm).detach().to_device(Device::Cpu).view([-1])).unwrap();
        let final_colors: Vec<f32> = Vec::try_from(colors.clamp(0.0, 1.0).detach().to_device(Device::Cpu).view([-1])).unwrap();
        let final_opacities: Vec<f32> = Vec::try_from(logit_opacities.sigmoid().detach().to_device(Device::Cpu).view([-1])).unwrap();

        let splats: Vec<GaussianSplat> = (0..final_n)
            .map(|i| GaussianSplat {
                pos: [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]],
                scale: [final_scales[i * 3], final_scales[i * 3 + 1], final_scales[i * 3 + 2]],
                rotation: [rots[i * 4], rots[i * 4 + 1], rots[i * 4 + 2], rots[i * 4 + 3]],
                color: [final_colors[i * 3], final_colors[i * 3 + 1], final_colors[i * 3 + 2]],
                opacity: final_opacities[i],
            })
            .collect();

        Ok(splats)
    }

    fn estimate_cell_radius(&self, bounds: &CellBounds) -> f32 {
        let dx = (bounds.max.x - bounds.min.x) as f32;
        let dy = (bounds.max.y - bounds.min.y) as f32;
        let dz = (bounds.max.z - bounds.min.z) as f32;
        (dx * dx + dy * dy + dz * dz).sqrt() / 2.0
    }

    /// Train a batch of cells sequentially (for train_universe compatibility)
    pub fn train_cell_batch(
        &self,
        cells: &[(CellEntry, CellData)],
    ) -> Result<Vec<Vec<GaussianSplat>>> {
        let mut results = Vec::with_capacity(cells.len());

        for (entry, cell) in cells {
            tracing::debug!("Batch processing cell: {:?}", entry.id);
            let trained = self.train_cell(cell)?;
            results.push(trained);
        }

        Ok(results)
    }

    /// Densify Gaussians: clone small splats, split large splats with high gradients
    ///
    /// Returns new parameter tensors with potentially more splats
    fn densify_gaussians(
        &self,
        positions: &Tensor,
        log_scales: &Tensor,
        rotations: &Tensor,
        colors: &Tensor,
        logit_opacities: &Tensor,
        grad_avg: &Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let n = positions.size()[0];

        // Identify high-gradient splats
        let high_grad_mask = grad_avg.gt(self.config.grad_threshold_position as f64);
        let high_grad_indices = high_grad_mask.nonzero().squeeze_dim(1);
        let num_high_grad = high_grad_indices.size()[0];

        if num_high_grad == 0 {
            // No densification needed
            return (
                positions.shallow_clone(),
                log_scales.shallow_clone(),
                rotations.shallow_clone(),
                colors.shallow_clone(),
                logit_opacities.shallow_clone(),
            );
        }

        // Get scales for high-gradient splats to decide clone vs split
        let scales = log_scales.exp();
        let max_scales = scales.max_dim(1, false).0; // [N] max scale per splat

        // Clone mask: small splats (max_scale < threshold)
        let clone_threshold = self.config.split_scale_threshold as f64;
        let small_mask = max_scales.lt(clone_threshold);
        let clone_mask = &high_grad_mask & &small_mask;

        // Split mask: large splats (max_scale >= threshold)
        let split_mask = &high_grad_mask & &small_mask.logical_not();

        let clone_indices = clone_mask.nonzero().squeeze_dim(1);
        let split_indices = split_mask.nonzero().squeeze_dim(1);

        let num_clones = clone_indices.size()[0];
        let num_splits = split_indices.size()[0];

        tracing::debug!(
            "Densification: {} high-grad splats -> {} clones, {} splits",
            num_high_grad, num_clones, num_splits
        );

        // Collect new splats
        let mut all_positions = vec![positions.shallow_clone()];
        let mut all_log_scales = vec![log_scales.shallow_clone()];
        let mut all_rotations = vec![rotations.shallow_clone()];
        let mut all_colors = vec![colors.shallow_clone()];
        let mut all_logit_opacities = vec![logit_opacities.shallow_clone()];

        // Clone operation: duplicate with small random perturbation
        if num_clones > 0 {
            let cloned_pos = positions.index_select(0, &clone_indices);
            let cloned_scales = log_scales.index_select(0, &clone_indices);
            let cloned_rots = rotations.index_select(0, &clone_indices);
            let cloned_colors = colors.index_select(0, &clone_indices);
            let cloned_opacities = logit_opacities.index_select(0, &clone_indices);

            // Add small random offset to positions
            let perturbation = Tensor::randn([num_clones, 3], (Kind::Float, self.device)) * 0.01;
            let new_pos = &cloned_pos + &perturbation;

            all_positions.push(new_pos);
            all_log_scales.push(cloned_scales);
            all_rotations.push(cloned_rots);
            all_colors.push(cloned_colors);
            all_logit_opacities.push(cloned_opacities);
        }

        // Split operation: create 2 new splats with half scale, offset along major axis
        if num_splits > 0 {
            let split_pos = positions.index_select(0, &split_indices);
            let split_scales = log_scales.index_select(0, &split_indices);
            let split_rots = rotations.index_select(0, &split_indices);
            let split_colors = colors.index_select(0, &split_indices);
            let split_opacities = logit_opacities.index_select(0, &split_indices);

            // Reduce scale by factor of 1.6 (halving volume)
            let scale_reduction = (1.6f64).ln();
            let new_scales = &split_scales - scale_reduction;

            // Offset positions along major axis (use x-axis simplified)
            let actual_scales = split_scales.exp();
            let offset = &actual_scales * 0.25;

            // Create two copies offset in opposite directions
            let offset_x = offset.select(1, 0).unsqueeze(1);
            let zeros = Tensor::zeros([num_splits, 2], (Kind::Float, self.device));
            let offset_vec = Tensor::cat(&[&offset_x, &zeros], 1);

            let new_pos_1 = &split_pos + &offset_vec;
            let new_pos_2 = &split_pos - &offset_vec;

            // Add both split copies
            all_positions.push(new_pos_1);
            all_log_scales.push(new_scales.shallow_clone());
            all_rotations.push(split_rots.shallow_clone());
            all_colors.push(split_colors.shallow_clone());
            all_logit_opacities.push(split_opacities.shallow_clone());

            all_positions.push(new_pos_2);
            all_log_scales.push(new_scales);
            all_rotations.push(split_rots);
            all_colors.push(split_colors);
            all_logit_opacities.push(split_opacities);
        }

        // Concatenate all tensors
        (
            Tensor::cat(&all_positions, 0),
            Tensor::cat(&all_log_scales, 0),
            Tensor::cat(&all_rotations, 0),
            Tensor::cat(&all_colors, 0),
            Tensor::cat(&all_logit_opacities, 0),
        )
    }

    /// Prune low-opacity Gaussians
    ///
    /// Returns filtered parameter tensors with fewer splats
    fn prune_gaussians(
        &self,
        positions: &Tensor,
        log_scales: &Tensor,
        rotations: &Tensor,
        colors: &Tensor,
        logit_opacities: &Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        // Compute actual opacities
        let opacities = logit_opacities.sigmoid();

        // Keep splats with opacity > threshold
        let keep_mask = opacities.gt(self.config.prune_opacity_threshold as f64);
        let keep_indices = keep_mask.nonzero().squeeze_dim(1);

        let num_kept = keep_indices.size()[0];
        let num_original = positions.size()[0];

        if num_kept == num_original {
            // No pruning needed
            return (
                positions.shallow_clone(),
                log_scales.shallow_clone(),
                rotations.shallow_clone(),
                colors.shallow_clone(),
                logit_opacities.shallow_clone(),
            );
        }

        tracing::debug!(
            "Pruning: {} -> {} splats (removed {} low-opacity)",
            num_original, num_kept, num_original - num_kept
        );

        (
            positions.index_select(0, &keep_indices),
            log_scales.index_select(0, &keep_indices),
            rotations.index_select(0, &keep_indices),
            colors.index_select(0, &keep_indices),
            logit_opacities.index_select(0, &keep_indices),
        )
    }
}

/// Convert image::RgbImage to tch::Tensor [H, W, 3]
fn image_to_torch_tensor(image: &image::RgbImage, device: Device) -> Tensor {
    let (w, h) = image.dimensions();
    let data: Vec<f32> = image
        .pixels()
        .flat_map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0])
        .collect();

    Tensor::from_slice(&data)
        .view([h as i64, w as i64, 3])
        .to_device(device)
}

/// Train all cells in universe using torch backend
pub fn train_universe(
    input_dir: &std::path::Path,
    output_dir: &std::path::Path,
    config: TrainConfig,
    device: TorchDevice,
) -> Result<()> {
    let manifest = CellManifest::load(&input_dir.join("index.json"))?;
    let trainer = TorchTrainer::new(config.clone(), device)?;

    std::fs::create_dir_all(output_dir.join("cells"))?;

    let mut new_manifest = CellManifest::new(manifest.config.clone());

    // Collect all cells for batched training
    let mut all_cells: Vec<(CellEntry, CellData)> = Vec::new();
    for entry in &manifest.cells {
        let cell_path = input_dir.join("cells").join(&entry.file_name);
        let cell = CellData::load(&cell_path)?;
        all_cells.push((entry.clone(), cell));
    }

    tracing::info!(
        "Training {} cells with TorchTrainer on {:?} (batch size: {})",
        all_cells.len(),
        device,
        config.batch_cells
    );

    // Process in batches for better GPU utilization
    let batch_size = config.batch_cells.max(1);
    for batch in all_cells.chunks(batch_size) {
        let trained_batch = trainer.train_cell_batch(batch)?;

        for ((entry, cell), trained_splats) in batch.iter().zip(trained_batch) {
            let mut new_cell = CellData::new(entry.id, cell.metadata.bounds.clone());
            for splat in trained_splats {
                new_cell.add_splat(splat);
            }

            let out_path = output_dir.join("cells").join(&entry.file_name);
            new_cell.save(&out_path)?;

            new_manifest.add_cell(CellEntry {
                id: entry.id,
                file_name: entry.file_name.clone(),
                splat_count: new_cell.metadata.splat_count,
                file_size_bytes: std::fs::metadata(&out_path)?.len(),
            });
        }
    }

    new_manifest.save(&output_dir.join("index.json"))?;

    tracing::info!("Training complete. Output: {:?}", output_dir);
    Ok(())
}

/// Train selected cells in a universe using torch backend (for landmark-focused training)
pub fn train_selected_cells(
    input_dir: &std::path::Path,
    output_dir: &std::path::Path,
    cells_to_train: &[universe_data::CellEntry],
    config: TrainConfig,
    device: TorchDevice,
) -> Result<()> {
    let manifest = CellManifest::load(&input_dir.join("index.json"))?;
    let trainer = TorchTrainer::new(config.clone(), device)?;

    std::fs::create_dir_all(output_dir.join("cells"))?;

    let mut new_manifest = CellManifest::new(manifest.config.clone());

    tracing::info!(
        "Training {} selected cells (out of {} total) with TorchTrainer on {:?}",
        cells_to_train.len(),
        manifest.cells.len(),
        device
    );

    // Collect selected cells for batched training
    let mut selected_cells: Vec<(CellEntry, CellData)> = Vec::new();
    for entry in cells_to_train {
        let cell_path = input_dir.join("cells").join(&entry.file_name);
        let cell = CellData::load(&cell_path)?;
        selected_cells.push((entry.clone(), cell));
    }

    // Process in batches for better GPU utilization
    let batch_size = config.batch_cells.max(1);
    for batch in selected_cells.chunks(batch_size) {
        let trained_batch = trainer.train_cell_batch(batch)?;

        for ((entry, cell), trained_splats) in batch.iter().zip(trained_batch) {
            let mut new_cell = CellData::new(entry.id, cell.metadata.bounds.clone());
            for splat in trained_splats {
                new_cell.add_splat(splat);
            }

            let out_path = output_dir.join("cells").join(&entry.file_name);
            new_cell.save(&out_path)?;

            new_manifest.add_cell(CellEntry {
                id: entry.id,
                file_name: entry.file_name.clone(),
                splat_count: new_cell.metadata.splat_count,
                file_size_bytes: std::fs::metadata(&out_path)?.len(),
            });
        }
    }

    new_manifest.save(&output_dir.join("index.json"))?;

    tracing::info!("Selective training complete. Output: {:?}", output_dir);
    Ok(())
}

fn inverse_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}
