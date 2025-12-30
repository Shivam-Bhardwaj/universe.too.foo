//! TorchTrainer - CUDA-accelerated training for Gaussian splatting

use anyhow::{bail, Result};
use indicatif::{ProgressBar, ProgressStyle};
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::camera::{generate_training_cameras, Camera};
use crate::ground_truth::GroundTruthRenderer;
use crate::TrainConfig;

use super::loss::combined_loss;
use super::rasterizer::render_gaussians;

use universe_core::grid::CellBounds;
use universe_data::{CellData, CellEntry, CellManifest, GaussianSplat};

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

                // Compute loss
                let loss = combined_loss(&rendered, &gt_tensor, self.config.lambda_dssim);

                let loss_val: f32 = loss.double_value(&[]) as f32;
                total_loss += loss_val;

                // Backward pass and optimizer step
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
            }

            let avg_loss = total_loss / self.config.views_per_iter as f32;
            if iter == 0 {
                first_loss = avg_loss;
            }
            best_loss = best_loss.min(avg_loss);

            if iter % self.config.log_interval == 0 {
                pb.set_message(format!("{:.4}", avg_loss));
            }

            pb.inc(1);
        }

        pb.finish_and_clear();
        tracing::info!("Cell trained: start={:.4} -> final={:.4}", first_loss, best_loss);

        // Export trained splats - flatten tensors for Vec conversion
        let n = cell.splats.len();
        let pos: Vec<f32> = Vec::try_from(positions.detach().to_device(Device::Cpu).view([-1])).unwrap();
        let final_scales: Vec<f32> = Vec::try_from(log_scales.exp().detach().to_device(Device::Cpu).view([-1])).unwrap();
        let rot_norm = rotations.pow_tensor_scalar(2).sum_dim_intlist([1i64].as_slice(), true, Kind::Float).sqrt();
        let rots: Vec<f32> = Vec::try_from((&rotations / &rot_norm).detach().to_device(Device::Cpu).view([-1])).unwrap();
        let final_colors: Vec<f32> = Vec::try_from(colors.clamp(0.0, 1.0).detach().to_device(Device::Cpu).view([-1])).unwrap();
        let final_opacities: Vec<f32> = Vec::try_from(logit_opacities.sigmoid().detach().to_device(Device::Cpu).view([-1])).unwrap();

        let splats: Vec<GaussianSplat> = (0..n)
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

fn inverse_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}
