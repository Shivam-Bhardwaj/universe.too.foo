//! Training loop with adaptive density control

use anyhow::Result;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use indicatif::{ProgressBar, ProgressStyle};

use crate::camera::{generate_training_cameras, Camera};
use crate::ground_truth::GroundTruthRenderer;
use crate::loss::combined_loss;
use crate::model::GaussianCloud;
use crate::rasterizer::{render_gaussians, RasterizerConfig};

use universe_core::grid::CellBounds;
use universe_data::{CellData, GaussianSplat};

/// Training configuration
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub learning_rate: f64,
    pub iterations: usize,
    pub views_per_iter: usize,
    pub image_size: (u32, u32),
    pub lambda_dssim: f32,
    pub densify_interval: usize,
    pub densify_until: usize,
    pub prune_opacity_threshold: f32,
    pub grad_threshold_position: f32,
    pub log_interval: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            iterations: 1000,
            views_per_iter: 4,
            image_size: (256, 256),
            lambda_dssim: 0.2,
            densify_interval: 100,
            densify_until: 800,
            prune_opacity_threshold: 0.01,
            grad_threshold_position: 0.0001,
            log_interval: 50,
        }
    }
}

/// Cell trainer
pub struct Trainer<B: AutodiffBackend> {
    config: TrainConfig,
    device: B::Device,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(config: TrainConfig, device: B::Device) -> Self {
        Self { config, device }
    }

    /// Train a single cell
    pub fn train_cell(&self, cell: &CellData) -> Result<Vec<GaussianSplat>> {
        tracing::info!(
            "Training cell ({},{},{}) with {} splats",
            cell.metadata.id.l,
            cell.metadata.id.theta,
            cell.metadata.id.phi,
            cell.splats.len()
        );

        if cell.splats.is_empty() {
            return Ok(vec![]);
        }

        // Extract initial splat data
        let positions: Vec<_> = cell.splats.iter().map(|s| s.pos).collect();
        let scales: Vec<_> = cell.splats.iter().map(|s| s.scale).collect();
        let rotations: Vec<_> = cell.splats.iter().map(|s| s.rotation).collect();
        let colors: Vec<_> = cell.splats.iter().map(|s| s.color).collect();
        let opacities: Vec<_> = cell.splats.iter().map(|s| s.opacity).collect();

        // Create model
        let mut model = GaussianCloud::<B>::from_splats(
            &self.device,
            positions,
            scales,
            rotations,
            colors,
            opacities,
        );

        // Create optimizer
        let mut optim = AdamConfig::new().init();

        // Generate camera views around cell
        let center = glam::Vec3::new(
            cell.metadata.bounds.centroid.x as f32,
            cell.metadata.bounds.centroid.y as f32,
            cell.metadata.bounds.centroid.z as f32,
        );
        let radius = self.estimate_cell_radius(&cell.metadata.bounds);
        let cameras = generate_training_cameras(
            center,
            radius,
            self.config.views_per_iter * 10,
            self.config.image_size,
        );

        // Ground truth renderer
        let gt_renderer = GroundTruthRenderer::new(self.config.image_size.0, self.config.image_size.1);

        // Rasterizer config
        let raster_config = RasterizerConfig {
            image_width: self.config.image_size.0,
            image_height: self.config.image_size.1,
            ..Default::default()
        };

        // Progress bar
        let pb = ProgressBar::new(self.config.iterations as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed}] [{bar:30}] {pos}/{len} loss={msg}")
                .expect("template"),
        );

        // Training loop
        let mut best_loss = f32::INFINITY;

        for iter in 0..self.config.iterations {
            let mut total_loss = 0.0;

            // Sample random camera views
            for view_idx in 0..self.config.views_per_iter {
                let cam_idx = (iter * self.config.views_per_iter + view_idx) % cameras.len();
                let camera = &cameras[cam_idx];

                // Render ground truth
                let gt_image = gt_renderer.render(&cell.splats, camera);
                let gt_tensor = gt_renderer.image_to_tensor::<B>(&gt_image, &self.device);

                // Forward pass
                let rendered = render_gaussians(
                    model.positions.val(),
                    model.scales(),
                    model.normalized_rotations(),
                    model.colors.val().clamp(0.0, 1.0),
                    model.opacities(),
                    camera,
                    &raster_config,
                );

                // Compute loss
                let loss = combined_loss(rendered, gt_tensor, self.config.lambda_dssim);
                let loss_val = loss.clone().into_scalar().elem::<f32>();
                total_loss += loss_val;

                // Backward pass
                let grads = loss.backward();

                // Optimizer step
                let grads_params = GradientsParams::from_grads(grads, &model);
                model = optim.step(self.config.learning_rate, model, grads_params);
            }

            let avg_loss = total_loss / self.config.views_per_iter as f32;
            best_loss = best_loss.min(avg_loss);

            // Logging
            if iter % self.config.log_interval == 0 {
                pb.set_message(format!("{:.4}", avg_loss));
            }

            // Adaptive density control
            if iter > 0
                && iter < self.config.densify_until
                && iter % self.config.densify_interval == 0
            {
                // TODO: Implement densification (clone/split high-gradient splats)
                // TODO: Implement pruning (remove low-opacity splats)
                tracing::debug!("Densify step at iter {}", iter);
            }

            pb.inc(1);
        }

        pb.finish_with_message(format!("final={:.4}", best_loss));

        // Export trained splats
        Ok(model.to_splats())
    }

    fn estimate_cell_radius(&self, bounds: &CellBounds) -> f32 {
        let dx = (bounds.max.x - bounds.min.x) as f32;
        let dy = (bounds.max.y - bounds.min.y) as f32;
        let dz = (bounds.max.z - bounds.min.z) as f32;
        (dx * dx + dy * dy + dz * dz).sqrt() / 2.0
    }
}

/// Train all cells in a universe
pub fn train_universe<B: AutodiffBackend>(
    input_dir: &std::path::Path,
    output_dir: &std::path::Path,
    config: TrainConfig,
    device: B::Device,
) -> Result<()> {
    use universe_data::CellManifest;

    let manifest = CellManifest::load(&input_dir.join("index.json"))?;
    let trainer = Trainer::<B>::new(config, device);

    std::fs::create_dir_all(output_dir.join("cells"))?;

    let mut new_manifest = CellManifest::new(manifest.config.clone());

    tracing::info!("Training {} cells", manifest.cells.len());

    for entry in &manifest.cells {
        let cell_path = input_dir.join("cells").join(&entry.file_name);
        let cell = CellData::load(&cell_path)?;

        let trained_splats = trainer.train_cell(&cell)?;

        // Save trained cell
        let mut new_cell = CellData::new(entry.id, cell.metadata.bounds.clone());
        for splat in trained_splats {
            new_cell.add_splat(splat);
        }

        let out_path = output_dir.join("cells").join(&entry.file_name);
        new_cell.save(&out_path)?;

        new_manifest.add_cell(universe_data::CellEntry {
            id: entry.id,
            file_name: entry.file_name.clone(),
            splat_count: new_cell.metadata.splat_count,
            file_size_bytes: std::fs::metadata(&out_path)?.len(),
        });
    }

    new_manifest.save(&output_dir.join("index.json"))?;

    tracing::info!("Training complete. Output: {:?}", output_dir);
    Ok(())
}

/// Train selected cells in a universe (for landmark-focused training)
pub fn train_selected_cells<B: AutodiffBackend>(
    input_dir: &std::path::Path,
    output_dir: &std::path::Path,
    cells_to_train: &[universe_data::CellEntry],
    config: TrainConfig,
    device: B::Device,
) -> Result<()> {
    use universe_data::CellManifest;

    let manifest = CellManifest::load(&input_dir.join("index.json"))?;
    let trainer = Trainer::<B>::new(config, device);

    std::fs::create_dir_all(output_dir.join("cells"))?;

    let mut new_manifest = CellManifest::new(manifest.config.clone());

    tracing::info!("Training {} selected cells (out of {} total)", cells_to_train.len(), manifest.cells.len());

    for entry in cells_to_train {
        let cell_path = input_dir.join("cells").join(&entry.file_name);
        let cell = CellData::load(&cell_path)?;

        let trained_splats = trainer.train_cell(&cell)?;

        // Save trained cell
        let mut new_cell = CellData::new(entry.id, cell.metadata.bounds.clone());
        for splat in trained_splats {
            new_cell.add_splat(splat);
        }

        let out_path = output_dir.join("cells").join(&entry.file_name);
        new_cell.save(&out_path)?;

        new_manifest.add_cell(universe_data::CellEntry {
            id: entry.id,
            file_name: entry.file_name.clone(),
            splat_count: new_cell.metadata.splat_count,
            file_size_bytes: std::fs::metadata(&out_path)?.len(),
        });
    }

    new_manifest.save(&output_dir.join("index.json"))?;

    tracing::info!("Selective training complete. Output: {:?}", output_dir);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_autodiff::Autodiff;
    use burn::backend::NdArray;
    use glam::Vec3;
    use universe_data::GaussianSplat;

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_training_loop_convergence() {
        // Test that the complete training loop can converge on a simple scene
        let device = Default::default();

        // Create simple training config
        let config = TrainConfig {
            learning_rate: 0.01,
            iterations: 50,
            views_per_iter: 2,
            image_size: (64, 64),
            lambda_dssim: 0.0,  // Pure L1 loss for simplicity
            densify_interval: 100,
            densify_until: 0,  // No densification in test
            prune_opacity_threshold: 0.01,
            grad_threshold_position: 0.0001,
            log_interval: 10,
        };

        // Create simple scene: 3 Gaussians at different positions
        let splats = vec![
            GaussianSplat {
                pos: [0.0, 0.0, 0.0],
                scale: [0.3, 0.3, 0.3],
                rotation: [0.0, 0.0, 0.0, 1.0],
                color: [1.0, 0.0, 0.0],  // Red
                opacity: 0.9,
            },
            GaussianSplat {
                pos: [0.5, 0.0, 0.0],
                scale: [0.3, 0.3, 0.3],
                rotation: [0.0, 0.0, 0.0, 1.0],
                color: [0.0, 1.0, 0.0],  // Green
                opacity: 0.9,
            },
            GaussianSplat {
                pos: [-0.5, 0.0, 0.0],
                scale: [0.3, 0.3, 0.3],
                rotation: [0.0, 0.0, 0.0, 1.0],
                color: [0.0, 0.0, 1.0],  // Blue
                opacity: 0.9,
            },
        ];

        // Create model with perturbed initial parameters
        let positions: Vec<_> = splats.iter().map(|s| {
            [s.pos[0] + 0.1, s.pos[1] + 0.1, s.pos[2]]  // Offset positions
        }).collect();
        let scales: Vec<_> = splats.iter().map(|s| s.scale).collect();
        let rotations: Vec<_> = splats.iter().map(|s| s.rotation).collect();
        let colors: Vec<_> = splats.iter().map(|s| {
            [0.5, 0.5, 0.5]  // Start with gray (should learn colors)
        }).collect();
        let opacities: Vec<_> = splats.iter().map(|s| s.opacity).collect();

        let mut model = GaussianCloud::<TestBackend>::from_splats(
            &device,
            positions,
            scales,
            rotations,
            colors,
            opacities,
        );

        // Create optimizer
        let mut optim = AdamConfig::new().init();

        // Generate cameras
        let center = Vec3::ZERO;
        let radius = 2.0;
        let cameras = generate_training_cameras(center, radius, 10, config.image_size);

        // Ground truth renderer
        let gt_renderer = GroundTruthRenderer::new(config.image_size.0, config.image_size.1);

        // Rasterizer config
        let raster_config = RasterizerConfig {
            image_width: config.image_size.0,
            image_height: config.image_size.1,
            ..Default::default()
        };

        // Track initial and final loss
        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        println!("\nTraining 3-Gaussian scene:");

        // Training loop
        for iter in 0..config.iterations {
            let mut total_loss = 0.0;

            for view_idx in 0..config.views_per_iter {
                let cam_idx = (iter * config.views_per_iter + view_idx) % cameras.len();
                let camera = &cameras[cam_idx];

                // Ground truth
                let gt_image = gt_renderer.render(&splats, camera);
                let gt_tensor = gt_renderer.image_to_tensor::<TestBackend>(&gt_image, &device);

                // Forward pass
                let rendered = render_gaussians(
                    model.positions.val(),
                    model.scales(),
                    model.normalized_rotations(),
                    model.colors.val().clamp(0.0, 1.0),
                    model.opacities(),
                    camera,
                    &raster_config,
                );

                // Loss (L1)
                let loss = (rendered - gt_tensor).abs().mean();
                let loss_val = loss.clone().into_scalar().elem::<f32>();
                total_loss += loss_val;

                // Backward pass
                let grads = loss.backward();

                // Optimizer step
                let grads_params = GradientsParams::from_grads(grads, &model);
                model = optim.step(config.learning_rate, model, grads_params);
            }

            let avg_loss = total_loss / config.views_per_iter as f32;

            if iter == 0 {
                initial_loss = avg_loss;
            }
            final_loss = avg_loss;

            if iter % config.log_interval == 0 {
                println!("  Iter {}: loss={:.6}", iter, avg_loss);
            }
        }

        println!("Initial loss: {:.6}", initial_loss);
        println!("Final loss:   {:.6}", final_loss);

        // Verify convergence
        assert!(initial_loss > 0.0, "Initial loss should be non-zero");
        assert!(final_loss < initial_loss * 0.8,
            "Loss should decrease by at least 20%: {:.6} -> {:.6}",
            initial_loss, final_loss);

        // Verify model can export splats
        let trained_splats = model.to_splats();
        assert_eq!(trained_splats.len(), 3, "Should have 3 splats");

        println!("\nTraining converged successfully!");
    }
}
