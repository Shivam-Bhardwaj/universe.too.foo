//! Data ingestion pipeline: bins astronomical data into HLG cells

use crate::splat::GaussianSplat;
use crate::cell::CellData;
use crate::manifest::{CellManifest, CellEntry};
use crate::ephemeris::{EphemerisProvider, SolarSystemBody};
use crate::stars::{StarCatalog, StarRecord};

use universe_core::grid::{HLGGrid, HLGConfig, CellId};

use hifitime::Epoch;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use anyhow::Result;

/// Pipeline configuration
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub grid_config: HLGConfig,
    pub min_opacity: f32,
    pub max_splats_per_cell: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            grid_config: HLGConfig::default(),
            min_opacity: 0.001,
            max_splats_per_cell: 500_000,
        }
    }
}

/// Main data pipeline
pub struct DataPipeline {
    config: PipelineConfig,
    grid: HLGGrid,
}

impl DataPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let grid = HLGGrid::new(config.grid_config.clone());
        Self { config, grid }
    }

    pub fn with_defaults() -> Self {
        Self::new(PipelineConfig::default())
    }

    /// Ingest star catalog into HLG cells
    pub fn ingest_stars(&self, catalog: &StarCatalog, output_dir: &Path) -> Result<CellManifest> {
        tracing::info!("Ingesting {} stars", catalog.len());

        let cells_dir = output_dir.join("cells");
        std::fs::create_dir_all(&cells_dir)?;

        // Progress bar
        let pb = ProgressBar::new(catalog.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40}] {pos}/{len} ({per_sec})")
            .expect("template error"));

        // Parallel binning
        let bins: Mutex<HashMap<CellId, Vec<GaussianSplat>>> = Mutex::new(HashMap::new());

        catalog.iter().par_bridge().for_each(|star| {
            if let Some(splat_data) = self.process_star(star) {
                let (cell_id, splat) = splat_data;
                bins.lock().unwrap().entry(cell_id).or_default().push(splat);
            }
            pb.inc(1);
        });

        pb.finish_with_message("Binning complete");

        // Write cells
        let bins = bins.into_inner().unwrap();
        let mut manifest = CellManifest::new(self.config.grid_config.clone());

        tracing::info!("Writing {} cells", bins.len());

        for (cell_id, mut splats) in bins {
            // Limit per cell
            if splats.len() > self.config.max_splats_per_cell {
                splats.sort_by(|a, b| b.opacity.partial_cmp(&a.opacity).unwrap());
                splats.truncate(self.config.max_splats_per_cell);
            }

            let bounds = self.grid.cell_to_bounds(cell_id);
            let mut cell = CellData::new(cell_id, bounds);
            for s in splats { cell.add_splat(s); }

            let path = cells_dir.join(cell_id.file_name());
            cell.save(&path)?;

            manifest.add_cell(CellEntry {
                id: cell_id,
                file_name: cell_id.file_name(),
                splat_count: cell.metadata.splat_count,
                file_size_bytes: std::fs::metadata(&path)?.len(),
            });
        }

        manifest.save(&output_dir.join("index.json"))?;

        tracing::info!("Created {} cells, {} splats, {:.1} MB",
            manifest.cells.len(), manifest.total_splats,
            manifest.total_size_bytes as f64 / 1e6);

        Ok(manifest)
    }

    /// Generate planetary splats at epoch
    pub fn generate_planets(&self, eph: &EphemerisProvider, epoch: Epoch, output_dir: &Path) -> Result<CellManifest> {
        let cells_dir = output_dir.join("cells");
        std::fs::create_dir_all(&cells_dir)?;

        let mut manifest = CellManifest::new(self.config.grid_config.clone());

        for body in SolarSystemBody::planets() {
            let pos = eph.body_position(*body, epoch)?;

            if let Some(cell_id) = self.grid.cartesian_to_cell(pos) {
                let bounds = self.grid.cell_to_bounds(cell_id);

                let splat = GaussianSplat::sphere(
                    [(pos.x - bounds.centroid.x) as f32,
                     (pos.y - bounds.centroid.y) as f32,
                     (pos.z - bounds.centroid.z) as f32],
                    body.radius_m() as f32,
                    body.color(),
                    1.0,
                );

                let path = cells_dir.join(cell_id.file_name());
                let mut cell = if path.exists() {
                    CellData::load(&path)?
                } else {
                    CellData::new(cell_id, bounds)
                };

                cell.add_splat(splat);
                cell.save(&path)?;

                manifest.add_cell(CellEntry {
                    id: cell_id,
                    file_name: cell_id.file_name(),
                    splat_count: cell.metadata.splat_count,
                    file_size_bytes: std::fs::metadata(&path)?.len(),
                });

                tracing::info!("{}: Cell ({},{},{})", body.name(), cell_id.l, cell_id.theta, cell_id.phi);
            }
        }

        Ok(manifest)
    }

    /// Ingest landmarks (galaxies, nebulae, clusters, etc.) into HLG cells
    pub fn ingest_landmarks(&self, landmarks: &[crate::landmarks::Landmark], output_dir: &Path) -> Result<CellManifest> {
        tracing::info!("Ingesting {} landmarks", landmarks.len());

        let cells_dir = output_dir.join("cells");
        std::fs::create_dir_all(&cells_dir)?;

        let mut manifest = CellManifest::new(self.config.grid_config.clone());
        let r_min = self.config.grid_config.r_min;

        for landmark in landmarks {
            // Skip objects inside R_MIN (Sun is procedural, no need to include as splat)
            if landmark.distance() < r_min {
                tracing::debug!("Skipping landmark '{}' (inside R_MIN)", landmark.name);
                continue;
            }

            // Convert landmark to cell + splat
            if let Some((cell_id, splat)) = self.process_landmark(landmark) {
                let path = cells_dir.join(cell_id.file_name());

                // Load existing cell if present, or create new one
                let mut cell = if path.exists() {
                    CellData::load(&path)?
                } else {
                    let bounds = self.grid.cell_to_bounds(cell_id);
                    CellData::new(cell_id, bounds)
                };

                cell.add_splat(splat);
                cell.save(&path)?;

                // Update manifest
                manifest.add_cell(CellEntry {
                    id: cell_id,
                    file_name: cell_id.file_name(),
                    splat_count: cell.metadata.splat_count,
                    file_size_bytes: std::fs::metadata(&path)?.len(),
                });

                tracing::info!("Landmark '{}': Cell ({},{},{})",
                    landmark.name, cell_id.l, cell_id.theta, cell_id.phi);
            } else {
                tracing::warn!("Failed to process landmark '{}'", landmark.name);
            }
        }

        tracing::info!("Landmark ingestion complete: {} cells, {} splats",
            manifest.cells.len(), manifest.total_splats);

        Ok(manifest)
    }

    fn process_landmark(&self, landmark: &crate::landmarks::Landmark) -> Option<(CellId, GaussianSplat)> {
        use universe_core::coordinates::CartesianPosition;

        let pos = CartesianPosition {
            x: landmark.pos_meters.x,
            y: landmark.pos_meters.y,
            z: landmark.pos_meters.z,
        };

        let cell_id = self.grid.cartesian_to_cell(pos)?;
        let bounds = self.grid.cell_to_bounds(cell_id);

        let visual_radius = landmark.visual_radius() as f32;
        let (color, opacity) = landmark.visual_appearance();

        let splat = GaussianSplat::sphere(
            [(pos.x - bounds.centroid.x) as f32,
             (pos.y - bounds.centroid.y) as f32,
             (pos.z - bounds.centroid.z) as f32],
            visual_radius,
            color,
            opacity,
        );

        Some((cell_id, splat))
    }

    fn process_star(&self, star: &StarRecord) -> Option<(CellId, GaussianSplat)> {
        let pos = star.to_cartesian()?;
        let cell_id = self.grid.cartesian_to_cell(pos)?;
        let bounds = self.grid.cell_to_bounds(cell_id);

        let opacity = magnitude_to_opacity(star.phot_g_mean_mag);
        if opacity < self.config.min_opacity { return None; }

        // Stars are effectively point sources at astronomical distances.
        // Using physical radius would make them sub-pixel and invisible, so we encode a *visual* radius
        // that approximates a constant screen-space size under a canonical camera.
        //
        // Canonical camera (matches default client/server-ish settings):
        // - vertical FOV: 60°
        // - render height: 720px
        const CANON_FOV_Y_RAD: f32 = std::f32::consts::FRAC_PI_3; // 60°
        const CANON_HEIGHT_PX: f32 = 720.0;
        let focal_px = (CANON_HEIGHT_PX / 2.0) / (CANON_FOV_Y_RAD / 2.0).tan();

        // Derive a \"pixel radius\" from brightness (opacity already tracks magnitude).
        // Brighter stars get a larger footprint, but we clamp to keep them reasonable.
        let radius_px = (1.0 + 6.0 * opacity.sqrt()).clamp(1.0, 8.0);

        // Convert to an angular radius (radians), then to a world-space radius at the star's distance.
        let theta = radius_px / focal_px;
        let dist_m = pos.magnitude() as f32;
        let visual_radius_m = (dist_m * theta).max(1.0);

        let splat = GaussianSplat::sphere(
            [(pos.x - bounds.centroid.x) as f32,
             (pos.y - bounds.centroid.y) as f32,
             (pos.z - bounds.centroid.z) as f32],
            visual_radius_m,
            star.color_rgb(),
            opacity,
        );

        Some((cell_id, splat))
    }
}

fn magnitude_to_opacity(mag: f64) -> f32 {
    (10.0_f64.powf(-mag / 5.0) as f32).clamp(0.001, 1.0)
}

/// Merge manifests
pub fn merge_manifests(mut manifests: Vec<CellManifest>) -> Result<CellManifest> {
    let mut base = manifests.pop().ok_or_else(|| anyhow::anyhow!("No manifests"))?;
    for m in manifests { base.merge(m); }
    Ok(base)
}
