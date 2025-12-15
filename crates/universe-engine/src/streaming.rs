//! Streaming policy and cell selection
//! Phase 1.3: Tile selection + LRU in the engine

use universe_core::grid::CellId;
use universe_data::CellManifest;
use glam::DVec3;

/// Streaming policy: selects visible cells based on camera frustum
pub struct StreamingPolicy {
    manifest: Option<CellManifest>,
}

impl StreamingPolicy {
    pub fn new(manifest: Option<CellManifest>) -> Self {
        Self { manifest }
    }

    /// Phase 1.3: Get visible cells based on camera frustum and distance
    /// Implements streaming policy: select cells that intersect the view frustum
    /// and are within reasonable distance, prioritizing closer cells.
    pub fn get_visible_cells(
        &self,
        camera_pos: DVec3,
        camera_forward: DVec3,
        fov_y: f32,
        max_distance: f64,
    ) -> Vec<CellId> {
        match &self.manifest {
            Some(manifest) => {
                let mut visible_with_dist: Vec<(CellId, f64)> = Vec::new();

                // Compute frustum planes (simplified: use distance + angle check)
                let half_fov = (fov_y / 2.0) as f64;
                let cos_half_fov = half_fov.cos();

                // Compute bounds from cell IDs (efficient, no file I/O)
                let grid = universe_core::grid::HLGGrid::new(manifest.config.clone());

                for entry in &manifest.cells {
                    // Compute cell bounds from ID (no file I/O needed)
                    let bounds = grid.cell_to_bounds(entry.id);
                    let cell_center = DVec3::new(
                        bounds.centroid.x,
                        bounds.centroid.y,
                        bounds.centroid.z,
                    );

                    // Distance check
                    let to_cell = cell_center - camera_pos;
                    let dist = to_cell.length();

                    if dist > max_distance {
                        continue;
                    }

                    // Frustum check: cell center should be within view cone
                    if dist > 1e-6 {
                        let dir_to_cell = to_cell / dist;
                        let dot = camera_forward.dot(dir_to_cell);

                        // Approximate: if cell is within FOV cone
                        if dot >= cos_half_fov {
                            visible_with_dist.push((entry.id, dist));
                        }
                    }
                }

                // Sort by distance (closer first) for LRU priority
                visible_with_dist.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });

                visible_with_dist.into_iter().map(|(id, _)| id).collect()
            }
            None => Vec::new(),
        }
    }
}
