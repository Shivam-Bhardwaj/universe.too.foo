//! VRAM streaming and caching

use universe_core::grid::CellId;
use universe_data::{CellData, CellManifest};
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;
use anyhow::Result;

/// LRU cache for cell data in system RAM
pub struct CellCache {
    /// Cached cells
    cells: HashMap<CellId, Arc<CellData>>,
    /// LRU order (front = oldest)
    lru_order: VecDeque<CellId>,
    /// Maximum cells to cache
    max_cells: usize,
    /// Base path for cell files
    cells_dir: std::path::PathBuf,
}

impl CellCache {
    pub fn new(cells_dir: &Path, max_cells: usize) -> Self {
        Self {
            cells: HashMap::new(),
            lru_order: VecDeque::new(),
            max_cells,
            cells_dir: cells_dir.to_path_buf(),
        }
    }

    /// Get cell, loading from disk if needed
    pub fn get(&mut self, id: CellId) -> Result<Arc<CellData>> {
        // Check cache
        if let Some(cell) = self.cells.get(&id) {
            // Move to back of LRU
            self.lru_order.retain(|&x| x != id);
            self.lru_order.push_back(id);
            return Ok(Arc::clone(cell));
        }

        // Load from disk
        let file_name = id.file_name();
        let path = self.cells_dir.join(&file_name);
        let cell = CellData::load(&path)?;
        let cell = Arc::new(cell);

        // Evict if at capacity
        while self.cells.len() >= self.max_cells {
            if let Some(old_id) = self.lru_order.pop_front() {
                self.cells.remove(&old_id);
            }
        }

        // Insert
        self.cells.insert(id, Arc::clone(&cell));
        self.lru_order.push_back(id);

        Ok(cell)
    }

    /// Prefetch cells (non-blocking hint)
    pub fn prefetch(&mut self, ids: &[CellId]) {
        for &id in ids {
            if !self.cells.contains_key(&id) {
                // Could spawn async load here
                let _ = self.get(id);
            }
        }
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.cells.clear();
        self.lru_order.clear();
    }
}

/// GPU buffer region
#[derive(Clone, Debug)]
pub struct BufferRegion {
    pub offset: u64,
    pub size: u64,
    pub splat_count: u32,
}

/// VRAM cache for splat data
pub struct GpuCache {
    /// Main splat buffer
    pub buffer: wgpu::Buffer,
    /// Buffer capacity in bytes
    capacity: u64,
    /// Allocated regions per cell
    allocations: HashMap<CellId, BufferRegion>,
    /// Free regions (simple bump allocator with defrag)
    next_offset: u64,
    /// LRU for eviction
    lru_order: VecDeque<CellId>,
}

impl GpuCache {
    pub fn new(device: &wgpu::Device, capacity_bytes: u64) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Splat Buffer"),
            size: capacity_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            capacity: capacity_bytes,
            allocations: HashMap::new(),
            next_offset: 0,
            lru_order: VecDeque::new(),
        }
    }

    /// Check if cell is resident
    pub fn is_resident(&self, id: CellId) -> bool {
        self.allocations.contains_key(&id)
    }

    /// Get region for cell (if resident)
    pub fn get_region(&self, id: CellId) -> Option<&BufferRegion> {
        self.allocations.get(&id)
    }

    /// Upload cell data to GPU
    pub fn upload(
        &mut self,
        queue: &wgpu::Queue,
        id: CellId,
        data: &[u8],
        splat_count: u32,
    ) -> Result<BufferRegion> {
        let size = data.len() as u64;

        // Check if we need to evict
        while self.next_offset + size > self.capacity {
            if let Some(old_id) = self.lru_order.pop_front() {
                // Simple eviction: just remove from tracking
                // In production, would defragment or use proper allocator
                if let Some(region) = self.allocations.remove(&old_id) {
                    // If evicting from end, reclaim space
                    if region.offset + region.size == self.next_offset {
                        self.next_offset = region.offset;
                    }
                }
            } else {
                anyhow::bail!("GPU buffer full, cannot allocate {} bytes", size);
            }
        }

        let region = BufferRegion {
            offset: self.next_offset,
            size,
            splat_count,
        };

        // Write to buffer
        queue.write_buffer(&self.buffer, region.offset, data);

        // Track allocation
        self.allocations.insert(id, region.clone());
        self.lru_order.push_back(id);
        self.next_offset += size;

        Ok(region)
    }

    /// Total resident splats
    pub fn total_splats(&self) -> u32 {
        self.allocations.values().map(|r| r.splat_count).sum()
    }

    /// Reset allocator (call when defragmenting)
    pub fn reset(&mut self) {
        self.allocations.clear();
        self.lru_order.clear();
        self.next_offset = 0;
    }
}

/// Streaming manager combining CPU and GPU caches
pub struct StreamingManager {
    pub cell_cache: CellCache,
    pub manifest: Option<CellManifest>,
}

impl StreamingManager {
    pub fn new(universe_dir: &Path, max_cpu_cells: usize) -> Result<Self> {
        // Try to load manifest, but don't fail if it doesn't exist
        let manifest_path = universe_dir.join("index.json");
        let manifest = if manifest_path.exists() {
            match CellManifest::load(&manifest_path) {
                Ok(m) => {
                    tracing::info!("Loaded universe manifest with {} cells", m.cells.len());
                    Some(m)
                }
                Err(e) => {
                    tracing::warn!("Failed to load manifest: {}", e);
                    None
                }
            }
        } else {
            tracing::info!("No universe data found at {:?}, rendering planets only", universe_dir);
            None
        };

        // Cells are stored in `universe_dir/cells/` by the ingestion pipeline.
        // Support a legacy flat layout (cells directly under `universe_dir`) for compatibility.
        let preferred_cells_dir = universe_dir.join("cells");
        let cells_dir = if preferred_cells_dir.is_dir() {
            preferred_cells_dir
        } else {
            universe_dir.to_path_buf()
        };
        if cells_dir == universe_dir {
            tracing::warn!(
                "Using legacy flat cell layout under {:?}. Prefer {:?}/cells/ for new datasets.",
                universe_dir,
                universe_dir
            );
        }

        let cell_cache = CellCache::new(&cells_dir, max_cpu_cells);

        Ok(Self { cell_cache, manifest })
    }

    /// Phase 1.3: Get visible cells based on camera frustum and distance
    /// Implements streaming policy: select cells that intersect the view frustum
    /// and are within reasonable distance, prioritizing closer cells.
    pub fn get_visible_cells_frustum(&self, camera_pos: glam::DVec3, camera_forward: glam::DVec3, 
                            _camera_up: glam::DVec3, fov_y: f32, _aspect: f32, 
                            max_distance: f64) -> Vec<CellId> {
        match &self.manifest {
            Some(manifest) => {
                let mut visible_with_dist: Vec<(CellId, f64)> = Vec::new();
                
                // Compute frustum planes (simplified: use distance + angle check)
                // For astronomical scales, we use a cone-based culling
                let half_fov = (fov_y / 2.0) as f64;
                let cos_half_fov = half_fov.cos();
                
                // Compute bounds from cell IDs (efficient, no file I/O)
                let grid = universe_core::grid::HLGGrid::new(manifest.config.clone());
                
                for entry in &manifest.cells {
                    // Compute cell bounds from ID (no file I/O needed)
                    let bounds = grid.cell_to_bounds(entry.id);
                    let cell_center = glam::DVec3::new(
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
    
    /// Legacy method for compatibility (returns all cells)
    pub fn get_visible_cells(&self) -> Vec<CellId> {
        match &self.manifest {
            Some(manifest) => manifest.cells.iter().map(|entry| entry.id).collect(),
            None => Vec::new(),
        }
    }
}
