//! Tile-based Gaussian Splatting rasterization
//!
//! This crate provides the core mathematics and shader code for proper 3D Gaussian
//! Splatting with tile-based sorting and alpha compositing.
//!
//! # Architecture
//!
//! The rendering pipeline consists of 5 compute passes:
//! 1. **Project & Cull**: Transform 3D Gaussians to 2D screen-space ellipses
//! 2. **Tile Assignment**: Assign each splat to overlapping 16x16 tiles
//! 3. **Radix Sort**: Sort splats by (tile_id, depth) key
//! 4. **Tile Ranges**: Compute start/count for each tile
//! 5. **Rasterize**: Per-tile front-to-back alpha blending

pub mod covariance;
pub mod gpu_types;

// Re-export shader source strings
pub mod shaders {
    pub const PROJECT: &str = include_str!("shaders/project.wgsl");
    pub const TILE_ASSIGN: &str = include_str!("shaders/tile_assign.wgsl");
    pub const SORT: &str = include_str!("shaders/sort.wgsl");
    pub const TILE_RANGES: &str = include_str!("shaders/tile_ranges.wgsl");
    pub const RASTER_TILE: &str = include_str!("shaders/raster_tile.wgsl");
}

pub use covariance::*;
pub use gpu_types::*;
