//! GPU-compatible data types for tile-based Gaussian rasterization

use bytemuck::{Pod, Zeroable};

/// Tile size in pixels (16x16 = 256 threads per workgroup)
pub const TILE_SIZE: u32 = 16;

/// Maximum splats that can overlap a single tile
pub const MAX_SPLATS_PER_TILE: u32 = 512;

/// Early termination alpha threshold
pub const ALPHA_THRESHOLD: f32 = 0.99;

/// 2D projected Gaussian data for rendering
///
/// This is the output of the projection compute shader, containing
/// all the information needed to render a Gaussian as a 2D ellipse.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Splat2D {
    /// Screen position in pixels (x, y)
    pub center: [f32; 2],        // 8 bytes (offset 0)
    /// Inverse covariance in conic form (a, b, c)
    /// Gaussian = exp(-0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2))
    pub conic: [f32; 3],         // 12 bytes (offset 8)
    /// Logarithmic depth for sorting (higher = closer in reverse-Z)
    pub depth: f32,              // 4 bytes (offset 20)
    /// RGB color [0, 1]
    pub color: [f32; 3],         // 12 bytes (offset 24)
    /// Opacity [0, 1]
    pub opacity: f32,            // 4 bytes (offset 36)
    /// Bounding radius in pixels (3 sigma)
    pub radius: f32,             // 4 bytes (offset 40)
    /// Padding to 64 bytes (need 5 more f32s = 20 bytes)
    pub _pad: [f32; 5],          // 20 bytes (offset 44)
    // Total: 64 bytes
}

impl Splat2D {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Create a "culled" splat that will be ignored during rendering
    pub fn culled() -> Self {
        Self {
            center: [0.0, 0.0],
            conic: [0.0, 0.0, 0.0],
            depth: 0.0,
            color: [0.0, 0.0, 0.0],
            opacity: 0.0, // Zero opacity means culled
            radius: 0.0,
            _pad: [0.0; 5],
        }
    }
}

/// Key-value pair for tile sorting
///
/// The key encodes both tile ID and depth, allowing a single radix sort
/// to group splats by tile and sort by depth within each tile.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TileKey {
    /// Upper 32 bits: tile_id, Lower 32 bits: depth_bits
    /// This allows sorting by tile first, then by depth within tile
    pub key_high: u32, // tile_id
    pub key_low: u32,  // depth (as u32 bits)
    /// Index into the Splat2D buffer
    pub splat_idx: u32,
    /// Padding
    pub _pad: u32,
}

impl TileKey {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Create a sort key from tile ID and depth
    pub fn new(tile_id: u32, depth: f32, splat_idx: u32) -> Self {
        // Convert depth to u32 for bitwise sorting
        // For reverse-Z, higher depth values are closer, so we want ascending sort
        let depth_bits = depth.to_bits();
        Self {
            key_high: tile_id,
            key_low: depth_bits,
            splat_idx,
            _pad: 0,
        }
    }
}

/// Per-tile range information
///
/// After sorting, this tells us where each tile's splats are in the sorted buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct TileRange {
    /// Starting index in the sorted TileKey buffer
    pub start: u32,
    /// Number of splats in this tile
    pub count: u32,
}

impl TileRange {
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

/// Camera parameters extended for tile-based rasterization
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RasterCamera {
    /// View matrix
    pub view: [[f32; 4]; 4],
    /// Projection matrix
    pub proj: [[f32; 4]; 4],
    /// Combined view-projection matrix
    pub view_proj: [[f32; 4]; 4],
    /// Camera position (always 0,0,0 due to floating origin)
    pub position: [f32; 3],
    pub _pad0: f32,
    /// Near plane distance
    pub near: f32,
    /// Far plane distance
    pub far: f32,
    /// Vertical field of view in radians
    pub fov_y: f32,
    /// Logarithmic depth coefficient
    pub log_depth_c: f32,
    /// Focal length X in pixels
    pub focal_x: f32,
    /// Focal length Y in pixels
    pub focal_y: f32,
    /// Screen width in pixels
    pub width: f32,
    /// Screen height in pixels
    pub height: f32,
}

impl RasterCamera {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Compute number of tiles for current screen size
    pub fn num_tiles(&self) -> (u32, u32) {
        let tiles_x = (self.width as u32 + TILE_SIZE - 1) / TILE_SIZE;
        let tiles_y = (self.height as u32 + TILE_SIZE - 1) / TILE_SIZE;
        (tiles_x, tiles_y)
    }

    /// Compute total number of tiles
    pub fn total_tiles(&self) -> u32 {
        let (tx, ty) = self.num_tiles();
        tx * ty
    }
}

/// Tile assignment parameters
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TileParams {
    /// Number of tiles in X direction
    pub num_tiles_x: u32,
    /// Number of tiles in Y direction
    pub num_tiles_y: u32,
    /// Screen width in pixels
    pub screen_width: f32,
    /// Screen height in pixels
    pub screen_height: f32,
}

impl TileParams {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    pub fn new(width: u32, height: u32) -> Self {
        Self {
            num_tiles_x: (width + TILE_SIZE - 1) / TILE_SIZE,
            num_tiles_y: (height + TILE_SIZE - 1) / TILE_SIZE,
            screen_width: width as f32,
            screen_height: height as f32,
        }
    }
}

/// Dispatch parameters for compute shaders
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DispatchParams {
    /// Total number of splats
    pub num_splats: u32,
    /// Total number of tiles
    pub num_tiles: u32,
    /// Total number of tile keys (after assignment)
    pub num_keys: u32,
    pub _pad: u32,
}

impl DispatchParams {
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        // Verify struct sizes for GPU alignment
        assert_eq!(Splat2D::SIZE, 64);
        assert_eq!(TileKey::SIZE, 16);
        assert_eq!(TileRange::SIZE, 8);
    }

    #[test]
    fn test_tile_params() {
        let params = TileParams::new(1920, 1080);
        assert_eq!(params.num_tiles_x, 120); // 1920 / 16 = 120
        assert_eq!(params.num_tiles_y, 68);  // ceil(1080 / 16) = 68
    }
}
