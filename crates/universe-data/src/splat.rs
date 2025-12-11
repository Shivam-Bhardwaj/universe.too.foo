use bytemuck::{Pod, Zeroable};

/// A single Gaussian Splat primitive
///
/// Memory layout: 56 bytes, GPU-friendly alignment
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GaussianSplat {
    /// Position relative to cell centroid (meters)
    pub pos: [f32; 3],
    /// Scale along each axis (meters)
    pub scale: [f32; 3],
    /// Rotation quaternion (x, y, z, w)
    pub rotation: [f32; 4],
    /// RGB color [0, 1]
    pub color: [f32; 3],
    /// Opacity [0, 1]
    pub opacity: f32,
}

impl GaussianSplat {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    pub fn new(
        pos: [f32; 3],
        scale: [f32; 3],
        rotation: [f32; 4],
        color: [f32; 3],
        opacity: f32,
    ) -> Self {
        Self { pos, scale, rotation, color, opacity }
    }

    /// Create a simple spherical splat (uniform scale, no rotation)
    pub fn sphere(pos: [f32; 3], radius: f32, color: [f32; 3], opacity: f32) -> Self {
        Self {
            pos,
            scale: [radius, radius, radius],
            rotation: [0.0, 0.0, 0.0, 1.0], // Identity quaternion
            color,
            opacity,
        }
    }
}

/// Compressed splat for storage (24 bytes)
/// Positions as u16 normalized to cell bounds
/// Rotations as i8 quaternion
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct CompressedSplat {
    pub pos: [u16; 3],      // 6 bytes
    pub scale: [u16; 3],    // 6 bytes
    pub rotation: [i8; 4],  // 4 bytes
    pub color: [u8; 3],     // 3 bytes
    pub opacity: u8,        // 1 byte
    pub _padding: [u8; 4],  // 4 bytes for alignment
}

impl CompressedSplat {
    /// Decompress to full precision splat
    pub fn decompress(&self, cell_min: [f32; 3], cell_max: [f32; 3]) -> GaussianSplat {
        let pos = [
            lerp(cell_min[0], cell_max[0], self.pos[0] as f32 / 65535.0),
            lerp(cell_min[1], cell_max[1], self.pos[1] as f32 / 65535.0),
            lerp(cell_min[2], cell_max[2], self.pos[2] as f32 / 65535.0),
        ];

        // Scale: log-encoded, 0-65535 maps to 1e-3 to 1e6 meters
        let scale = [
            decode_log_scale(self.scale[0]),
            decode_log_scale(self.scale[1]),
            decode_log_scale(self.scale[2]),
        ];

        // Rotation: i8 to normalized quaternion
        let rotation = normalize_quat([
            self.rotation[0] as f32 / 127.0,
            self.rotation[1] as f32 / 127.0,
            self.rotation[2] as f32 / 127.0,
            self.rotation[3] as f32 / 127.0,
        ]);

        let color = [
            self.color[0] as f32 / 255.0,
            self.color[1] as f32 / 255.0,
            self.color[2] as f32 / 255.0,
        ];

        let opacity = self.opacity as f32 / 255.0;

        GaussianSplat { pos, scale, rotation, color, opacity }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn decode_log_scale(v: u16) -> f32 {
    let t = v as f32 / 65535.0;
    10.0_f32.powf(-3.0 + t * 9.0)  // 1e-3 to 1e6
}

fn normalize_quat(q: [f32; 4]) -> [f32; 4] {
    let len = (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]).sqrt();
    if len > 0.0 {
        [q[0]/len, q[1]/len, q[2]/len, q[3]/len]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}
