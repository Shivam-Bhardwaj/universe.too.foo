//! GPU-compatible data types

use bytemuck::{Pod, Zeroable};

/// Gaussian Splat vertex data for GPU
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuSplat {
    /// Position relative to camera (floating origin)
    pub pos: [f32; 3],
    /// Padding for alignment
    pub _pad0: f32,
    /// Scale (axis lengths)
    pub scale: [f32; 3],
    /// Padding
    pub _pad1: f32,
    /// Rotation quaternion
    pub rotation: [f32; 4],
    /// RGB color
    pub color: [f32; 3],
    /// Opacity
    pub opacity: f32,
}

impl GpuSplat {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    pub fn from_splat(splat: &universe_data::GaussianSplat, camera_offset: glam::Vec3) -> Self {
        Self {
            pos: [
                splat.pos[0] - camera_offset.x,
                splat.pos[1] - camera_offset.y,
                splat.pos[2] - camera_offset.z,
            ],
            _pad0: 0.0,
            scale: splat.scale,
            _pad1: 0.0,
            rotation: splat.rotation,
            color: splat.color,
            opacity: splat.opacity,
        }
    }
}

/// Instance data for instanced rendering
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SplatInstance {
    pub pos: [f32; 3],
    pub scale: f32,        // Uniform scale for billboard
    pub color: [f32; 3],
    pub opacity: f32,
}

/// Indirect draw arguments
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DrawIndirect {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}
