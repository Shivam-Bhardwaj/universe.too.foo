use bytemuck::{Pod, Zeroable};

/// Keplerian + residual decode parameters.
///
/// Host layout must be compatible with WGSL uniform layout.
/// We enforce 16-byte alignment for GPU-friendly uploads.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct KeplerParams {
    pub semi_major_axis: f32,
    pub eccentricity: f32,
    pub inclination: f32,
    pub arg_periapsis: f32,

    pub long_asc_node: f32,
    pub mean_anomaly_0: f32,
    pub residual_scale: f32,
    pub count: u32,
}

/// Packed residual sample.
///
/// Binary format: one `u32` per timestep.
/// Interpretation (matches Python `struct.pack('hh', radial, transverse)` on little-endian):
/// Low 16 = Radial, High 16 = Transverse.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PackedResidual {
    pub data: u32,
}
