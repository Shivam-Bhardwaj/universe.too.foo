//! GPU-facing, memory-aligned structs for orbital assets.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// Keplerian + residual decode parameters.
///
/// This is intended for a uniform buffer, so we enforce 16-byte alignment.
///
/// Fields (matching `*_orbit.json`):
/// - `a`, `e`, `i`, `w`, `O`, `M0`: Kepler elements
/// - `scale`: quantization scale for residual decode
/// - `count`: number of packed residual samples
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Serialize, Deserialize, Pod, Zeroable)]
pub struct KeplerParams {
    pub a: f32,
    pub e: f32,
    pub i: f32,
    pub w: f32,

    // NOTE: These names intentionally match the JSON/WGSL conventions.
    pub O: f32,
    pub M0: f32,

    #[serde(alias = "residual_scale")]
    pub scale: f32,

    pub count: u32,
}

/// Packed residual sample: (radial:int16 | transverse:int16) stored in a `u32`.
///
/// This is intended for a storage buffer.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PackedResidual {
    pub data: u32,
}

/// Neural network weights exported as a flat little-endian `f32` buffer.
///
/// In WGSL this is typically bound as `var<storage, read> weights: array<f32>;`.
pub type NeuralWeights = Vec<f32>;
