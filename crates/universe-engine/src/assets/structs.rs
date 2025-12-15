//! GPU-facing structs for orbital assets.
//!
//! The field order here is chosen to match the engine/shader conventions.

use bytemuck::{Pod, Zeroable};

// 1. The Carrier (Kepler Orbit)
// Matches 'orbit.json' data, but aligned for GPU Uniforms (16-byte align)
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct KeplerParams {
    pub semi_major_axis: f32,
    pub eccentricity: f32,
    pub inclination: f32,
    pub arg_periapsis: f32,
    pub long_asc_node: f32,
    pub mean_anomaly_0: f32,
    // Padding/Extra info for Shader
    pub residual_scale: f32,
    pub count: u32,
    pub _pad1: f32,
    pub _pad2: f32,
    pub _pad3: f32,
    pub _pad4: f32,
}

// 2. The Surprise (Compressed Residuals)
// Matches 'residuals.bin' (u32 per frame)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PackedResidual {
    pub data: u32, // High 16: Radial, Low 16: Transverse
}

// 3. The Brain (Neural Network Weights)
// Matches 'neural_decoder.bin' (Sequence of f32)
// No struct needed, just Vec<f32>
