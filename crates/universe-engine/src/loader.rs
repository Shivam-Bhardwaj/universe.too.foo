//! Asset loading (CPU-side IO) for orbital engine binary formats.

use crate::structs::{KeplerParams, NeuralWeights, PackedResidual};

use anyhow::{bail, ensure, Context, Result};
use std::fs::{self, File};
use std::io::Read;

/// Load a single orbit asset bundle.
///
/// `path_prefix` is the shared prefix for both files:
/// - `{prefix}_orbit.json`
/// - `{prefix}_residuals.bin`
pub fn load_orbit(path_prefix: &str) -> Result<(KeplerParams, Vec<PackedResidual>)> {
    let orbit_path = format!("{path_prefix}_orbit.json");
    let residuals_path = format!("{path_prefix}_residuals.bin");

    // --- orbit.json -> KeplerParams ---
    let orbit_bytes = fs::read(&orbit_path)
        .with_context(|| format!("failed to read orbit metadata: {orbit_path}"))?;

    let mut params: KeplerParams = serde_json::from_slice(&orbit_bytes)
        .with_context(|| format!("failed to parse orbit metadata JSON: {orbit_path}"))?;

    // --- residuals.bin -> Vec<PackedResidual> ---
    let mut f = File::open(&residuals_path)
        .with_context(|| format!("failed to open residuals bitstream: {residuals_path}"))?;

    let len = f
        .metadata()
        .with_context(|| format!("failed to stat residuals bitstream: {residuals_path}"))?
        .len() as usize;

    let elem_size = std::mem::size_of::<PackedResidual>();
    ensure!(
        len % elem_size == 0,
        "residuals.bin length {len} is not a multiple of element size {elem_size}"
    );

    let n = len / elem_size;
    let mut residuals = vec![PackedResidual { data: 0 }; n];

    f.read_exact(bytemuck::cast_slice_mut(&mut residuals))
        .with_context(|| format!("failed to read residuals bitstream: {residuals_path}"))?;

    // Endianness safety: assets are authored little-endian.
    for r in &mut residuals {
        r.data = u32::from_le(r.data);
    }

    // Validate or fill count.
    if params.count == 0 {
        params.count = residuals.len() as u32;
    } else if params.count as usize != residuals.len() {
        bail!(
            "residual count mismatch: orbit.json count={} but residuals.bin contains {} samples",
            params.count,
            residuals.len()
        );
    }

    Ok((params, residuals))
}

/// Load a flat float32 neural weight buffer (little-endian).
pub fn load_brain(path: &str) -> Result<NeuralWeights> {
    let mut f = File::open(path).with_context(|| format!("failed to open neural weights: {path}"))?;

    let len = f
        .metadata()
        .with_context(|| format!("failed to stat neural weights: {path}"))?
        .len() as usize;

    ensure!(
        len % 4 == 0,
        "neural weights file length {len} is not a multiple of 4 bytes"
    );

    let n = len / 4;
    let mut weights = vec![0.0f32; n];

    f.read_exact(bytemuck::cast_slice_mut(&mut weights))
        .with_context(|| format!("failed to read neural weights: {path}"))?;

    // Endianness safety: assets are authored little-endian.
    for w in &mut weights {
        *w = f32::from_bits(u32::from_le(w.to_bits()));
    }

    Ok(weights)
}
