use std::fs::File;
use std::io::{BufReader, Read};

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;

use crate::assets::structs::{KeplerParams, PackedResidual};

#[derive(Deserialize)]
struct OrbitMeta {
    a: f32,
    e: f32,
    i: f32,
    w: f32,
    #[serde(rename = "O")]
    o: f32,
    #[serde(rename = "M0")]
    m0: f32,
    residual_scale: f32,
    count: u32,
}

pub fn load_orbit_data(base_path: &str) -> Result<(KeplerParams, Vec<PackedResidual>)> {
    // 1) orbit.json
    let json_path = format!("{base_path}_orbit.json");
    let file = File::open(&json_path).context("Failed to open orbit.json")?;
    let meta: OrbitMeta = serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("Failed to parse orbit.json: {json_path}"))?;

    let params = KeplerParams {
        semi_major_axis: meta.a,
        eccentricity: meta.e,
        inclination: meta.i,
        arg_periapsis: meta.w,
        long_asc_node: meta.o,
        mean_anomaly_0: meta.m0,
        residual_scale: meta.residual_scale,
        count: meta.count,
    };

    // 2) residuals.bin
    let bin_path = format!("{base_path}_residuals.bin");
    let mut file = File::open(&bin_path).context("Failed to open residuals.bin")?;

    let len = file
        .metadata()
        .with_context(|| format!("Failed to stat residuals.bin: {bin_path}"))?
        .len() as usize;

    let elem_size = std::mem::size_of::<PackedResidual>();
    ensure!(
        len % elem_size == 0,
        "residuals.bin length {len} is not a multiple of {elem_size}"
    );

    let n = len / elem_size;
    let mut residuals = vec![PackedResidual { data: 0 }; n];

    file.read_exact(bytemuck::cast_slice_mut(&mut residuals))
        .with_context(|| format!("Failed to read residuals.bin: {bin_path}"))?;

    // Endianness safety (assets are written little-endian).
    if cfg!(target_endian = "big") {
        for r in &mut residuals {
            r.data = u32::from_le(r.data);
        }
    }

    if params.count != residuals.len() as u32 {
        bail!(
            "Orbit meta count mismatch: orbit.json count={} but residuals.bin contains {} samples",
            params.count,
            residuals.len()
        );
    }

    Ok((params, residuals))
}

pub fn load_neural_brain(path: &str) -> Result<Vec<f32>> {
    let mut file = File::open(path).context("Failed to open neural_decoder.bin")?;

    let len = file
        .metadata()
        .with_context(|| format!("Failed to stat neural_decoder.bin: {path}"))?
        .len() as usize;

    ensure!(
        len % 4 == 0,
        "neural_decoder.bin length {len} is not a multiple of 4 bytes"
    );

    let n = len / 4;
    let mut weights = vec![0.0f32; n];

    file.read_exact(bytemuck::cast_slice_mut(&mut weights))
        .with_context(|| format!("Failed to read neural_decoder.bin: {path}"))?;

    // Endianness safety (assets are written little-endian).
    if cfg!(target_endian = "big") {
        for w in &mut weights {
            *w = f32::from_bits(u32::from_le(w.to_bits()));
        }
    }

    Ok(weights)
}
