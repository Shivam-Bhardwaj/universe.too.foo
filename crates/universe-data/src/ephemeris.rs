//! Solar system ephemeris provider using ANISE
//!
//! ANISE is a pure Rust replacement for NASA SPICE, providing high-fidelity
//! planetary positions without C dependencies.

use anise::prelude::*;
use hifitime::Epoch;
use universe_core::coordinates::CartesianPosition;
use std::path::Path;
use thiserror::Error;
use anyhow::Result;

#[derive(Error, Debug)]
pub enum EphemerisError {
    #[error("Failed to load SPK file: {0}")]
    SpkLoadError(String),
    #[error("Body not found: {0}")]
    BodyNotFound(String),
    #[error("Epoch out of range: {0}")]
    EpochOutOfRange(String),
}

/// Solar system body identifiers (NAIF IDs)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SolarSystemBody {
    Sun,
    Mercury,
    Venus,
    Earth,
    Moon,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
    Pluto,
}

impl SolarSystemBody {
    /// NAIF ID for the body barycenter
    pub fn naif_id(&self) -> i32 {
        match self {
            Self::Sun => 10,
            Self::Mercury => 1,      // Mercury barycenter
            Self::Venus => 2,        // Venus barycenter
            Self::Earth => 3,        // Earth-Moon barycenter
            Self::Moon => 301,       // Moon
            Self::Mars => 4,         // Mars barycenter
            Self::Jupiter => 5,      // Jupiter barycenter
            Self::Saturn => 6,       // Saturn barycenter
            Self::Uranus => 7,       // Uranus barycenter
            Self::Neptune => 8,      // Neptune barycenter
            Self::Pluto => 9,        // Pluto barycenter
        }
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Sun => "Sun",
            Self::Mercury => "Mercury",
            Self::Venus => "Venus",
            Self::Earth => "Earth",
            Self::Moon => "Moon",
            Self::Mars => "Mars",
            Self::Jupiter => "Jupiter",
            Self::Saturn => "Saturn",
            Self::Uranus => "Uranus",
            Self::Neptune => "Neptune",
            Self::Pluto => "Pluto",
        }
    }

    /// Approximate mean radius in meters
    pub fn radius_m(&self) -> f64 {
        match self {
            Self::Sun => 6.9634e8,
            Self::Mercury => 2.4397e6,
            Self::Venus => 6.0518e6,
            Self::Earth => 6.371e6,
            Self::Moon => 1.7374e6,
            Self::Mars => 3.3895e6,
            Self::Jupiter => 6.9911e7,
            Self::Saturn => 5.8232e7,
            Self::Uranus => 2.5362e7,
            Self::Neptune => 2.4622e7,
            Self::Pluto => 1.188e6,
        }
    }

    /// Approximate visual color (RGB normalized)
    pub fn color(&self) -> [f32; 3] {
        match self {
            Self::Sun => [1.0, 0.95, 0.8],
            Self::Mercury => [0.6, 0.6, 0.6],
            Self::Venus => [0.9, 0.85, 0.7],
            Self::Earth => [0.2, 0.4, 0.8],
            Self::Moon => [0.7, 0.7, 0.7],
            Self::Mars => [0.8, 0.4, 0.2],
            Self::Jupiter => [0.9, 0.8, 0.6],
            Self::Saturn => [0.9, 0.85, 0.6],
            Self::Uranus => [0.6, 0.8, 0.9],
            Self::Neptune => [0.3, 0.5, 0.9],
            Self::Pluto => [0.8, 0.75, 0.7],
        }
    }

    /// All bodies including Sun
    pub fn all() -> &'static [SolarSystemBody] {
        &[
            Self::Sun, Self::Mercury, Self::Venus, Self::Earth, Self::Moon,
            Self::Mars, Self::Jupiter, Self::Saturn, Self::Uranus,
            Self::Neptune, Self::Pluto,
        ]
    }

    /// Planets only (no Sun, no Moon)
    pub fn planets() -> &'static [SolarSystemBody] {
        &[
            Self::Mercury, Self::Venus, Self::Earth, Self::Mars,
            Self::Jupiter, Self::Saturn, Self::Uranus, Self::Neptune, Self::Pluto,
        ]
    }
}

/// Velocity in meters per second
#[derive(Clone, Copy, Debug)]
pub struct Velocity {
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
}

/// Full state vector (position + velocity)
#[derive(Clone, Copy, Debug)]
pub struct BodyState {
    pub position: CartesianPosition,
    pub velocity: Velocity,
}

/// Ephemeris provider using ANISE
pub struct EphemerisProvider {
    almanac: Almanac,
}

impl EphemerisProvider {
    /// Load ephemeris from SPK file(s)
    ///
    /// Recommended: Use NASA DE440 for high accuracy
    /// Download from: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
    pub fn load(spk_paths: &[&Path]) -> Result<Self> {
        let mut almanac = Almanac::default();

        for path in spk_paths {
            tracing::info!("Loading SPK: {:?}", path);
            let path_str = path.to_str()
                .ok_or_else(|| EphemerisError::SpkLoadError(format!("Invalid path: {:?}", path)))?;
            let spk = SPK::load(path_str)
                .map_err(|e| EphemerisError::SpkLoadError(format!("{:?}: {}", path, e)))?;
            almanac = almanac.with_spk(spk);
        }

        Ok(Self { almanac })
    }

    /// Load from default data directory
    pub fn load_default(data_dir: &Path) -> Result<Self> {
        let de440_path = data_dir.join("de440.bsp");

        if !de440_path.exists() {
            anyhow::bail!(
                "DE440 ephemeris not found at {:?}. Run 'helios fetch-ephemeris' first.",
                de440_path
            );
        }

        Self::load(&[&de440_path])
    }

    /// Get body position at epoch, Sun-centered, J2000 ecliptic frame
    /// Returns position in meters
    pub fn body_position(&self, body: SolarSystemBody, epoch: Epoch) -> Result<CartesianPosition> {
        // Sun is at origin in our heliocentric frame
        if body == SolarSystemBody::Sun {
            return Ok(CartesianPosition::new(0.0, 0.0, 0.0));
        }

        // Create frame references
        let body_frame = Frame::from_ephem_j2000(body.naif_id());
        let sun_frame = Frame::from_ephem_j2000(SolarSystemBody::Sun.naif_id());

        // Query position relative to Sun
        let state = self.almanac.translate(
            body_frame,
            sun_frame,
            epoch,
            None, // No aberration correction for visualization
        )?;

        // ANISE returns kilometers, convert to meters
        Ok(CartesianPosition::new(
            state.radius_km.x * 1000.0,
            state.radius_km.y * 1000.0,
            state.radius_km.z * 1000.0,
        ))
    }

    /// Get full state (position + velocity) at epoch
    pub fn body_state(&self, body: SolarSystemBody, epoch: Epoch) -> Result<BodyState> {
        if body == SolarSystemBody::Sun {
            return Ok(BodyState {
                position: CartesianPosition::new(0.0, 0.0, 0.0),
                velocity: Velocity { vx: 0.0, vy: 0.0, vz: 0.0 },
            });
        }

        let body_frame = Frame::from_ephem_j2000(body.naif_id());
        let sun_frame = Frame::from_ephem_j2000(SolarSystemBody::Sun.naif_id());

        let state = self.almanac.translate(body_frame, sun_frame, epoch, None)?;

        Ok(BodyState {
            position: CartesianPosition::new(
                state.radius_km.x * 1000.0,
                state.radius_km.y * 1000.0,
                state.radius_km.z * 1000.0,
            ),
            velocity: Velocity {
                vx: state.velocity_km_s.x * 1000.0,
                vy: state.velocity_km_s.y * 1000.0,
                vz: state.velocity_km_s.z * 1000.0,
            },
        })
    }

    /// Get positions of all major bodies at epoch
    pub fn system_snapshot(&self, epoch: Epoch) -> Result<Vec<(SolarSystemBody, CartesianPosition)>> {
        let mut positions = Vec::new();

        for body in SolarSystemBody::all() {
            match self.body_position(*body, epoch) {
                Ok(pos) => positions.push((*body, pos)),
                Err(e) => tracing::warn!("Could not get position for {}: {}", body.name(), e),
            }
        }

        Ok(positions)
    }
}

/// Download DE440 ephemeris file from NASA NAIF
pub async fn download_de440(output_dir: &Path) -> Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};
    use tokio::io::AsyncWriteExt;
    use futures_util::StreamExt;

    let url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp";
    let output_path = output_dir.join("de440.bsp");

    if output_path.exists() {
        tracing::info!("DE440 already exists at {:?}", output_path);
        println!("DE440 ephemeris already downloaded.");
        return Ok(());
    }

    std::fs::create_dir_all(output_dir)?;

    println!("Downloading DE440 ephemeris from NASA NAIF...");
    println!("URL: {}", url);
    println!("This is ~120MB and may take a few minutes.");

    let client = reqwest::Client::new();
    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        anyhow::bail!("Download failed with status: {}", response.status());
    }

    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
        .expect("Invalid progress template")
        .progress_chars("=>-"));

    let mut file = tokio::fs::File::create(&output_path).await?;
    let mut stream = response.bytes_stream();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        file.write_all(&chunk).await?;
        pb.inc(chunk.len() as u64);
    }

    file.flush().await?;
    pb.finish_with_message("Download complete");

    println!("Saved to {:?}", output_path);

    Ok(())
}
