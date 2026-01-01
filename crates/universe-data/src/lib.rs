pub mod splat;
pub mod cell;
pub mod manifest;

// The following modules are *pipeline / training* utilities and are not required
// for the runtime WASM viewer build. Keeping them behind cfg avoids pulling in
// native-only deps (reqwest/rustls/ring, tokio full, etc.) when targeting wasm32.
#[cfg(not(target_arch = "wasm32"))]
pub mod canonical;
#[cfg(not(target_arch = "wasm32"))]
pub mod compression;
#[cfg(not(target_arch = "wasm32"))]
pub mod crossmatch;
#[cfg(not(target_arch = "wasm32"))]
pub mod ephemeris;
#[cfg(not(target_arch = "wasm32"))]
pub mod ml_compression;
#[cfg(not(target_arch = "wasm32"))]
pub mod ml_shards;
#[cfg(not(target_arch = "wasm32"))]
pub mod stars;
#[cfg(not(target_arch = "wasm32"))]
pub mod landmarks;
#[cfg(not(target_arch = "wasm32"))]
pub mod pipeline;

pub use splat::{GaussianSplat, CompressedSplat};
pub use cell::{CellData, CellMetadata};
pub use manifest::{CellManifest, CellEntry};

#[cfg(not(target_arch = "wasm32"))]
pub use ephemeris::{EphemerisProvider, SolarSystemBody, BodyState, Velocity, download_de440};
#[cfg(not(target_arch = "wasm32"))]
pub use stars::{StarCatalog, StarRecord, generate_synthetic_stars};
#[cfg(not(target_arch = "wasm32"))]
pub use landmarks::{Landmark, LandmarkKind, Position as LandmarkPosition, load_landmarks_json};
#[cfg(not(target_arch = "wasm32"))]
pub use pipeline::{DataPipeline, PipelineConfig, merge_manifests};
