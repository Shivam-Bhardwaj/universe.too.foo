//! Phase 2.1: Shared engine core for native and WASM builds
//! 
//! This crate contains the core rendering engine logic that can be compiled
//! to both native (for reference viewer) and WASM (for web viewer).
//! 
//! Platform-specific adapters (file I/O, WebGPU initialization) are abstracted
//! via traits to allow the same code to run in both environments.

#![cfg_attr(target_arch = "wasm32", allow(clippy::unused_unit))]

pub mod camera;
pub mod streaming;
pub mod renderer;
pub mod http_adapter;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-export core types
pub use camera::*;
pub use streaming::*;
pub use renderer::*;
