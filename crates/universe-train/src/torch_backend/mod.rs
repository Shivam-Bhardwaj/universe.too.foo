//! LibTorch (tch-rs) training backend for CUDA acceleration
//!
//! This module provides an alternative training backend using raw LibTorch
//! for CUDA-accelerated Gaussian splatting optimization.

mod loss;
mod model;
mod rasterizer;
mod trainer;

pub use loss::{combined_loss, dssim_loss, l1_loss};
pub use model::TorchGaussianCloud;
pub use rasterizer::render_gaussians;
pub use trainer::{train_universe, TorchDevice, TorchTrainer};
