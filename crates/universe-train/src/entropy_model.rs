//! Phase 5.1: Learned entropy model for compression
//!
//! This module defines a *baseline* context model that can be trained to
//! predict symbol logits for quantized splat attributes. The current
//! implementation is intentionally simple and compile-safe; it can be
//! upgraded to a richer architecture (transformer, hyperprior, etc.) later.

use burn::module::Module;
use burn::nn;
use burn::prelude::*;

/// Phase 5.1: Context model for predicting symbol logits.
///
/// Input: per-splat feature vector (e.g. 16 floats, matching GPU splat layout)
/// Output: per-attribute logits over discrete symbol alphabets.
#[derive(Module, Debug)]
pub struct EntropyContextModel<B: Backend> {
    // Shared trunk
    encoder: nn::Linear<B>,
    trunk1: nn::Linear<B>,
    trunk2: nn::Linear<B>,

    // Attribute heads
    pos_head: nn::Linear<B>,      // e.g. 12-bit residual symbols (4096)
    scale_head: nn::Linear<B>,    // 8-bit symbols (256)
    rotation_head: nn::Linear<B>, // 10-bit symbols (1024)
    color_head: nn::Linear<B>,    // 8-bit symbols (256)
    opacity_head: nn::Linear<B>,  // 8-bit symbols (256)
}

impl<B: Backend> EntropyContextModel<B> {
    /// Create a new model.
    ///
    /// - `in_features`: feature width for each splat (default: 16)
    /// - `hidden`: trunk width (default: 128)
    pub fn new(device: &B::Device, in_features: usize, hidden: usize) -> Self {
        let encoder = nn::LinearConfig::new(in_features, hidden).init(device);
        let trunk1 = nn::LinearConfig::new(hidden, hidden * 2).init(device);
        let trunk2 = nn::LinearConfig::new(hidden * 2, hidden).init(device);

        let pos_head = nn::LinearConfig::new(hidden, 4096).init(device);
        let scale_head = nn::LinearConfig::new(hidden, 256).init(device);
        let rotation_head = nn::LinearConfig::new(hidden, 1024).init(device);
        let color_head = nn::LinearConfig::new(hidden, 256).init(device);
        let opacity_head = nn::LinearConfig::new(hidden, 256).init(device);

        Self {
            encoder,
            trunk1,
            trunk2,
            pos_head,
            scale_head,
            rotation_head,
            color_head,
            opacity_head,
        }
    }

    /// Forward pass.
    ///
    /// `x`: shape [N, in_features]
    pub fn forward(&self, x: Tensor<B, 2>) -> EntropyLogits<B> {
        let h0 = self.encoder.forward(x);
        let h1 = self.trunk1.forward(h0);
        let h2 = self.trunk2.forward(h1);

        EntropyLogits {
            pos: self.pos_head.forward(h2.clone()),
            scale: self.scale_head.forward(h2.clone()),
            rotation: self.rotation_head.forward(h2.clone()),
            color: self.color_head.forward(h2.clone()),
            opacity: self.opacity_head.forward(h2),
        }
    }
}

/// Phase 5.1: Predicted logits for each attribute.
///
/// Each tensor is shape [N, symbols].
pub struct EntropyLogits<B: Backend> {
    pub pos: Tensor<B, 2>,
    pub scale: Tensor<B, 2>,
    pub rotation: Tensor<B, 2>,
    pub color: Tensor<B, 2>,
    pub opacity: Tensor<B, 2>,
}

/// Phase 5.2: WASM-friendly entropy decoder (placeholder).
///
/// Real implementation will use arithmetic/range decoding with model-predicted
/// CDFs and careful bit IO, designed to be parallelizable.
pub struct EntropyDecoder;

impl EntropyDecoder {
    pub fn decode_parallel(_probabilities: &[f32], _encoded_data: &[u8]) -> anyhow::Result<Vec<u16>> {
        Ok(Vec::new())
    }
}




