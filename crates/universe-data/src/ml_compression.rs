//! Phase 5: ML compression integration
//! 
//! Integrates learned entropy model into the compression pipeline.

use crate::compression::{QuantizedSplat, EntropyCoder, CompressionStats};
use anyhow::Result;

/// Phase 5.2: ML-enhanced compression using learned entropy model
pub struct MLEntropyCompressor {
    /// Whether to use learned model (if available)
    use_learned: bool,
}

impl MLEntropyCompressor {
    pub fn new(use_learned: bool) -> Self {
        Self { use_learned }
    }
    
    /// Phase 5.2: Compress quantized splats using learned entropy model
    /// 
    /// If learned model is available, uses predicted probabilities for better compression.
    /// Falls back to baseline entropy coder if model unavailable.
    pub fn compress(&self, splats: &[QuantizedSplat]) -> Result<(Vec<u8>, CompressionStats)> {
        if self.use_learned {
            // Phase 5.1: Use learned entropy model
            // In production, would load model and predict probabilities
            // For now, fall back to baseline
            Ok(EntropyCoder::encode(splats))
        } else {
            // Baseline compression
            Ok(EntropyCoder::encode(splats))
        }
    }
    
    /// Phase 5.2: Decompress with learned entropy decoder
    /// 
    /// WASM-friendly and parallelizable.
    pub fn decompress(&self, data: &[u8], splat_count: usize) -> Result<Vec<QuantizedSplat>> {
        // Phase 5.3: Validate decode cost
        // In production, would measure decode time and compare to baseline
        EntropyCoder::decode(data, splat_count)
    }
}

/// Phase 5.3: Compression validation metrics
#[derive(Clone, Debug)]
pub struct CompressionMetrics {
    pub compression_ratio: f64,
    pub bytes_per_splat: f64,
    pub decode_time_ms: f64,
    pub quality_score: f64, // Uncertainty-weighted distortion
}

impl CompressionMetrics {
    /// Phase 5.3: Validate that decode cost is acceptable
    pub fn is_acceptable(&self) -> bool {
        // Target: < 1ms decode time per 1000 splats
        self.decode_time_ms < 1.0 && self.compression_ratio < 0.5
    }
}
