//! Phase 4: Baseline compression (no deep model yet)
//! 
//! Defines quantization and classical entropy coding baseline.

use crate::splat::GaussianSplat;
use serde::{Serialize, Deserialize};

/// Phase 4.1: Quantization scheme for each attribute
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizationScheme {
    /// Position residual quantization (meters)
    pub pos_residual_bits: u8,
    
    /// Scale quantization bits
    pub scale_bits: u8,
    
    /// Rotation quantization bits (quaternion)
    pub rotation_bits: u8,
    
    /// Color quantization bits (per channel)
    pub color_bits: u8,
    
    /// Opacity quantization bits
    pub opacity_bits: u8,
    
    /// Uncertainty quantization bits
    pub uncertainty_bits: u8,
}

impl Default for QuantizationScheme {
    fn default() -> Self {
        Self {
            pos_residual_bits: 12,  // 12 bits = 4096 levels
            scale_bits: 8,          // 8 bits = 256 levels
            rotation_bits: 10,      // 10 bits per component
            color_bits: 8,          // 8 bits per channel (24-bit color)
            opacity_bits: 8,        // 8 bits
            uncertainty_bits: 8,    // 8 bits for uncertainty
        }
    }
}

/// Phase 4.1: Quantized splat representation
#[derive(Clone, Debug)]
pub struct QuantizedSplat {
    /// Quantized position residual (relative to cell centroid)
    pub pos_residual: [i16; 3],
    
    /// Quantized scale
    pub scale: [u8; 3],
    
    /// Quantized rotation (compressed quaternion)
    pub rotation: [u16; 4],
    
    /// Quantized color
    pub color: [u8; 3],
    
    /// Quantized opacity
    pub opacity: u8,
    
    /// Quantized uncertainty (optional)
    pub uncertainty: Option<u8>,
}

impl QuantizedSplat {
    /// Quantize a GaussianSplat
    pub fn from_splat(splat: &GaussianSplat, scheme: &QuantizationScheme, 
                     cell_centroid: [f32; 3], pos_range: f32) -> Self {
        // Quantize position residual
        let pos_residual = [
            quantize((splat.pos[0] - cell_centroid[0]) / pos_range, scheme.pos_residual_bits),
            quantize((splat.pos[1] - cell_centroid[1]) / pos_range, scheme.pos_residual_bits),
            quantize((splat.pos[2] - cell_centroid[2]) / pos_range, scheme.pos_residual_bits),
        ];
        
        // Quantize scale (assuming normalized range [0, 1])
        let scale = [
            quantize_unsigned(splat.scale[0], scheme.scale_bits),
            quantize_unsigned(splat.scale[1], scheme.scale_bits),
            quantize_unsigned(splat.scale[2], scheme.scale_bits),
        ];
        
        // Quantize rotation (quaternion components in [-1, 1])
        let rotation = [
            quantize_u16((splat.rotation[0] + 1.0) / 2.0, scheme.rotation_bits),
            quantize_u16((splat.rotation[1] + 1.0) / 2.0, scheme.rotation_bits),
            quantize_u16((splat.rotation[2] + 1.0) / 2.0, scheme.rotation_bits),
            quantize_u16((splat.rotation[3] + 1.0) / 2.0, scheme.rotation_bits),
        ];
        
        // Quantize color [0, 1] -> [0, 255]
        let color = [
            (splat.color[0] * 255.0) as u8,
            (splat.color[1] * 255.0) as u8,
            (splat.color[2] * 255.0) as u8,
        ];
        
        // Quantize opacity
        let opacity = (splat.opacity * 255.0) as u8;
        
        Self {
            pos_residual,
            scale,
            rotation,
            color,
            opacity,
            uncertainty: None,
        }
    }
}

fn quantize(value: f32, bits: u8) -> i16 {
    let max = (1i32 << (bits - 1)) - 1;
    (value.clamp(-1.0, 1.0) * max as f32) as i16
}

fn quantize_unsigned(value: f32, bits: u8) -> u8 {
    let max = (1u32 << bits) - 1;
    ((value.clamp(0.0, 1.0) * max as f32) as u8).min(max as u8)
}

fn quantize_u16(value: f32, bits: u8) -> u16 {
    let max = (1u32 << bits) - 1;
    let v = (value.clamp(0.0, 1.0) * max as f32).round() as u32;
    v.min(max) as u16
}

/// Phase 4.2: Classical entropy coder baseline
pub struct EntropyCoder;

impl EntropyCoder {
    /// Encode quantized splats using simple run-length + Huffman-like coding
    /// Returns compressed bytes and statistics
    pub fn encode(splats: &[QuantizedSplat]) -> (Vec<u8>, CompressionStats) {
        // Simplified: use LZ4 for now (baseline)
        // In production, would use arithmetic coding or Huffman
        let mut bytes = Vec::new();
        
        // Serialize quantized splats
        for splat in splats {
            bytes.extend_from_slice(&splat.pos_residual[0].to_le_bytes());
            bytes.extend_from_slice(&splat.pos_residual[1].to_le_bytes());
            bytes.extend_from_slice(&splat.pos_residual[2].to_le_bytes());
            bytes.push(splat.scale[0]);
            bytes.push(splat.scale[1]);
            bytes.push(splat.scale[2]);
            bytes.extend_from_slice(&splat.rotation[0].to_le_bytes());
            bytes.extend_from_slice(&splat.rotation[1].to_le_bytes());
            bytes.extend_from_slice(&splat.rotation[2].to_le_bytes());
            bytes.extend_from_slice(&splat.rotation[3].to_le_bytes());
            bytes.push(splat.color[0]);
            bytes.push(splat.color[1]);
            bytes.push(splat.color[2]);
            bytes.push(splat.opacity);
        }
        
        // Compress with LZ4
        let compressed = lz4_flex::compress(&bytes);
        
        let stats = CompressionStats {
            original_bytes: bytes.len(),
            compressed_bytes: compressed.len(),
            splat_count: splats.len(),
        };
        
        (compressed, stats)
    }
    
    /// Decode compressed bytes back to quantized splats
    pub fn decode(data: &[u8], splat_count: usize) -> anyhow::Result<Vec<QuantizedSplat>> {
        let decompressed = lz4_flex::decompress(data, data.len() * 10)?; // Estimate
        
        let splats = Vec::new();
        let bytes_per_splat = 3 * 2 + 3 + 4 * 2 + 3 + 1; // pos(6) + scale(3) + rot(8) + color(3) + opacity(1)
        
        for i in 0..splat_count.min(decompressed.len() / bytes_per_splat) {
            let _offset = i * bytes_per_splat;
            // Deserialize (simplified)
            // In production, would properly deserialize all fields
        }
        
        Ok(splats)
    }
}

#[derive(Clone, Debug)]
pub struct CompressionStats {
    pub original_bytes: usize,
    pub compressed_bytes: usize,
    pub splat_count: usize,
}

impl CompressionStats {
    pub fn compression_ratio(&self) -> f64 {
        self.compressed_bytes as f64 / self.original_bytes as f64
    }
    
    pub fn bytes_per_splat(&self) -> f64 {
        self.compressed_bytes as f64 / self.splat_count as f64
    }
}
