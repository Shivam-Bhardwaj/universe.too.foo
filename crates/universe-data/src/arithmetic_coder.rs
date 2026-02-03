//! Range/Arithmetic Coder for Gaussian Splat Compression
//!
//! Implements entropy coding for quantized splat attributes.
//! Uses a simple range coder that can work with either:
//! - Uniform probability distributions (baseline)
//! - Model-predicted CDFs (learned compression)

use crate::compression::{QuantizedSplat, CompressionStats};
use anyhow::Result;

/// Precision bits for range coder
const RANGE_BITS: u32 = 32;
const TOP_VALUE: u64 = 1u64 << RANGE_BITS;
const BOTTOM_VALUE: u64 = TOP_VALUE >> 8;
const MAX_RANGE: u64 = BOTTOM_VALUE;

/// Cumulative Distribution Function table for a single attribute
#[derive(Clone, Debug)]
pub struct CDFTable {
    /// Cumulative counts [0, c1, c2, ..., total]
    /// Length = num_symbols + 1
    pub cumulative: Vec<u32>,
    /// Total probability mass
    pub total: u32,
}

impl CDFTable {
    /// Create uniform CDF for given number of symbols
    pub fn uniform(num_symbols: usize) -> Self {
        let mut cumulative = Vec::with_capacity(num_symbols + 1);
        for i in 0..=num_symbols {
            cumulative.push(i as u32);
        }
        Self {
            total: num_symbols as u32,
            cumulative,
        }
    }

    /// Create CDF from probability counts
    pub fn from_counts(counts: &[u32]) -> Self {
        let mut cumulative = Vec::with_capacity(counts.len() + 1);
        cumulative.push(0);
        let mut sum = 0u32;
        for &c in counts {
            sum += c.max(1); // Ensure minimum count of 1 for valid encoding
            cumulative.push(sum);
        }
        Self {
            total: sum,
            cumulative,
        }
    }

    /// Get symbol range for encoding
    pub fn symbol_range(&self, symbol: usize) -> (u32, u32) {
        (self.cumulative[symbol], self.cumulative[symbol + 1])
    }

    /// Find symbol from cumulative value (for decoding)
    pub fn find_symbol(&self, value: u32) -> usize {
        // Binary search for symbol
        let mut lo = 0;
        let mut hi = self.cumulative.len() - 1;
        while lo < hi - 1 {
            let mid = (lo + hi) / 2;
            if self.cumulative[mid] <= value {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// Collection of CDFs for all splat attributes
#[derive(Clone)]
pub struct PredictedCDFs {
    /// CDFs for position residuals (3 components, 12-bit = 4096 symbols each)
    pub pos: [CDFTable; 3],
    /// CDFs for scales (3 components, 8-bit = 256 symbols each)
    pub scale: [CDFTable; 3],
    /// CDFs for rotations (4 components, 10-bit = 1024 symbols each)
    pub rotation: [CDFTable; 4],
    /// CDFs for colors (3 components, 8-bit = 256 symbols each)
    pub color: [CDFTable; 3],
    /// CDF for opacity (8-bit = 256 symbols)
    pub opacity: CDFTable,
}

impl PredictedCDFs {
    /// Create uniform CDFs for baseline compression
    pub fn uniform() -> Self {
        Self {
            pos: [
                CDFTable::uniform(4096),
                CDFTable::uniform(4096),
                CDFTable::uniform(4096),
            ],
            scale: [
                CDFTable::uniform(256),
                CDFTable::uniform(256),
                CDFTable::uniform(256),
            ],
            rotation: [
                CDFTable::uniform(1024),
                CDFTable::uniform(1024),
                CDFTable::uniform(1024),
                CDFTable::uniform(1024),
            ],
            color: [
                CDFTable::uniform(256),
                CDFTable::uniform(256),
                CDFTable::uniform(256),
            ],
            opacity: CDFTable::uniform(256),
        }
    }
}

/// Range encoder
pub struct RangeEncoder {
    low: u64,
    range: u64,
    buffer: u8,
    outstanding_bytes: usize,
    output: Vec<u8>,
}

impl RangeEncoder {
    pub fn new() -> Self {
        Self {
            low: 0,
            range: TOP_VALUE,
            buffer: 0,
            outstanding_bytes: 0,
            output: Vec::new(),
        }
    }

    /// Encode a single symbol given its CDF range
    pub fn encode_symbol(&mut self, low_count: u32, high_count: u32, total: u32) {
        let range = self.range / total as u64;
        self.low += range * low_count as u64;
        self.range = range * (high_count - low_count) as u64;
        self.normalize();
    }

    /// Normalize range and emit bytes
    fn normalize(&mut self) {
        while self.range < BOTTOM_VALUE {
            if self.low < (0xFF << (RANGE_BITS - 8)) as u64 {
                self.emit_byte(self.buffer);
                for _ in 0..self.outstanding_bytes {
                    self.emit_byte(0xFF);
                }
                self.outstanding_bytes = 0;
                self.buffer = (self.low >> (RANGE_BITS - 8)) as u8;
            } else if self.low >= TOP_VALUE {
                self.emit_byte(self.buffer + 1);
                for _ in 0..self.outstanding_bytes {
                    self.emit_byte(0x00);
                }
                self.outstanding_bytes = 0;
                self.buffer = (self.low >> (RANGE_BITS - 8)) as u8;
            } else {
                self.outstanding_bytes += 1;
            }
            self.low = (self.low << 8) & (TOP_VALUE - 1);
            self.range <<= 8;
        }
    }

    fn emit_byte(&mut self, byte: u8) {
        self.output.push(byte);
    }

    /// Finish encoding and return compressed bytes
    pub fn finish(mut self) -> Vec<u8> {
        // Flush remaining bits
        for _ in 0..5 {
            self.shift_low();
        }
        self.output
    }

    fn shift_low(&mut self) {
        if self.low < (0xFF << (RANGE_BITS - 8)) as u64 {
            self.emit_byte(self.buffer);
            for _ in 0..self.outstanding_bytes {
                self.emit_byte(0xFF);
            }
            self.outstanding_bytes = 0;
            self.buffer = (self.low >> (RANGE_BITS - 8)) as u8;
        } else if self.low >= TOP_VALUE {
            self.emit_byte(self.buffer + 1);
            for _ in 0..self.outstanding_bytes {
                self.emit_byte(0x00);
            }
            self.outstanding_bytes = 0;
            self.buffer = (self.low >> (RANGE_BITS - 8)) as u8;
        } else {
            self.outstanding_bytes += 1;
        }
        self.low = (self.low << 8) & (TOP_VALUE - 1);
    }
}

/// Range decoder
pub struct RangeDecoder<'a> {
    low: u64,
    range: u64,
    code: u64,
    data: &'a [u8],
    pos: usize,
}

impl<'a> RangeDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut decoder = Self {
            low: 0,
            range: TOP_VALUE,
            code: 0,
            data,
            pos: 0,
        };
        // Initialize code from first bytes
        for _ in 0..5 {
            decoder.code = (decoder.code << 8) | decoder.read_byte() as u64;
        }
        decoder
    }

    fn read_byte(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            b
        } else {
            0
        }
    }

    /// Decode a symbol given its CDF table
    pub fn decode_symbol(&mut self, cdf: &CDFTable) -> usize {
        let range = self.range / cdf.total as u64;
        let value = ((self.code - self.low) / range).min(cdf.total as u64 - 1) as u32;
        let symbol = cdf.find_symbol(value);

        let (low_count, high_count) = cdf.symbol_range(symbol);
        self.low += range * low_count as u64;
        self.range = range * (high_count - low_count) as u64;
        self.normalize();

        symbol
    }

    fn normalize(&mut self) {
        while self.range < BOTTOM_VALUE {
            self.code = (self.code << 8) | self.read_byte() as u64;
            self.low <<= 8;
            self.range <<= 8;
        }
    }
}

/// Encode quantized splats using range coding with provided CDFs
pub fn encode_splats(splats: &[QuantizedSplat], cdfs: &PredictedCDFs) -> Vec<u8> {
    let mut encoder = RangeEncoder::new();

    for splat in splats {
        // Encode position residuals (i16 -> unsigned offset)
        for (i, &val) in splat.pos_residual.iter().enumerate() {
            let symbol = (val as i32 + 2048).clamp(0, 4095) as usize;
            let (lo, hi) = cdfs.pos[i].symbol_range(symbol);
            encoder.encode_symbol(lo, hi, cdfs.pos[i].total);
        }

        // Encode scales
        for (i, &val) in splat.scale.iter().enumerate() {
            let (lo, hi) = cdfs.scale[i].symbol_range(val as usize);
            encoder.encode_symbol(lo, hi, cdfs.scale[i].total);
        }

        // Encode rotations (u16 -> 10-bit)
        for (i, &val) in splat.rotation.iter().enumerate() {
            let symbol = (val >> 6).min(1023) as usize; // Top 10 bits
            let (lo, hi) = cdfs.rotation[i].symbol_range(symbol);
            encoder.encode_symbol(lo, hi, cdfs.rotation[i].total);
        }

        // Encode colors
        for (i, &val) in splat.color.iter().enumerate() {
            let (lo, hi) = cdfs.color[i].symbol_range(val as usize);
            encoder.encode_symbol(lo, hi, cdfs.color[i].total);
        }

        // Encode opacity
        let (lo, hi) = cdfs.opacity.symbol_range(splat.opacity as usize);
        encoder.encode_symbol(lo, hi, cdfs.opacity.total);
    }

    encoder.finish()
}

/// Decode splats using range coding with provided CDFs
pub fn decode_splats(data: &[u8], count: usize, cdfs: &PredictedCDFs) -> Result<Vec<QuantizedSplat>> {
    let mut decoder = RangeDecoder::new(data);
    let mut splats = Vec::with_capacity(count);

    for _ in 0..count {
        // Decode position residuals
        let pos_residual = [
            (decoder.decode_symbol(&cdfs.pos[0]) as i32 - 2048) as i16,
            (decoder.decode_symbol(&cdfs.pos[1]) as i32 - 2048) as i16,
            (decoder.decode_symbol(&cdfs.pos[2]) as i32 - 2048) as i16,
        ];

        // Decode scales
        let scale = [
            decoder.decode_symbol(&cdfs.scale[0]) as u8,
            decoder.decode_symbol(&cdfs.scale[1]) as u8,
            decoder.decode_symbol(&cdfs.scale[2]) as u8,
        ];

        // Decode rotations (10-bit -> 16-bit)
        let rotation = [
            (decoder.decode_symbol(&cdfs.rotation[0]) << 6) as u16,
            (decoder.decode_symbol(&cdfs.rotation[1]) << 6) as u16,
            (decoder.decode_symbol(&cdfs.rotation[2]) << 6) as u16,
            (decoder.decode_symbol(&cdfs.rotation[3]) << 6) as u16,
        ];

        // Decode colors
        let color = [
            decoder.decode_symbol(&cdfs.color[0]) as u8,
            decoder.decode_symbol(&cdfs.color[1]) as u8,
            decoder.decode_symbol(&cdfs.color[2]) as u8,
        ];

        // Decode opacity
        let opacity = decoder.decode_symbol(&cdfs.opacity) as u8;

        splats.push(QuantizedSplat {
            pos_residual,
            scale,
            rotation,
            color,
            opacity,
            uncertainty: None,
        });
    }

    Ok(splats)
}

/// Arithmetic coder for splat compression (main interface)
pub struct SplatArithmeticCoder;

impl SplatArithmeticCoder {
    /// Encode quantized splats with baseline (uniform) CDFs
    pub fn encode_baseline(splats: &[QuantizedSplat]) -> (Vec<u8>, CompressionStats) {
        let cdfs = PredictedCDFs::uniform();
        let compressed = encode_splats(splats, &cdfs);

        let original_bytes = splats.len() * std::mem::size_of::<QuantizedSplat>();
        let stats = CompressionStats {
            original_bytes,
            compressed_bytes: compressed.len(),
            splat_count: splats.len(),
        };

        (compressed, stats)
    }

    /// Encode quantized splats with model-predicted CDFs
    pub fn encode_with_cdfs(splats: &[QuantizedSplat], cdfs: &PredictedCDFs) -> (Vec<u8>, CompressionStats) {
        let compressed = encode_splats(splats, cdfs);

        let original_bytes = splats.len() * std::mem::size_of::<QuantizedSplat>();
        let stats = CompressionStats {
            original_bytes,
            compressed_bytes: compressed.len(),
            splat_count: splats.len(),
        };

        (compressed, stats)
    }

    /// Decode splats with baseline (uniform) CDFs
    pub fn decode_baseline(data: &[u8], count: usize) -> Result<Vec<QuantizedSplat>> {
        let cdfs = PredictedCDFs::uniform();
        decode_splats(data, count, &cdfs)
    }

    /// Decode splats with model-predicted CDFs
    pub fn decode_with_cdfs(data: &[u8], count: usize, cdfs: &PredictedCDFs) -> Result<Vec<QuantizedSplat>> {
        decode_splats(data, count, cdfs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdf_uniform() {
        let cdf = CDFTable::uniform(256);
        assert_eq!(cdf.cumulative.len(), 257);
        assert_eq!(cdf.total, 256);
        assert_eq!(cdf.symbol_range(0), (0, 1));
        assert_eq!(cdf.symbol_range(255), (255, 256));
    }

    #[test]
    fn test_cdf_find_symbol() {
        let cdf = CDFTable::uniform(256);
        assert_eq!(cdf.find_symbol(0), 0);
        assert_eq!(cdf.find_symbol(127), 127);
        assert_eq!(cdf.find_symbol(255), 255);
    }

    #[test]
    fn test_roundtrip_baseline() {
        let splats = vec![
            QuantizedSplat {
                pos_residual: [100, -200, 300],
                scale: [50, 100, 150],
                rotation: [256, 512, 768, 1024],
                color: [255, 128, 64],
                opacity: 200,
                uncertainty: None,
            },
            QuantizedSplat {
                pos_residual: [-500, 1000, -1500],
                scale: [10, 20, 30],
                rotation: [0, 128, 256, 384],
                color: [0, 255, 0],
                opacity: 128,
                uncertainty: None,
            },
        ];

        let (compressed, stats) = SplatArithmeticCoder::encode_baseline(&splats);
        assert!(compressed.len() > 0);
        assert_eq!(stats.splat_count, 2);

        let decoded = SplatArithmeticCoder::decode_baseline(&compressed, 2).unwrap();
        assert_eq!(decoded.len(), 2);

        // Check first splat (allowing for 10-bit rotation quantization)
        assert_eq!(decoded[0].pos_residual, splats[0].pos_residual);
        assert_eq!(decoded[0].scale, splats[0].scale);
        assert_eq!(decoded[0].color, splats[0].color);
        assert_eq!(decoded[0].opacity, splats[0].opacity);
    }

    #[test]
    fn test_compression_ratio() {
        // Create 100 splats with random-ish data
        let splats: Vec<QuantizedSplat> = (0..100)
            .map(|i| QuantizedSplat {
                pos_residual: [(i * 13 % 4096 - 2048) as i16, (i * 17 % 4096 - 2048) as i16, (i * 19 % 4096 - 2048) as i16],
                scale: [(i * 7 % 256) as u8, (i * 11 % 256) as u8, (i * 23 % 256) as u8],
                rotation: [(i * 29 % 1024) as u16 * 64, (i * 31 % 1024) as u16 * 64, (i * 37 % 1024) as u16 * 64, (i * 41 % 1024) as u16 * 64],
                color: [(i * 43 % 256) as u8, (i * 47 % 256) as u8, (i * 53 % 256) as u8],
                opacity: (i * 59 % 256) as u8,
                uncertainty: None,
            })
            .collect();

        let (compressed, stats) = SplatArithmeticCoder::encode_baseline(&splats);

        println!("Original: {} bytes", stats.original_bytes);
        println!("Compressed: {} bytes", stats.compressed_bytes);
        println!("Ratio: {:.2}", stats.compression_ratio());
        println!("Bytes per splat: {:.2}", stats.bytes_per_splat());

        // Should achieve some compression with uniform CDFs
        // (entropy is less than log2 of each symbol alphabet combined)
        assert!(stats.compressed_bytes < stats.original_bytes);
    }
}
