//! Phase 5.3: Entropy model trainer for learned compression
//!
//! Trains the EntropyContextModel to predict symbol distributions for
//! arithmetic coding, achieving better compression than uniform CDFs.
//!
//! This module provides:
//! - `EntropyTrainer`: Trains the context model on quantized splat data
//! - `EntropyTrainConfig`: Training hyperparameters
//! - CDF serialization/deserialization for deployment

use std::path::Path;
use universe_data::arithmetic_coder::{CDFTable, PredictedCDFs};
use universe_data::compression::QuantizedSplat;

/// Training configuration for entropy model
#[derive(Clone, Debug)]
pub struct EntropyTrainConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size (number of splats per batch)
    pub batch_size: usize,
    /// Smoothing factor for count-based CDFs (Laplace smoothing)
    pub smoothing: f32,
    /// Minimum probability (prevents zero probabilities)
    pub min_prob: f32,
}

impl Default for EntropyTrainConfig {
    fn default() -> Self {
        Self {
            epochs: 1,  // Count-based doesn't need multiple epochs
            batch_size: 10000,
            smoothing: 1.0,  // Laplace smoothing
            min_prob: 1e-6,
        }
    }
}

/// Training metrics for entropy model
#[derive(Clone, Debug)]
pub struct EntropyTrainingMetrics {
    pub splat_count: usize,
    pub estimated_bits_per_splat: f32,
    pub position_entropy: f32,
    pub scale_entropy: f32,
    pub rotation_entropy: f32,
    pub color_entropy: f32,
    pub opacity_entropy: f32,
}

/// Entropy model trainer using histogram-based CDF estimation
///
/// This is a simplified but effective approach that builds CDFs directly from
/// observed symbol frequencies. For larger datasets, this achieves near-optimal
/// compression without the complexity of neural network training.
pub struct EntropyTrainer {
    config: EntropyTrainConfig,
    // Accumulated histogram counts
    pos_counts: [Vec<u32>; 3],
    scale_counts: [Vec<u32>; 3],
    rotation_counts: [Vec<u32>; 4],
    color_counts: [Vec<u32>; 3],
    opacity_counts: Vec<u32>,
    total_splats: usize,
}

impl EntropyTrainer {
    /// Create new trainer with default config
    pub fn new(config: EntropyTrainConfig) -> Self {
        Self {
            config,
            pos_counts: [
                vec![0u32; 4096],
                vec![0u32; 4096],
                vec![0u32; 4096],
            ],
            scale_counts: [
                vec![0u32; 256],
                vec![0u32; 256],
                vec![0u32; 256],
            ],
            rotation_counts: [
                vec![0u32; 1024],
                vec![0u32; 1024],
                vec![0u32; 1024],
                vec![0u32; 1024],
            ],
            color_counts: [
                vec![0u32; 256],
                vec![0u32; 256],
                vec![0u32; 256],
            ],
            opacity_counts: vec![0u32; 256],
            total_splats: 0,
        }
    }

    /// Train on quantized splats (accumulate histogram counts)
    pub fn train(&mut self, splats: &[QuantizedSplat]) -> EntropyTrainingMetrics {
        tracing::info!("Training entropy model on {} splats", splats.len());

        for splat in splats {
            self.add_splat(splat);
        }

        self.total_splats += splats.len();

        // Calculate entropy estimates
        let pos_entropy = self.estimate_entropy(&self.pos_counts[0], 4096)
            + self.estimate_entropy(&self.pos_counts[1], 4096)
            + self.estimate_entropy(&self.pos_counts[2], 4096);

        let scale_entropy = self.estimate_entropy(&self.scale_counts[0], 256)
            + self.estimate_entropy(&self.scale_counts[1], 256)
            + self.estimate_entropy(&self.scale_counts[2], 256);

        let rotation_entropy = self.estimate_entropy(&self.rotation_counts[0], 1024)
            + self.estimate_entropy(&self.rotation_counts[1], 1024)
            + self.estimate_entropy(&self.rotation_counts[2], 1024)
            + self.estimate_entropy(&self.rotation_counts[3], 1024);

        let color_entropy = self.estimate_entropy(&self.color_counts[0], 256)
            + self.estimate_entropy(&self.color_counts[1], 256)
            + self.estimate_entropy(&self.color_counts[2], 256);

        let opacity_entropy = self.estimate_entropy(&self.opacity_counts, 256);

        let total_entropy = pos_entropy + scale_entropy + rotation_entropy + color_entropy + opacity_entropy;

        // Bits per splat = sum of all attribute entropies
        let bits_per_splat = total_entropy;

        tracing::info!(
            "Entropy estimates: pos={:.2}, scale={:.2}, rot={:.2}, color={:.2}, opacity={:.2}",
            pos_entropy, scale_entropy, rotation_entropy, color_entropy, opacity_entropy
        );
        tracing::info!("Total estimated bits/splat: {:.2} ({:.2} bytes/splat)", bits_per_splat, bits_per_splat / 8.0);

        EntropyTrainingMetrics {
            splat_count: self.total_splats,
            estimated_bits_per_splat: bits_per_splat,
            position_entropy: pos_entropy,
            scale_entropy,
            rotation_entropy,
            color_entropy,
            opacity_entropy,
        }
    }

    /// Add a single splat to histogram
    fn add_splat(&mut self, splat: &QuantizedSplat) {
        // Position (offset by 2048 to convert to unsigned)
        for (i, &val) in splat.pos_residual.iter().enumerate() {
            let symbol = (val as i32 + 2048).clamp(0, 4095) as usize;
            self.pos_counts[i][symbol] += 1;
        }

        // Scale
        for (i, &val) in splat.scale.iter().enumerate() {
            self.scale_counts[i][val as usize] += 1;
        }

        // Rotation (top 10 bits)
        for (i, &val) in splat.rotation.iter().enumerate() {
            let symbol = (val >> 6).min(1023) as usize;
            self.rotation_counts[i][symbol] += 1;
        }

        // Color
        for (i, &val) in splat.color.iter().enumerate() {
            self.color_counts[i][val as usize] += 1;
        }

        // Opacity
        self.opacity_counts[splat.opacity as usize] += 1;
    }

    /// Estimate entropy in bits for a histogram
    fn estimate_entropy(&self, counts: &[u32], num_symbols: usize) -> f32 {
        let total: u32 = counts.iter().sum();
        if total == 0 {
            return (num_symbols as f32).log2(); // Maximum entropy
        }

        let smoothing = self.config.smoothing;
        let total_smoothed = total as f32 + smoothing * num_symbols as f32;

        let mut entropy = 0.0f32;
        for &count in counts {
            if count > 0 {
                let prob = (count as f32 + smoothing) / total_smoothed;
                entropy -= prob * prob.log2();
            }
        }

        entropy
    }

    /// Build CDFs from accumulated counts
    pub fn build_cdfs(&self) -> PredictedCDFs {
        let smoothing = self.config.smoothing as u32;

        PredictedCDFs {
            pos: [
                self.counts_to_cdf(&self.pos_counts[0], smoothing),
                self.counts_to_cdf(&self.pos_counts[1], smoothing),
                self.counts_to_cdf(&self.pos_counts[2], smoothing),
            ],
            scale: [
                self.counts_to_cdf(&self.scale_counts[0], smoothing),
                self.counts_to_cdf(&self.scale_counts[1], smoothing),
                self.counts_to_cdf(&self.scale_counts[2], smoothing),
            ],
            rotation: [
                self.counts_to_cdf(&self.rotation_counts[0], smoothing),
                self.counts_to_cdf(&self.rotation_counts[1], smoothing),
                self.counts_to_cdf(&self.rotation_counts[2], smoothing),
                self.counts_to_cdf(&self.rotation_counts[3], smoothing),
            ],
            color: [
                self.counts_to_cdf(&self.color_counts[0], smoothing),
                self.counts_to_cdf(&self.color_counts[1], smoothing),
                self.counts_to_cdf(&self.color_counts[2], smoothing),
            ],
            opacity: self.counts_to_cdf(&self.opacity_counts, smoothing),
        }
    }

    /// Convert histogram counts to CDF with smoothing
    fn counts_to_cdf(&self, counts: &[u32], smoothing: u32) -> CDFTable {
        let n = counts.len();
        let mut cumulative = Vec::with_capacity(n);
        let mut sum = 0u32;

        for &count in counts {
            cumulative.push(sum);
            sum += count + smoothing; // Add smoothing to each bin
        }

        CDFTable { cumulative, total: sum }
    }

    /// Save learned CDFs to file
    pub fn save_cdfs(&self, path: &Path) -> anyhow::Result<()> {
        let cdfs = self.build_cdfs();
        let data = serialize_cdfs(&cdfs);
        std::fs::write(path, data)?;
        tracing::info!("Saved CDFs to {:?} ({} bytes)", path, std::fs::metadata(path)?.len());
        Ok(())
    }

    /// Get total splats processed
    pub fn total_splats(&self) -> usize {
        self.total_splats
    }

    /// Reset histogram counts
    pub fn reset(&mut self) {
        for counts in &mut self.pos_counts {
            counts.fill(0);
        }
        for counts in &mut self.scale_counts {
            counts.fill(0);
        }
        for counts in &mut self.rotation_counts {
            counts.fill(0);
        }
        for counts in &mut self.color_counts {
            counts.fill(0);
        }
        self.opacity_counts.fill(0);
        self.total_splats = 0;
    }
}

impl Default for EntropyTrainer {
    fn default() -> Self {
        Self::new(EntropyTrainConfig::default())
    }
}

/// Serialize learned CDFs for deployment
pub fn serialize_cdfs(cdfs: &PredictedCDFs) -> Vec<u8> {
    let mut data = Vec::new();

    // Version
    data.extend_from_slice(&1u32.to_le_bytes());

    fn write_cdf(data: &mut Vec<u8>, table: &CDFTable) {
        // Number of symbols
        data.extend_from_slice(&(table.cumulative.len() as u32).to_le_bytes());
        // Total
        data.extend_from_slice(&table.total.to_le_bytes());
        // Cumulative values
        for &val in &table.cumulative {
            data.extend_from_slice(&val.to_le_bytes());
        }
    }

    // Position (3)
    for t in &cdfs.pos {
        write_cdf(&mut data, t);
    }
    // Scale (3)
    for t in &cdfs.scale {
        write_cdf(&mut data, t);
    }
    // Rotation (4)
    for t in &cdfs.rotation {
        write_cdf(&mut data, t);
    }
    // Color (3)
    for t in &cdfs.color {
        write_cdf(&mut data, t);
    }
    // Opacity (1)
    write_cdf(&mut data, &cdfs.opacity);

    data
}

/// Deserialize learned CDFs
pub fn deserialize_cdfs(data: &[u8]) -> anyhow::Result<PredictedCDFs> {
    let mut cursor = 0usize;

    fn read_u32(data: &[u8], cursor: &mut usize) -> anyhow::Result<u32> {
        if *cursor + 4 > data.len() {
            anyhow::bail!("Unexpected end of CDF data");
        }
        let val = u32::from_le_bytes([
            data[*cursor],
            data[*cursor + 1],
            data[*cursor + 2],
            data[*cursor + 3],
        ]);
        *cursor += 4;
        Ok(val)
    }

    fn read_cdf(data: &[u8], cursor: &mut usize) -> anyhow::Result<CDFTable> {
        let num_symbols = read_u32(data, cursor)? as usize;
        let total = read_u32(data, cursor)?;
        let mut cumulative = Vec::with_capacity(num_symbols);
        for _ in 0..num_symbols {
            cumulative.push(read_u32(data, cursor)?);
        }
        Ok(CDFTable { cumulative, total })
    }

    // Version check
    let version = read_u32(data, &mut cursor)?;
    if version != 1 {
        anyhow::bail!("Unsupported CDF version: {}", version);
    }

    // Read all tables
    let pos = [
        read_cdf(data, &mut cursor)?,
        read_cdf(data, &mut cursor)?,
        read_cdf(data, &mut cursor)?,
    ];
    let scale = [
        read_cdf(data, &mut cursor)?,
        read_cdf(data, &mut cursor)?,
        read_cdf(data, &mut cursor)?,
    ];
    let rotation = [
        read_cdf(data, &mut cursor)?,
        read_cdf(data, &mut cursor)?,
        read_cdf(data, &mut cursor)?,
        read_cdf(data, &mut cursor)?,
    ];
    let color = [
        read_cdf(data, &mut cursor)?,
        read_cdf(data, &mut cursor)?,
        read_cdf(data, &mut cursor)?,
    ];
    let opacity = read_cdf(data, &mut cursor)?;

    Ok(PredictedCDFs {
        pos,
        scale,
        rotation,
        color,
        opacity,
    })
}

/// Load CDFs from file
pub fn load_cdfs(path: &Path) -> anyhow::Result<PredictedCDFs> {
    let data = std::fs::read(path)?;
    deserialize_cdfs(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_trainer_basic() {
        let splats = vec![
            QuantizedSplat {
                pos_residual: [100, -200, 300],
                scale: [50, 100, 150],
                rotation: [256, 512, 768, 1024],
                color: [255, 128, 64],
                opacity: 200,
                uncertainty: None,
            },
        ];

        let mut trainer = EntropyTrainer::new(EntropyTrainConfig::default());
        let metrics = trainer.train(&splats);

        assert_eq!(metrics.splat_count, 1);
        assert!(metrics.estimated_bits_per_splat > 0.0);
    }

    #[test]
    fn test_cdf_serialization_roundtrip() {
        let cdfs = PredictedCDFs::uniform();
        let serialized = serialize_cdfs(&cdfs);
        let deserialized = deserialize_cdfs(&serialized).unwrap();

        // Check key properties
        assert_eq!(cdfs.pos[0].total, deserialized.pos[0].total);
        assert_eq!(cdfs.opacity.total, deserialized.opacity.total);
        assert_eq!(cdfs.pos[0].cumulative.len(), deserialized.pos[0].cumulative.len());
    }

    #[test]
    fn test_learned_cdfs_improve_compression() {
        // Create 1000 splats with clustered values
        let splats: Vec<QuantizedSplat> = (0..1000)
            .map(|i| QuantizedSplat {
                // Clustered around 0
                pos_residual: [
                    ((i % 100) as i32 - 50) as i16,
                    ((i % 50) as i32 - 25) as i16,
                    ((i % 20) as i32 - 10) as i16,
                ],
                // Small scales
                scale: [10, 15, 20],
                // Similar rotations
                rotation: [512 * 64, 512 * 64, 512 * 64, 512 * 64],
                // Varied colors
                color: [(i % 256) as u8, ((i * 2) % 256) as u8, ((i * 3) % 256) as u8],
                opacity: 200,
                uncertainty: None,
            })
            .collect();

        let mut trainer = EntropyTrainer::new(EntropyTrainConfig::default());
        let metrics = trainer.train(&splats);

        // For clustered data, entropy should be lower than maximum
        // Maximum entropy for all attributes would be:
        // pos: 3 * 12 bits = 36 bits
        // scale: 3 * 8 bits = 24 bits
        // rotation: 4 * 10 bits = 40 bits
        // color: 3 * 8 bits = 24 bits
        // opacity: 8 bits
        // Total max: 132 bits
        // With clustering, should be much lower
        assert!(metrics.estimated_bits_per_splat < 100.0);

        // Build and verify CDFs
        let cdfs = trainer.build_cdfs();

        // Opacity CDF should strongly favor 200
        let opacity_200_prob = cdfs.opacity.cumulative.get(201).unwrap_or(&0)
            - cdfs.opacity.cumulative.get(200).unwrap_or(&0);
        let opacity_0_prob = cdfs.opacity.cumulative.get(1).unwrap_or(&0);

        // Value 200 should have much higher count than 0
        assert!(opacity_200_prob > opacity_0_prob * 10);
    }

    #[test]
    fn test_entropy_estimates() {
        // All same values = 0 entropy
        let splats: Vec<QuantizedSplat> = (0..100)
            .map(|_| QuantizedSplat {
                pos_residual: [0, 0, 0],
                scale: [100, 100, 100],
                rotation: [512 * 64, 512 * 64, 512 * 64, 512 * 64],
                color: [128, 128, 128],
                opacity: 200,
                uncertainty: None,
            })
            .collect();

        let mut trainer = EntropyTrainer::new(EntropyTrainConfig {
            smoothing: 0.01, // Low smoothing to see near-zero entropy
            ..Default::default()
        });
        let metrics = trainer.train(&splats);

        // Entropy should be very low for constant values
        // Each attribute contributes near-zero entropy
        println!("Bits per splat for constant data: {:.2}", metrics.estimated_bits_per_splat);
        assert!(metrics.estimated_bits_per_splat < 5.0);
    }

    #[test]
    fn test_save_load_cdfs() {
        let splats: Vec<QuantizedSplat> = (0..100)
            .map(|i| QuantizedSplat {
                pos_residual: [(i % 100 - 50) as i16, 0, 0],
                scale: [50, 50, 50],
                rotation: [0, 0, 0, 0],
                color: [128, 128, 128],
                opacity: 200,
                uncertainty: None,
            })
            .collect();

        let mut trainer = EntropyTrainer::new(EntropyTrainConfig::default());
        trainer.train(&splats);

        // Save to temp file
        let temp_path = std::env::temp_dir().join("test_cdfs.bin");
        trainer.save_cdfs(&temp_path).unwrap();

        // Load back
        let loaded = load_cdfs(&temp_path).unwrap();
        let original = trainer.build_cdfs();

        // Verify equality
        assert_eq!(original.opacity.total, loaded.opacity.total);
        assert_eq!(original.pos[0].cumulative.len(), loaded.pos[0].cumulative.len());

        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }
}
