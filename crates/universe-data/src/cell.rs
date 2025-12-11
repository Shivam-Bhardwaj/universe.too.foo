use crate::splat::GaussianSplat;
use universe_core::grid::{CellId, CellBounds};
use serde::{Serialize, Deserialize};
use std::io::{Read, Write};
use anyhow::Result;

/// Metadata for a cell
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CellMetadata {
    pub id: CellId,
    pub bounds: CellBounds,
    pub splat_count: u32,
    pub compressed: bool,
}

/// Cell data container
pub struct CellData {
    pub metadata: CellMetadata,
    pub splats: Vec<GaussianSplat>,
}

impl CellData {
    pub fn new(id: CellId, bounds: CellBounds) -> Self {
        Self {
            metadata: CellMetadata {
                id,
                bounds,
                splat_count: 0,
                compressed: false,
            },
            splats: Vec::new(),
        }
    }

    pub fn add_splat(&mut self, splat: GaussianSplat) {
        self.splats.push(splat);
        self.metadata.splat_count = self.splats.len() as u32;
    }

    /// Serialize to compressed binary format
    pub fn serialize<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Write metadata as JSON header (length-prefixed)
        let meta_json = serde_json::to_vec(&self.metadata)?;
        let meta_len = meta_json.len() as u32;
        writer.write_all(&meta_len.to_le_bytes())?;
        writer.write_all(&meta_json)?;

        // Write splats as raw bytes
        let splat_bytes = bytemuck::cast_slice(&self.splats);

        // Compress with LZ4
        let compressed = lz4_flex::compress_prepend_size(splat_bytes);
        writer.write_all(&compressed)?;

        Ok(())
    }

    /// Deserialize from compressed binary format
    pub fn deserialize<R: Read>(reader: &mut R) -> Result<Self> {
        // Read metadata
        let mut meta_len_bytes = [0u8; 4];
        reader.read_exact(&mut meta_len_bytes)?;
        let meta_len = u32::from_le_bytes(meta_len_bytes) as usize;

        let mut meta_json = vec![0u8; meta_len];
        reader.read_exact(&mut meta_json)?;
        let metadata: CellMetadata = serde_json::from_slice(&meta_json)?;

        // Read and decompress splats
        let mut compressed = Vec::new();
        reader.read_to_end(&mut compressed)?;
        let splat_bytes = lz4_flex::decompress_size_prepended(&compressed)?;

        let splats: Vec<GaussianSplat> = bytemuck::cast_slice(&splat_bytes).to_vec();

        Ok(Self { metadata, splats })
    }

    /// Save to file
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let mut file = std::fs::File::create(path)?;
        self.serialize(&mut file)
    }

    /// Load from file
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        Self::deserialize(&mut file)
    }
}
