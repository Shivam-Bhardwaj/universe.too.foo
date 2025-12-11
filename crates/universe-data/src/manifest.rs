use universe_core::grid::{HLGConfig, CellId};
use serde::{Serialize, Deserialize};
use std::path::Path;
use anyhow::Result;

/// Entry for a single cell in the manifest
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CellEntry {
    pub id: CellId,
    pub file_name: String,
    pub splat_count: u32,
    pub file_size_bytes: u64,
}

/// Universe manifest - index of all cells
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CellManifest {
    pub version: u32,
    pub config: HLGConfig,
    pub total_splats: u64,
    pub total_size_bytes: u64,
    pub cells: Vec<CellEntry>,
}

impl CellManifest {
    pub fn new(config: HLGConfig) -> Self {
        Self {
            version: 1,
            config,
            total_splats: 0,
            total_size_bytes: 0,
            cells: Vec::new(),
        }
    }

    pub fn add_cell(&mut self, entry: CellEntry) {
        self.total_splats += entry.splat_count as u64;
        self.total_size_bytes += entry.file_size_bytes;
        self.cells.push(entry);
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }

    /// Merge another manifest into this one
    pub fn merge(&mut self, other: CellManifest) {
        for entry in other.cells {
            // Check if we already have this cell
            if let Some(existing) = self.cells.iter_mut().find(|e| e.id == entry.id) {
                // Update existing cell
                self.total_splats -= existing.splat_count as u64;
                self.total_size_bytes -= existing.file_size_bytes;
                *existing = entry.clone();
                self.total_splats += entry.splat_count as u64;
                self.total_size_bytes += entry.file_size_bytes;
            } else {
                self.add_cell(entry);
            }
        }
    }
}
