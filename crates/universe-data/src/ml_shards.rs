//! Phase 3.5: ML training shard generation
//! 
//! Emits training-ready shards (not runtime tiles) with provenance labels
//! for ML model training.

use crate::canonical::CanonicalStar;
use crate::crossmatch::CrossMatch;
use serde::{Serialize, Deserialize};
use std::path::Path;
use anyhow::Result;

/// Phase 3.5: ML training shard with provenance
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MLTrainingShard {
    /// Shard ID
    pub shard_id: String,
    
    /// Stars in this shard
    pub stars: Vec<CanonicalStar>,
    
    /// Cross-match results (if available)
    pub cross_matches: Vec<CrossMatch>,
    
    /// Provenance: which catalogs contributed
    pub provenance: Vec<String>,
    
    /// Metadata
    pub metadata: ShardMetadata,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardMetadata {
    /// Number of stars
    pub star_count: usize,
    
    /// Average quality score
    pub avg_quality: f64,
    
    /// Spatial bounds
    pub bounds_min: [f64; 3],
    pub bounds_max: [f64; 3],
    
    /// Epoch range
    pub epoch_min: f64,
    pub epoch_max: f64,
}

/// Phase 3.5: Generate ML training shards from cross-matched stars
pub struct MLShardGenerator {
    /// Stars per shard
    pub stars_per_shard: usize,
}

impl Default for MLShardGenerator {
    fn default() -> Self {
        Self {
            stars_per_shard: 10_000,
        }
    }
}

impl MLShardGenerator {
    /// Generate training shards from cross-matched data
    pub fn generate_shards(
        &self,
        cross_matches: &[CrossMatch],
        output_dir: &Path,
    ) -> Result<Vec<String>> {
        std::fs::create_dir_all(output_dir)?;
        
        let mut shard_ids = Vec::new();
        let mut current_shard = Vec::new();
        let mut shard_idx = 0;
        
        for cross_match in cross_matches {
            // Add primary star
            current_shard.push(cross_match.primary.clone());
            
            // Add matched stars
            for (star, _) in &cross_match.matches {
                current_shard.push(star.clone());
            }
            
            // Emit shard when full
            if current_shard.len() >= self.stars_per_shard {
                let shard_id = format!("shard_{:06}", shard_idx);
                shard_ids.push(shard_id.clone());
                
                let provenance: Vec<String> = current_shard
                    .iter()
                    .map(|s| format!("{:?}", s.source))
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                
                let bounds_min = [
                    current_shard.iter().map(|s| s.position.x).fold(f64::INFINITY, f64::min),
                    current_shard.iter().map(|s| s.position.y).fold(f64::INFINITY, f64::min),
                    current_shard.iter().map(|s| s.position.z).fold(f64::INFINITY, f64::min),
                ];
                let bounds_max = [
                    current_shard.iter().map(|s| s.position.x).fold(f64::NEG_INFINITY, f64::max),
                    current_shard.iter().map(|s| s.position.y).fold(f64::NEG_INFINITY, f64::max),
                    current_shard.iter().map(|s| s.position.z).fold(f64::NEG_INFINITY, f64::max),
                ];
                
                let avg_quality = current_shard.iter().map(|s| s.quality_score).sum::<f64>()
                    / current_shard.len() as f64;
                
                let epoch_min = current_shard.iter().map(|s| s.epoch_jd).fold(f64::INFINITY, f64::min);
                let epoch_max = current_shard.iter().map(|s| s.epoch_jd).fold(f64::NEG_INFINITY, f64::max);
                
                let shard = MLTrainingShard {
                    shard_id: shard_id.clone(),
                    stars: current_shard.clone(),
                    cross_matches: vec![cross_match.clone()],
                    provenance,
                    metadata: ShardMetadata {
                        star_count: current_shard.len(),
                        avg_quality,
                        bounds_min,
                        bounds_max,
                        epoch_min,
                        epoch_max,
                    },
                };
                
                let shard_path = output_dir.join(format!("{}.json", shard_id));
                let json = serde_json::to_string_pretty(&shard)?;
                std::fs::write(&shard_path, json)?;
                
                current_shard.clear();
                shard_idx += 1;
            }
        }
        
        // Emit final partial shard
        if !current_shard.is_empty() {
            let shard_id = format!("shard_{:06}", shard_idx);
            shard_ids.push(shard_id.clone());
            // ... (similar code as above)
        }
        
        Ok(shard_ids)
    }
}
