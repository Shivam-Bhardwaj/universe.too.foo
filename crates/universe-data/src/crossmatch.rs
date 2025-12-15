//! Phase 3.4: Cross-match pipeline for multi-catalog fusion
//! 
//! Matches stars across catalogs using probabilistic matching based on
//! position uncertainty and quality scores.

use crate::canonical::{CanonicalStar, CatalogSource};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Phase 3.4: Cross-match result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrossMatch {
    /// Primary star (highest quality)
    pub primary: CanonicalStar,
    
    /// Matched stars from other catalogs
    pub matches: Vec<(CanonicalStar, f64)>, // (star, match_confidence)
    
    /// Combined quality score
    pub combined_quality: f64,
}

/// Phase 3.4: Cross-match pipeline
pub struct CrossMatchPipeline {
    /// Maximum Mahalanobis distance for matching
    pub max_distance: f64,
    
    /// Minimum confidence for accepting a match
    pub min_confidence: f64,
}

impl Default for CrossMatchPipeline {
    fn default() -> Self {
        Self {
            max_distance: 3.0, // 3-sigma match
            min_confidence: 0.5,
        }
    }
}

impl CrossMatchPipeline {
    /// Phase 3.4: Cross-match stars from multiple catalogs
    /// 
    /// Uses probabilistic matching: stars are matched if their Mahalanobis
    /// distance is below threshold, weighted by quality scores.
    pub fn cross_match(
        &self,
        stars_by_catalog: &HashMap<CatalogSource, Vec<CanonicalStar>>,
    ) -> Vec<CrossMatch> {
        let mut results = Vec::new();
        
        // Group stars by approximate position (spatial hash)
        let mut spatial_hash: HashMap<(i32, i32, i32), Vec<CanonicalStar>> = HashMap::new();
        
        for stars in stars_by_catalog.values() {
            for star in stars {
                // Simple spatial hash (1 parsec bins)
                let x = (star.position.x / universe_core::constants::PARSEC) as i32;
                let y = (star.position.y / universe_core::constants::PARSEC) as i32;
                let z = (star.position.z / universe_core::constants::PARSEC) as i32;
                
                spatial_hash.entry((x, y, z)).or_insert_with(Vec::new).push(star.clone());
            }
        }
        
        // Process each spatial bin
        for bin_stars in spatial_hash.values() {
            if bin_stars.is_empty() {
                continue;
            }
            
            // Find best primary (highest quality)
            let primary_idx = bin_stars
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.quality_score.partial_cmp(&b.1.quality_score).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            
            let primary = bin_stars[primary_idx].clone();
            let mut matches = Vec::new();
            
            // Match other stars in bin
            for (idx, star) in bin_stars.iter().enumerate() {
                if idx == primary_idx {
                    continue;
                }
                
                let distance = primary.mahalanobis_distance(star);
                if distance <= self.max_distance {
                    // Confidence based on distance and quality
                    let confidence = (-distance / 2.0).exp() * star.quality_score / primary.quality_score;
                    if confidence >= self.min_confidence {
                        matches.push((star.clone(), confidence));
                    }
                }
            }
            
            // Compute combined quality
            let combined_quality = primary.quality_score
                + matches.iter().map(|(s, c)| s.quality_score * c).sum::<f64>();
            
            results.push(CrossMatch {
                primary,
                matches,
                combined_quality,
            });
        }
        
        results
    }
}



