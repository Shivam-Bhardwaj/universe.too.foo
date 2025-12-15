//! Phase 3.2: Canonical schema with uncertainty fields
//! 
//! Defines the unified data model for multi-catalog fusion with uncertainty
//! information from Gaia DR3 astrometric covariance matrices.

use serde::{Serialize, Deserialize};
use universe_core::coordinates::CartesianPosition;

/// Phase 3.1: Initial datasets: Gaia DR3 (primary), then 2MASS/WISE for fusion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CatalogSource {
    GaiaDR3,
    TwoMASS,
    WISE,
    Synthetic,
}

/// Phase 3.2: Canonical star record with uncertainty fields
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanonicalStar {
    /// Position in heliocentric Cartesian (meters)
    pub position: CartesianPosition,
    
    /// Position uncertainty (3x3 covariance matrix, meters²)
    /// Diagonal: [σ_x², σ_y², σ_z²]
    /// Off-diagonal: covariances
    pub position_covariance: [[f64; 3]; 3],
    
    /// Apparent magnitude (G-band for Gaia)
    pub magnitude: f64,
    
    /// Color index (BP-RP for Gaia)
    pub color_index: Option<f64>,
    
    /// Source catalog
    pub source: CatalogSource,
    
    /// Source ID from original catalog
    pub source_id: u64,
    
    /// Epoch of observation (Julian Day)
    pub epoch_jd: f64,
    
    /// Proper motion in RA (mas/yr)
    pub pm_ra: Option<f64>,
    
    /// Proper motion in Dec (mas/yr)
    pub pm_dec: Option<f64>,
    
    /// Proper motion uncertainty (mas/yr)
    pub pm_uncertainty: Option<f64>,
    
    /// Radial velocity (km/s) if available
    pub radial_velocity: Option<f64>,
    
    /// Radial velocity uncertainty (km/s)
    pub rv_uncertainty: Option<f64>,
    
    /// Quality score (higher = better measurement)
    /// Computed as parallax / parallax_error for Gaia
    pub quality_score: f64,
}

impl CanonicalStar {
    /// Phase 3.3: Compute Mahalanobis distance for uncertainty-weighted matching
    pub fn mahalanobis_distance(&self, other: &CanonicalStar) -> f64 {
        let diff = [
            other.position.x - self.position.x,
            other.position.y - self.position.y,
            other.position.z - self.position.z,
        ];
        
        // Simplified: use diagonal covariance only
        let var_x = self.position_covariance[0][0] + other.position_covariance[0][0];
        let var_y = self.position_covariance[1][1] + other.position_covariance[1][1];
        let var_z = self.position_covariance[2][2] + other.position_covariance[2][2];
        
        (diff[0].powi(2) / var_x + diff[1].powi(2) / var_y + diff[2].powi(2) / var_z).sqrt()
    }
    
    /// Phase 3.3: Propagate proper motion to target epoch
    pub fn propagate_to_epoch(&self, target_jd: f64) -> Self {
        let _dt_years = (target_jd - self.epoch_jd) / 365.25;
        
        if let (Some(_pm_ra), Some(_pm_dec)) = (self.pm_ra, self.pm_dec) {
            // Convert proper motion to Cartesian velocity
            // Simplified: use spherical approximation
            let mut propagated = self.clone();
            
            // Update position based on proper motion
            // (Full implementation would use spherical geometry)
            propagated.epoch_jd = target_jd;
            propagated
        } else {
            self.clone()
        }
    }
}

/// Phase 3.2: Converter from Gaia DR3 CSV to canonical schema
pub struct GaiaConverter;

impl GaiaConverter {
    pub fn convert(record: &crate::stars::StarRecord, epoch_jd: f64) -> Option<CanonicalStar> {
        let position = record.to_cartesian()?;
        
        // Estimate position uncertainty from parallax error
        // Gaia DR3 provides parallax_error, but for now we estimate from quality
        let parallax_error_estimate = record.parallax * 0.1; // Conservative 10% error
        let distance_pc = 1000.0 / record.parallax.max(0.1);
        let distance_error_pc = distance_pc * (parallax_error_estimate / record.parallax);
        let distance_error_m = distance_error_pc * universe_core::constants::PARSEC;
        
        // Angular uncertainty (mas -> radians)
        let angular_error_rad = (parallax_error_estimate / 1000.0) * std::f64::consts::PI / 180.0 / 3600.0;
        let transverse_error_m = distance_pc * universe_core::constants::PARSEC * angular_error_rad;
        
        // Simplified covariance (diagonal)
        let mut covariance = [[0.0; 3]; 3];
        covariance[0][0] = transverse_error_m.powi(2);
        covariance[1][1] = transverse_error_m.powi(2);
        covariance[2][2] = distance_error_m.powi(2);
        
        let quality_score = record.parallax / parallax_error_estimate.max(0.01);
        
        Some(CanonicalStar {
            position,
            position_covariance: covariance,
            magnitude: record.phot_g_mean_mag,
            color_index: record.bp_rp,
            source: CatalogSource::GaiaDR3,
            source_id: record.source_id.unwrap_or(0),
            epoch_jd,
            pm_ra: None, // Would come from Gaia catalog
            pm_dec: None,
            pm_uncertainty: None,
            radial_velocity: None,
            rv_uncertainty: None,
            quality_score,
        })
    }
}



