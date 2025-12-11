//! Star catalog processing for Gaia DR3 and other sources

use universe_core::coordinates::CartesianPosition;
use universe_core::constants::PARSEC;
use serde::{Serialize, Deserialize};
use std::path::Path;
use anyhow::Result;

/// Raw star data from catalog
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StarRecord {
    /// Right ascension in degrees [0, 360)
    pub ra: f64,
    /// Declination in degrees [-90, 90]
    pub dec: f64,
    /// Parallax in milliarcseconds
    pub parallax: f64,
    /// G-band apparent magnitude
    pub phot_g_mean_mag: f64,
    /// BP-RP color index (blue minus red)
    pub bp_rp: Option<f64>,
    /// Source ID for reference
    pub source_id: Option<u64>,
}

impl StarRecord {
    /// Convert equatorial coordinates (RA/Dec) to heliocentric Cartesian (meters)
    /// Uses J2000 equatorial frame, then rotates to ecliptic
    pub fn to_cartesian(&self) -> Option<CartesianPosition> {
        // Skip invalid parallax
        if self.parallax <= 0.0 || !self.parallax.is_finite() {
            return None;
        }

        // Distance in parsecs = 1000 / parallax (mas)
        let distance_pc = 1000.0 / self.parallax;

        // Skip unreasonably distant (noisy parallax) or nearby stars
        if distance_pc > 100_000.0 || distance_pc < 0.1 {
            return None;
        }

        // Convert to meters
        let distance_m = distance_pc * PARSEC;

        // Convert angles to radians
        let ra_rad = self.ra.to_radians();
        let dec_rad = self.dec.to_radians();

        // Equatorial (J2000) to Cartesian
        let cos_dec = dec_rad.cos();
        let x_eq = distance_m * cos_dec * ra_rad.cos();
        let y_eq = distance_m * cos_dec * ra_rad.sin();
        let z_eq = distance_m * dec_rad.sin();

        // Rotate from equatorial to ecliptic (J2000)
        // Obliquity ε = 23.4392911°
        let obliquity = 23.4392911_f64.to_radians();
        let cos_e = obliquity.cos();
        let sin_e = obliquity.sin();

        // Rotation matrix Rx(-ε)
        let x = x_eq;
        let y = y_eq * cos_e + z_eq * sin_e;
        let z = -y_eq * sin_e + z_eq * cos_e;

        Some(CartesianPosition::new(x, y, z))
    }

    /// Absolute magnitude from apparent magnitude and parallax
    pub fn absolute_magnitude(&self) -> Option<f64> {
        if self.parallax <= 0.0 {
            return None;
        }
        let distance_pc = 1000.0 / self.parallax;
        // M = m - 5*log10(d) + 5
        Some(self.phot_g_mean_mag - 5.0 * distance_pc.log10() + 5.0)
    }

    /// Estimated radius in meters (rough main-sequence approximation)
    pub fn estimated_radius_m(&self) -> f64 {
        let abs_mag = self.absolute_magnitude().unwrap_or(5.0);
        let solar_radius = 6.96e8;

        // Rough: brighter = bigger (main sequence approximation)
        // Sun: M = 4.83
        let mag_diff = 4.83 - abs_mag;
        let lum_ratio = 10.0_f64.powf(mag_diff / 2.5);
        let radius_ratio = lum_ratio.sqrt().clamp(0.1, 1000.0);

        solar_radius * radius_ratio
    }

    /// Convert BP-RP color to RGB
    pub fn color_rgb(&self) -> [f32; 3] {
        match self.bp_rp {
            Some(bp_rp) => bp_rp_to_rgb(bp_rp),
            None => [1.0, 0.98, 0.95], // Default white-ish
        }
    }
}

/// Convert Gaia BP-RP color index to RGB using blackbody approximation
fn bp_rp_to_rgb(bp_rp: f64) -> [f32; 3] {
    // Estimate color temperature from BP-RP
    let temp = if bp_rp < -0.5 {
        30000.0
    } else if bp_rp > 4.0 {
        2000.0
    } else {
        // Approximate formula
        let t1 = 1.0 / (0.92 * bp_rp + 1.7);
        let t2 = 1.0 / (0.92 * bp_rp + 0.62);
        4600.0 * (t1 + t2)
    };

    blackbody_to_rgb(temp)
}

/// Blackbody temperature to RGB (Tanner Helland algorithm)
fn blackbody_to_rgb(temp: f64) -> [f32; 3] {
    let temp = temp.clamp(1000.0, 40000.0) / 100.0;

    let r = if temp <= 66.0 {
        255.0
    } else {
        (329.698727446 * (temp - 60.0).powf(-0.1332047592)).clamp(0.0, 255.0)
    };

    let g = if temp <= 66.0 {
        (99.4708025861 * temp.ln() - 161.1195681661).clamp(0.0, 255.0)
    } else {
        (288.1221695283 * (temp - 60.0).powf(-0.0755148492)).clamp(0.0, 255.0)
    };

    let b = if temp >= 66.0 {
        255.0
    } else if temp <= 19.0 {
        0.0
    } else {
        (138.5177312231 * (temp - 10.0).ln() - 305.0447927307).clamp(0.0, 255.0)
    };

    [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
}

/// Star catalog container
pub struct StarCatalog {
    stars: Vec<StarRecord>,
}

impl StarCatalog {
    /// Load from Gaia-format CSV
    pub fn load_gaia_csv(path: &Path) -> Result<Self> {
        tracing::info!("Loading star catalog from {:?}", path);

        let mut stars = Vec::new();
        let mut reader = csv::Reader::from_path(path)?;

        for result in reader.deserialize() {
            let record: GaiaCsvRecord = result?;
            if let Some(star) = record.to_star_record() {
                stars.push(star);
            }
        }

        tracing::info!("Loaded {} valid stars", stars.len());
        Ok(Self { stars })
    }

    /// Load and merge multiple CSV files
    pub fn load_multiple(paths: &[&Path]) -> Result<Self> {
        let mut all_stars = Vec::new();
        for path in paths {
            let catalog = Self::load_gaia_csv(path)?;
            all_stars.extend(catalog.stars);
        }
        Ok(Self { stars: all_stars })
    }

    /// Create from existing records
    pub fn from_records(stars: Vec<StarRecord>) -> Self {
        Self { stars }
    }

    pub fn len(&self) -> usize { self.stars.len() }
    pub fn is_empty(&self) -> bool { self.stars.is_empty() }
    pub fn iter(&self) -> impl Iterator<Item = &StarRecord> { self.stars.iter() }

    /// Keep only stars brighter than threshold
    pub fn filter_by_magnitude(self, max_mag: f64) -> Self {
        Self {
            stars: self.stars.into_iter()
                .filter(|s| s.phot_g_mean_mag <= max_mag)
                .collect()
        }
    }

    /// Sort by magnitude (brightest first) and take top N
    pub fn take_brightest(mut self, n: usize) -> Self {
        self.stars.sort_by(|a, b|
            a.phot_g_mean_mag.partial_cmp(&b.phot_g_mean_mag).unwrap_or(std::cmp::Ordering::Equal)
        );
        self.stars.truncate(n);
        self
    }
}

/// CSV record matching Gaia DR3 export format
#[derive(Debug, Deserialize)]
struct GaiaCsvRecord {
    source_id: Option<u64>,
    ra: Option<f64>,
    dec: Option<f64>,
    parallax: Option<f64>,
    phot_g_mean_mag: Option<f64>,
    bp_rp: Option<f64>,
}

impl GaiaCsvRecord {
    fn to_star_record(&self) -> Option<StarRecord> {
        Some(StarRecord {
            ra: self.ra?,
            dec: self.dec?,
            parallax: self.parallax?,
            phot_g_mean_mag: self.phot_g_mean_mag?,
            bp_rp: self.bp_rp,
            source_id: self.source_id,
        })
    }
}

/// Generate synthetic stars for testing (deterministic)
pub fn generate_synthetic_stars(count: usize, seed: u64) -> Vec<StarRecord> {
    let mut rng = seed;
    let mut rand = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng >> 33) as f64 / (u32::MAX as f64)
    };

    (0..count).map(|i| {
        let ra = rand() * 360.0;
        let dec = (rand() * 2.0 - 1.0).asin().to_degrees();

        // Log-uniform distance: 10 to 10,000 pc
        let log_d = 1.0 + rand() * 3.0;
        let dist_pc = 10.0_f64.powf(log_d);
        let parallax = 1000.0 / dist_pc;

        // Random absolute mag -2 to +10, then apparent
        let abs_mag = -2.0 + rand() * 12.0;
        let dist_mod = 5.0 * dist_pc.log10() - 5.0;
        let phot_g_mean_mag = abs_mag + dist_mod;

        let bp_rp = Some(-0.5 + rand() * 4.5);

        StarRecord { ra, dec, parallax, phot_g_mean_mag, bp_rp, source_id: Some(i as u64) }
    }).collect()
}
