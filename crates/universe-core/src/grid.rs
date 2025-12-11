use crate::coordinates::*;
use crate::constants::*;
use serde::{Serialize, Deserialize};
use std::hash::{Hash, Hasher};

/// Configuration for the HLG
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HLGConfig {
    pub r_min: f64,
    pub log_base: f64,
    pub n_theta: u32,
    pub n_phi: u32,
}

impl Default for HLGConfig {
    fn default() -> Self {
        Self {
            r_min: R_MIN,
            log_base: LOG_BASE,
            n_theta: N_THETA,
            n_phi: N_PHI,
        }
    }
}

/// Unique cell identifier
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CellId {
    pub l: u32,      // Shell index
    pub theta: u32,  // Longitude index [0, N_THETA)
    pub phi: u32,    // Latitude index [0, N_PHI)
}

impl Hash for CellId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.l.hash(state);
        self.theta.hash(state);
        self.phi.hash(state);
    }
}

impl CellId {
    pub fn new(l: u32, theta: u32, phi: u32) -> Self {
        Self { l, theta, phi }
    }

    /// File name for this cell: "{l}_{theta}_{phi}.bin"
    pub fn file_name(&self) -> String {
        format!("{}_{}_{}.bin", self.l, self.theta, self.phi)
    }
}

/// Axis-Aligned Bounding Box in Cartesian space
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CellBounds {
    pub min: CartesianPosition,
    pub max: CartesianPosition,
    pub centroid: CartesianPosition,
}

/// The Heliocentric Logarithmic Grid
pub struct HLGGrid {
    config: HLGConfig,
    ln_base: f64,  // Precomputed ln(log_base)
    ln_r_min: f64, // Precomputed ln(r_min)
}

impl HLGGrid {
    pub fn new(config: HLGConfig) -> Self {
        let ln_base = config.log_base.ln();
        let ln_r_min = config.r_min.ln();
        Self { config, ln_base, ln_r_min }
    }

    pub fn with_defaults() -> Self {
        Self::new(HLGConfig::default())
    }

    pub fn config(&self) -> &HLGConfig {
        &self.config
    }

    /// Convert Cartesian position to Cell ID
    pub fn cartesian_to_cell(&self, pos: CartesianPosition) -> Option<CellId> {
        let r = pos.magnitude();

        // Below minimum radius — inside the Sun/Mercury orbit
        if r < self.config.r_min {
            return None;
        }

        // Shell index from logarithmic mapping
        let l_float = (r.ln() - self.ln_r_min) / self.ln_base;
        let l = l_float.floor() as u32;

        // Angular coordinates
        let spherical = pos.to_spherical();

        // Theta: [-π, π] → [0, N_THETA)
        let theta_norm = (spherical.theta + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);
        let theta = ((theta_norm * self.config.n_theta as f64).floor() as u32)
            .min(self.config.n_theta - 1);

        // Phi: [0, π] → [0, N_PHI)
        let phi_norm = spherical.phi / std::f64::consts::PI;
        let phi = ((phi_norm * self.config.n_phi as f64).floor() as u32)
            .min(self.config.n_phi - 1);

        Some(CellId { l, theta, phi })
    }

    /// Inner radius of shell L
    pub fn shell_inner_radius(&self, l: u32) -> f64 {
        self.config.r_min * self.config.log_base.powi(l as i32)
    }

    /// Outer radius of shell L
    pub fn shell_outer_radius(&self, l: u32) -> f64 {
        self.config.r_min * self.config.log_base.powi(l as i32 + 1)
    }

    /// Get the angular bounds for a cell
    pub fn cell_angular_bounds(&self, id: CellId) -> (f64, f64, f64, f64) {
        let theta_step = 2.0 * std::f64::consts::PI / self.config.n_theta as f64;
        let phi_step = std::f64::consts::PI / self.config.n_phi as f64;

        let theta_min = -std::f64::consts::PI + id.theta as f64 * theta_step;
        let theta_max = theta_min + theta_step;
        let phi_min = id.phi as f64 * phi_step;
        let phi_max = phi_min + phi_step;

        (theta_min, theta_max, phi_min, phi_max)
    }

    /// Compute AABB for a cell (conservative, encompasses the spherical cell)
    pub fn cell_to_bounds(&self, id: CellId) -> CellBounds {
        let r_inner = self.shell_inner_radius(id.l);
        let r_outer = self.shell_outer_radius(id.l);
        let (theta_min, theta_max, phi_min, phi_max) = self.cell_angular_bounds(id);

        // Sample corners of the spherical cell
        let corners = [
            (r_inner, theta_min, phi_min),
            (r_inner, theta_min, phi_max),
            (r_inner, theta_max, phi_min),
            (r_inner, theta_max, phi_max),
            (r_outer, theta_min, phi_min),
            (r_outer, theta_min, phi_max),
            (r_outer, theta_max, phi_min),
            (r_outer, theta_max, phi_max),
        ];

        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut min_z = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        let mut max_z = f64::MIN;

        for (r, theta, phi) in corners {
            let pos = SphericalPosition { r, theta, phi }.to_cartesian();
            min_x = min_x.min(pos.x);
            min_y = min_y.min(pos.y);
            min_z = min_z.min(pos.z);
            max_x = max_x.max(pos.x);
            max_y = max_y.max(pos.y);
            max_z = max_z.max(pos.z);
        }

        CellBounds {
            min: CartesianPosition::new(min_x, min_y, min_z),
            max: CartesianPosition::new(max_x, max_y, max_z),
            centroid: CartesianPosition::new(
                (min_x + max_x) / 2.0,
                (min_y + max_y) / 2.0,
                (min_z + max_z) / 2.0,
            ),
        }
    }

    /// Maximum shell index for a given outer distance
    pub fn max_shell_for_distance(&self, distance: f64) -> u32 {
        if distance <= self.config.r_min {
            return 0;
        }
        ((distance.ln() - self.ln_r_min) / self.ln_base).floor() as u32
    }
}
