use serde::{Serialize, Deserialize};
use crate::constants::AU;

/// High-precision Cartesian position (f64 for computation)
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CartesianPosition {
    pub x: f64,  // meters
    pub y: f64,
    pub z: f64,
}

impl CartesianPosition {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn from_au(x: f64, y: f64, z: f64) -> Self {
        Self::new(x * AU, y * AU, z * AU)
    }

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn to_spherical(&self) -> SphericalPosition {
        let r = self.magnitude();
        let theta = self.y.atan2(self.x);
        let phi = if r > 0.0 { (self.z / r).acos() } else { 0.0 };
        SphericalPosition { r, theta, phi }
    }
}

/// Spherical coordinates (r in meters, angles in radians)
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SphericalPosition {
    pub r: f64,      // radial distance [0, ∞)
    pub theta: f64,  // azimuth [-π, π]
    pub phi: f64,    // polar [0, π]
}

impl SphericalPosition {
    pub fn to_cartesian(&self) -> CartesianPosition {
        let sin_phi = self.phi.sin();
        CartesianPosition {
            x: self.r * sin_phi * self.theta.cos(),
            y: self.r * sin_phi * self.theta.sin(),
            z: self.r * self.phi.cos(),
        }
    }
}

/// GPU-friendly position (f32, relative to cell centroid)
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct LocalPosition {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
