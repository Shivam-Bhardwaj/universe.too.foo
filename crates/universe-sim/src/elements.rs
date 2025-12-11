//! Keplerian orbital elements and conversions

use nalgebra::{Vector3, Matrix3};
use std::f64::consts::PI;

/// Classical Keplerian orbital elements
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OrbitalElements {
    /// Semi-major axis (meters)
    pub a: f64,
    /// Eccentricity (dimensionless, 0 = circular)
    pub e: f64,
    /// Inclination (radians)
    pub i: f64,
    /// Longitude of ascending node / RAAN (radians)
    pub omega_big: f64,
    /// Argument of perihelion (radians)
    pub omega_small: f64,
    /// Mean anomaly at epoch (radians)
    pub m0: f64,
    /// Reference epoch (Julian centuries from J2000)
    pub epoch_jc: f64,
    /// Gravitational parameter μ = G(M_sun + M_body) (m³/s²)
    pub mu: f64,
}

impl OrbitalElements {
    /// Create from parameters
    pub fn new(
        a: f64, e: f64, i: f64,
        omega_big: f64, omega_small: f64,
        m0: f64, epoch_jc: f64, mu: f64,
    ) -> Self {
        Self { a, e, i, omega_big, omega_small, m0, epoch_jc, mu }
    }

    /// Mean motion (radians per second)
    pub fn mean_motion(&self) -> f64 {
        (self.mu / self.a.powi(3)).sqrt()
    }

    /// Orbital period (seconds)
    pub fn period(&self) -> f64 {
        2.0 * PI / self.mean_motion()
    }

    /// Mean anomaly at given Julian centuries from J2000
    pub fn mean_anomaly_at(&self, jc: f64) -> f64 {
        let dt_centuries = jc - self.epoch_jc;
        let dt_seconds = dt_centuries * 36525.0 * 86400.0;
        let m = self.m0 + self.mean_motion() * dt_seconds;
        normalize_angle(m)
    }

    /// Solve Kepler's equation: M = E - e*sin(E)
    /// Returns eccentric anomaly E
    pub fn eccentric_anomaly(&self, mean_anomaly: f64) -> f64 {
        let m = normalize_angle(mean_anomaly);
        let e = self.e;

        // Newton-Raphson iteration
        let mut ea = if e < 0.8 { m } else { PI };

        for _ in 0..50 {
            let f = ea - e * ea.sin() - m;
            let fp = 1.0 - e * ea.cos();
            let delta = f / fp;
            ea -= delta;

            if delta.abs() < 1e-12 {
                break;
            }
        }

        ea
    }

    /// True anomaly from eccentric anomaly
    pub fn true_anomaly(&self, eccentric_anomaly: f64) -> f64 {
        let ea = eccentric_anomaly;
        let e = self.e;

        // tan(ν/2) = sqrt((1+e)/(1-e)) * tan(E/2)
        let half_nu = ((1.0 + e) / (1.0 - e)).sqrt() * (ea / 2.0).tan();
        2.0 * half_nu.atan()
    }

    /// Distance from focus (Sun) at given true anomaly
    pub fn radius(&self, true_anomaly: f64) -> f64 {
        self.a * (1.0 - self.e.powi(2)) / (1.0 + self.e * true_anomaly.cos())
    }

    /// Position in orbital plane (perifocal frame)
    /// Returns (x, y) where x points to perihelion
    pub fn position_perifocal(&self, true_anomaly: f64) -> (f64, f64) {
        let r = self.radius(true_anomaly);
        let x = r * true_anomaly.cos();
        let y = r * true_anomaly.sin();
        (x, y)
    }

    /// Velocity in orbital plane (perifocal frame)
    pub fn velocity_perifocal(&self, true_anomaly: f64) -> (f64, f64) {
        let p = self.a * (1.0 - self.e.powi(2)); // Semi-latus rectum
        let h = (self.mu * p).sqrt(); // Specific angular momentum

        let vx = -self.mu / h * true_anomaly.sin();
        let vy = self.mu / h * (self.e + true_anomaly.cos());
        (vx, vy)
    }

    /// Rotation matrix from perifocal to heliocentric ecliptic J2000
    pub fn perifocal_to_ecliptic(&self) -> Matrix3<f64> {
        let cos_o = self.omega_big.cos();
        let sin_o = self.omega_big.sin();
        let cos_i = self.i.cos();
        let sin_i = self.i.sin();
        let cos_w = self.omega_small.cos();
        let sin_w = self.omega_small.sin();

        // Combined rotation: R_z(-Ω) * R_x(-i) * R_z(-ω)
        Matrix3::new(
            cos_o * cos_w - sin_o * sin_w * cos_i,
            -cos_o * sin_w - sin_o * cos_w * cos_i,
            sin_o * sin_i,

            sin_o * cos_w + cos_o * sin_w * cos_i,
            -sin_o * sin_w + cos_o * cos_w * cos_i,
            -cos_o * sin_i,

            sin_w * sin_i,
            cos_w * sin_i,
            cos_i,
        )
    }

    /// Position in heliocentric ecliptic J2000 frame (meters)
    pub fn position_ecliptic(&self, jc: f64) -> Vector3<f64> {
        let m = self.mean_anomaly_at(jc);
        let ea = self.eccentric_anomaly(m);
        let nu = self.true_anomaly(ea);
        let (x_pf, y_pf) = self.position_perifocal(nu);

        let pf = Vector3::new(x_pf, y_pf, 0.0);
        let rot = self.perifocal_to_ecliptic();

        rot * pf
    }

    /// Velocity in heliocentric ecliptic J2000 frame (m/s)
    pub fn velocity_ecliptic(&self, jc: f64) -> Vector3<f64> {
        let m = self.mean_anomaly_at(jc);
        let ea = self.eccentric_anomaly(m);
        let nu = self.true_anomaly(ea);
        let (vx_pf, vy_pf) = self.velocity_perifocal(nu);

        let vf = Vector3::new(vx_pf, vy_pf, 0.0);
        let rot = self.perifocal_to_ecliptic();

        rot * vf
    }
}

/// Normalize angle to [0, 2π)
fn normalize_angle(angle: f64) -> f64 {
    let mut a = angle % (2.0 * PI);
    if a < 0.0 { a += 2.0 * PI; }
    a
}

/// Secular perturbation rates (per Julian century)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SecularRates {
    /// Semi-major axis rate (m/century) - usually ~0
    pub da: f64,
    /// Eccentricity rate (1/century)
    pub de: f64,
    /// Inclination rate (rad/century)
    pub di: f64,
    /// RAAN precession rate (rad/century)
    pub d_omega_big: f64,
    /// Argument of perihelion rate (rad/century)
    pub d_omega_small: f64,
}

impl Default for SecularRates {
    fn default() -> Self {
        Self { da: 0.0, de: 0.0, di: 0.0, d_omega_big: 0.0, d_omega_small: 0.0 }
    }
}

impl OrbitalElements {
    /// Propagate elements with secular perturbations
    pub fn propagate(&self, target_jc: f64, rates: &SecularRates) -> OrbitalElements {
        let dt = target_jc - self.epoch_jc;

        OrbitalElements {
            a: self.a + rates.da * dt,
            e: (self.e + rates.de * dt).clamp(0.0, 0.99),
            i: self.i + rates.di * dt,
            omega_big: normalize_angle(self.omega_big + rates.d_omega_big * dt),
            omega_small: normalize_angle(self.omega_small + rates.d_omega_small * dt),
            m0: self.m0, // Mean anomaly handled separately
            epoch_jc: self.epoch_jc,
            mu: self.mu,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kepler_circular() {
        // Circular orbit: e=0 means E=M=ν
        let elements = OrbitalElements::new(
            1.496e11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.327e20, // μ for Sun
        );

        let m = PI / 4.0;
        let ea = elements.eccentric_anomaly(m);
        let nu = elements.true_anomaly(ea);

        assert!((ea - m).abs() < 1e-10);
        assert!((nu - m).abs() < 1e-10);
    }

    #[test]
    fn test_kepler_eccentric() {
        // Earth-like orbit
        let elements = OrbitalElements::new(
            1.496e11, 0.0167, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.327e20,
        );

        // At perihelion (M=0), E=0, ν=0
        let ea = elements.eccentric_anomaly(0.0);
        assert!(ea.abs() < 1e-10);

        // At aphelion (M=π), E=π, ν=π
        let ea = elements.eccentric_anomaly(PI);
        assert!((ea - PI).abs() < 1e-10);
    }
}
