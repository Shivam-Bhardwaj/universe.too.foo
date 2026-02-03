//! Chebyshev polynomial propagation for spacecraft
//!
//! Used for high-fidelity ephemeris interpolation (e.g. JPL Horizons data)

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// A single segment of Chebyshev coefficients valid for a time range
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChebyshevSegment {
    /// Start epoch (Julian Centuries from J2000)
    pub start_jc: f64,
    /// End epoch (Julian Centuries from J2000)
    pub end_jc: f64,
    /// Coefficients for X component
    pub coeffs_x: Vec<f64>,
    /// Coefficients for Y component
    pub coeffs_y: Vec<f64>,
    /// Coefficients for Z component
    pub coeffs_z: Vec<f64>,
}

impl ChebyshevSegment {
    /// Check if this segment covers the given epoch
    pub fn contains(&self, jc: f64) -> bool {
        jc >= self.start_jc && jc <= self.end_jc
    }

    /// Evaluate position at given epoch
    pub fn position(&self, jc: f64) -> Vector3<f64> {
        // Map time to [-1, 1] domain
        // t = 2 * (tau - a) / (b - a) - 1
        let t_norm = 2.0 * (jc - self.start_jc) / (self.end_jc - self.start_jc) - 1.0;

        let x = evaluate_chebyshev(&self.coeffs_x, t_norm);
        let y = evaluate_chebyshev(&self.coeffs_y, t_norm);
        let z = evaluate_chebyshev(&self.coeffs_z, t_norm);

        Vector3::new(x, y, z)
    }
}

/// Evaluates Chebyshev polynomial series at normalized time t using Clenshaw recurrence
/// t must be in [-1, 1]
fn evaluate_chebyshev(coeffs: &[f64], t: f64) -> f64 {
    let n = coeffs.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return coeffs[0];
    }

    // Clenshaw recurrence
    // b_k = a_k + 2*x*b_{k+1} - b_{k+2}
    let mut b2 = 0.0; // b_{n+2}
    let mut b1 = 0.0; // b_{n+1}
    let x2 = 2.0 * t;

    // Iterate from n-1 down to 1
    for i in (1..n).rev() {
        let b0 = coeffs[i] + x2 * b1 - b2;
        b2 = b1;
        b1 = b0;
    }

    // Final step: f(x) = a_0 + x*b_1 - b_2
    coeffs[0] + t * b1 - b2
}

/// Propagator containing multiple segments
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChebyshevPropagator {
    pub segments: Vec<ChebyshevSegment>,
    pub name: String,
}

impl ChebyshevPropagator {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            segments: Vec::new(),
            name: name.into(),
        }
    }

    pub fn add_segment(&mut self, segment: ChebyshevSegment) {
        self.segments.push(segment);
        // Ensure sorted by start time for faster lookup
        self.segments.sort_by(|a, b| a.start_jc.partial_cmp(&b.start_jc).unwrap());
    }

    /// Evaluate position at epoch
    /// Returns None if epoch is outside all segments
    pub fn position(&self, jc: f64) -> Option<Vector3<f64>> {
        // Binary search or linear scan (typically few segments active)
        // For efficiency in animation, we might want to cache the last used index
        // but for now simple search is fine.
        
        // Find segment containing jc
        let segment = self.segments.iter().find(|s| s.contains(jc));
        
        match segment {
            Some(s) => Some(s.position(jc)),
            None => None, // Out of range
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chebyshev_eval() {
        // T_0(x) = 1
        // T_1(x) = x
        // T_2(x) = 2x^2 - 1
        
        // Test f(x) = 1*T_0 + 1*T_1 = 1 + x
        let coeffs = vec![1.0, 1.0];
        assert!((evaluate_chebyshev(&coeffs, 0.0) - 1.0).abs() < 1e-10); // 1 + 0 = 1
        assert!((evaluate_chebyshev(&coeffs, 1.0) - 2.0).abs() < 1e-10); // 1 + 1 = 2
        assert!((evaluate_chebyshev(&coeffs, -1.0) - 0.0).abs() < 1e-10); // 1 - 1 = 0

        // Test f(x) = T_2 = 2x^2 - 1
        let coeffs2 = vec![0.0, 0.0, 1.0];
        assert!((evaluate_chebyshev(&coeffs2, 0.0) + 1.0).abs() < 1e-10); // 0 - 1 = -1
        assert!((evaluate_chebyshev(&coeffs2, 1.0) - 1.0).abs() < 1e-10); // 2(1)^2 - 1 = 1
        assert!((evaluate_chebyshev(&coeffs2, 0.5) + 0.5).abs() < 1e-10); // 2(0.25) - 1 = -0.5
    }
}


