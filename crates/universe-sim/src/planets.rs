//! Planetary orbital elements and physical properties
//!
//! Data from JPL Horizons and NASA fact sheets
//! Reference epoch: J2000.0 (2000-01-01T12:00:00 TDB)

use crate::elements::{OrbitalElements, SecularRates};
use std::f64::consts::PI;

/// Gravitational parameter of the Sun (m³/s²)
pub const MU_SUN: f64 = 1.32712440018e20;

/// Julian centuries per day
pub const JC_PER_DAY: f64 = 1.0 / 36525.0;

/// Body identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Body {
    Sun,
    Mercury,
    Venus,
    Earth,
    Moon,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
    Pluto,
}

impl Body {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Sun => "Sun",
            Self::Mercury => "Mercury",
            Self::Venus => "Venus",
            Self::Earth => "Earth",
            Self::Moon => "Moon",
            Self::Mars => "Mars",
            Self::Jupiter => "Jupiter",
            Self::Saturn => "Saturn",
            Self::Uranus => "Uranus",
            Self::Neptune => "Neptune",
            Self::Pluto => "Pluto",
        }
    }

    /// Mean radius in meters
    pub fn radius(&self) -> f64 {
        match self {
            Self::Sun => 6.9634e8,
            Self::Mercury => 2.4397e6,
            Self::Venus => 6.0518e6,
            Self::Earth => 6.371e6,
            Self::Moon => 1.7374e6,
            Self::Mars => 3.3895e6,
            Self::Jupiter => 6.9911e7,
            Self::Saturn => 5.8232e7,
            Self::Uranus => 2.5362e7,
            Self::Neptune => 2.4622e7,
            Self::Pluto => 1.188e6,
        }
    }

    /// Orbital elements at J2000
    /// Returns None for Sun (it's the center)
    pub fn j2000_elements(&self) -> Option<OrbitalElements> {
        let deg = PI / 180.0;

        match self {
            Self::Sun => None,

            Self::Mercury => Some(OrbitalElements::new(
                57.909e9,           // a (m)
                0.20563,            // e
                7.005 * deg,        // i
                48.331 * deg,       // Ω
                29.124 * deg,       // ω
                174.796 * deg,      // M₀
                0.0,                // epoch (J2000)
                MU_SUN,
            )),

            Self::Venus => Some(OrbitalElements::new(
                108.21e9,
                0.00677,
                3.3946 * deg,
                76.680 * deg,
                54.884 * deg,
                50.115 * deg,
                0.0,
                MU_SUN,
            )),

            Self::Earth => Some(OrbitalElements::new(
                149.598e9,
                0.01671,
                0.00005 * deg,      // ~0 for ecliptic reference
                -11.26064 * deg,    // Undefined for i≈0, use vernal equinox
                102.94719 * deg,    // Longitude of perihelion
                100.46435 * deg,    // Mean longitude
                0.0,
                MU_SUN,
            )),

            Self::Moon => {
                // Moon orbits Earth - handled specially
                // These are approximate osculating elements
                Some(OrbitalElements::new(
                    384400e3,           // a (m) - from Earth
                    0.0549,             // e
                    5.145 * deg,        // i (to ecliptic)
                    125.08 * deg,       // Ω (regresses)
                    318.15 * deg,       // ω
                    135.27 * deg,       // M₀
                    0.0,
                    3.986004418e14,     // μ_Earth
                ))
            }

            Self::Mars => Some(OrbitalElements::new(
                227.956e9,
                0.0934,
                1.850 * deg,
                49.558 * deg,
                286.502 * deg,
                19.373 * deg,
                0.0,
                MU_SUN,
            )),

            Self::Jupiter => Some(OrbitalElements::new(
                778.479e9,
                0.0489,
                1.303 * deg,
                100.464 * deg,
                273.867 * deg,
                20.020 * deg,
                0.0,
                MU_SUN,
            )),

            Self::Saturn => Some(OrbitalElements::new(
                1432.041e9,
                0.0565,
                2.485 * deg,
                113.665 * deg,
                339.392 * deg,
                317.020 * deg,
                0.0,
                MU_SUN,
            )),

            Self::Uranus => Some(OrbitalElements::new(
                2867.043e9,
                0.0457,
                0.773 * deg,
                74.006 * deg,
                96.998857 * deg,
                142.2386 * deg,
                0.0,
                MU_SUN,
            )),

            Self::Neptune => Some(OrbitalElements::new(
                4514.953e9,
                0.0113,
                1.770 * deg,
                131.784 * deg,
                273.187 * deg,
                256.228 * deg,
                0.0,
                MU_SUN,
            )),

            Self::Pluto => Some(OrbitalElements::new(
                5869.656e9,
                0.2488,
                17.16 * deg,
                110.299 * deg,
                113.834 * deg,
                14.53 * deg,
                0.0,
                MU_SUN,
            )),
        }
    }

    /// Secular perturbation rates (per Julian century)
    /// Data from JPL for long-term accuracy
    pub fn secular_rates(&self) -> SecularRates {
        let deg = PI / 180.0;

        match self {
            Self::Sun => SecularRates::default(),

            Self::Mercury => SecularRates {
                da: 0.0,
                de: 0.00002123,
                di: -0.00590 * deg,
                d_omega_big: -0.12534 * deg,
                d_omega_small: 0.16047 * deg,
            },

            Self::Venus => SecularRates {
                da: 0.0,
                de: -0.00004938,
                di: -0.00078 * deg,
                d_omega_big: -0.27769 * deg,
                d_omega_small: 0.00268 * deg,
            },

            Self::Earth => SecularRates {
                da: 0.0,
                de: -0.00004392,
                di: -0.01337 * deg,
                d_omega_big: -0.18047 * deg,  // Precession
                d_omega_small: 0.32327 * deg,
            },

            Self::Moon => SecularRates {
                da: 3.8e-2,  // ~3.8 cm/year recession
                de: 0.0,
                di: 0.0,
                d_omega_big: -0.05295 * deg * 36525.0 / 18.6, // ~18.6 year nodal cycle
                d_omega_small: 0.11140 * deg * 36525.0 / 8.85, // ~8.85 year apsidal cycle
            },

            Self::Mars => SecularRates {
                da: 0.0,
                de: 0.00007882,
                di: -0.00813 * deg,
                d_omega_big: -0.29257 * deg,
                d_omega_small: 0.44106 * deg,
            },

            Self::Jupiter => SecularRates {
                da: 0.0,
                de: -0.00012880,
                di: -0.00242 * deg,
                d_omega_big: 0.18966 * deg,
                d_omega_small: 0.17693 * deg,
            },

            Self::Saturn => SecularRates {
                da: 0.0,
                de: -0.00050991,
                di: 0.00193 * deg,
                d_omega_big: -0.26731 * deg,
                d_omega_small: -0.42568 * deg,
            },

            Self::Uranus => SecularRates {
                da: 0.0,
                de: -0.00020455,
                di: 0.00041 * deg,
                d_omega_big: 0.01140 * deg,
                d_omega_small: 0.02768 * deg,
            },

            Self::Neptune => SecularRates {
                da: 0.0,
                de: 0.00006171,
                di: -0.00333 * deg,
                d_omega_big: -0.01022 * deg,
                d_omega_small: -0.01043 * deg,
            },

            Self::Pluto => SecularRates {
                da: 0.0,
                de: 0.0,  // Poorly constrained
                di: 0.0,
                d_omega_big: 0.0,
                d_omega_small: 0.0,
            },
        }
    }

    /// All bodies including Sun
    pub fn all() -> &'static [Body] {
        &[
            Self::Sun, Self::Mercury, Self::Venus, Self::Earth, Self::Moon,
            Self::Mars, Self::Jupiter, Self::Saturn, Self::Uranus,
            Self::Neptune, Self::Pluto,
        ]
    }

    /// Planets only (heliocentric, no Sun/Moon)
    pub fn planets() -> &'static [Body] {
        &[
            Self::Mercury, Self::Venus, Self::Earth, Self::Mars,
            Self::Jupiter, Self::Saturn, Self::Uranus, Self::Neptune, Self::Pluto,
        ]
    }
}
