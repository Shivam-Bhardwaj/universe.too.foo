//! Solar system state management

use crate::elements::OrbitalElements;
use crate::planets::Body;
use universe_core::coordinates::CartesianPosition;
use hifitime::Epoch;
use nalgebra::Vector3;
use std::collections::HashMap;

/// Convert hifitime Epoch to Julian centuries from J2000
pub fn epoch_to_jc(epoch: Epoch) -> f64 {
    // J2000.0 = 2000-01-01T12:00:00 TDB
    // Julian Date of J2000.0 = 2451545.0
    let jd = epoch.to_jde_utc_days();
    (jd - 2451545.0) / 36525.0
}

/// Convert Julian centuries to hifitime Epoch
pub fn jc_to_epoch(jc: f64) -> Epoch {
    let jd = jc * 36525.0 + 2451545.0;
    Epoch::from_jde_utc(jd)
}

/// Body state at a specific time
#[derive(Clone, Debug)]
pub struct BodyState {
    pub position: CartesianPosition,
    pub velocity: Vector3<f64>,
}

/// Solar system snapshot at a specific epoch
#[derive(Clone, Debug)]
pub struct SystemSnapshot {
    pub epoch: Epoch,
    pub states: HashMap<Body, BodyState>,
}

/// Solar system model with propagation
pub struct SolarSystem {
    /// Current epoch
    current_epoch: Epoch,
    /// Cached propagated elements
    elements_cache: HashMap<Body, OrbitalElements>,
    /// Earth position (for Moon computation)
    earth_position: Vector3<f64>,
}

impl SolarSystem {
    /// Create at J2000 epoch
    pub fn new() -> Self {
        let j2000 = Epoch::from_gregorian_utc(2000, 1, 1, 12, 0, 0, 0);
        Self::at_epoch(j2000)
    }

    /// Create at specific epoch
    pub fn at_epoch(epoch: Epoch) -> Self {
        let mut system = Self {
            current_epoch: epoch,
            elements_cache: HashMap::new(),
            earth_position: Vector3::zeros(),
        };
        system.update_elements();
        system
    }

    /// Set current time
    pub fn set_epoch(&mut self, epoch: Epoch) {
        self.current_epoch = epoch;
        self.update_elements();
    }

    /// Get current epoch
    pub fn epoch(&self) -> Epoch {
        self.current_epoch
    }

    /// Update cached elements for current epoch
    fn update_elements(&mut self) {
        let jc = epoch_to_jc(self.current_epoch);

        for body in Body::planets() {
            if let Some(base_elements) = body.j2000_elements() {
                let rates = body.secular_rates();
                let propagated = base_elements.propagate(jc, &rates);
                self.elements_cache.insert(*body, propagated);
            }
        }

        // Compute Earth position for Moon
        if let Some(earth_elem) = self.elements_cache.get(&Body::Earth) {
            self.earth_position = earth_elem.position_ecliptic(jc);
        }
    }

    /// Get body position (heliocentric, meters)
    pub fn body_position(&self, body: Body) -> CartesianPosition {
        let jc = epoch_to_jc(self.current_epoch);

        match body {
            Body::Sun => CartesianPosition::new(0.0, 0.0, 0.0),

            Body::Moon => {
                // Moon position = Earth position + Moon's geocentric offset
                if let Some(moon_elem) = Body::Moon.j2000_elements() {
                    let rates = Body::Moon.secular_rates();
                    let propagated = moon_elem.propagate(jc, &rates);
                    let moon_rel = propagated.position_ecliptic(jc);
                    let pos = self.earth_position + moon_rel;
                    CartesianPosition::new(pos.x, pos.y, pos.z)
                } else {
                    // Fallback to Earth position
                    CartesianPosition::new(
                        self.earth_position.x,
                        self.earth_position.y,
                        self.earth_position.z,
                    )
                }
            }

            _ => {
                if let Some(elements) = self.elements_cache.get(&body) {
                    let pos = elements.position_ecliptic(jc);
                    CartesianPosition::new(pos.x, pos.y, pos.z)
                } else {
                    CartesianPosition::new(0.0, 0.0, 0.0)
                }
            }
        }
    }

    /// Get body state (position + velocity)
    pub fn body_state(&self, body: Body) -> BodyState {
        let jc = epoch_to_jc(self.current_epoch);
        let position = self.body_position(body);

        let velocity = match body {
            Body::Sun => Vector3::zeros(),
            Body::Moon => {
                // Approximate Moon velocity
                if let Some(moon_elem) = Body::Moon.j2000_elements() {
                    let rates = Body::Moon.secular_rates();
                    let propagated = moon_elem.propagate(jc, &rates);
                    let v_rel = propagated.velocity_ecliptic(jc);
                    // Add Earth's velocity
                    if let Some(earth_elem) = self.elements_cache.get(&Body::Earth) {
                        let v_earth = earth_elem.velocity_ecliptic(jc);
                        v_earth + v_rel
                    } else {
                        v_rel
                    }
                } else {
                    Vector3::zeros()
                }
            }
            _ => {
                if let Some(elements) = self.elements_cache.get(&body) {
                    elements.velocity_ecliptic(jc)
                } else {
                    Vector3::zeros()
                }
            }
        };

        BodyState { position, velocity }
    }

    /// Get snapshot of entire system
    pub fn snapshot(&self) -> SystemSnapshot {
        let mut states = HashMap::new();

        for body in Body::all() {
            states.insert(*body, self.body_state(*body));
        }

        SystemSnapshot {
            epoch: self.current_epoch,
            states,
        }
    }
}

impl Default for SolarSystem {
    fn default() -> Self {
        Self::new()
    }
}
