//! Validation against ephemeris data

use crate::system::SolarSystem;
use crate::planets::Body;
use universe_data::EphemerisProvider;
use universe_core::constants::AU;
use hifitime::{Epoch, Duration};
use anyhow::Result;

/// Validation result for a single body at one epoch
#[derive(Debug)]
pub struct ValidationPoint {
    pub body: Body,
    pub epoch: Epoch,
    pub computed_pos: (f64, f64, f64),  // AU
    pub ephemeris_pos: (f64, f64, f64), // AU
    pub error_km: f64,
    pub error_percent: f64,
}

/// Validate computed positions against ephemeris
pub fn validate_body(
    body: Body,
    system: &mut SolarSystem,
    ephemeris: &EphemerisProvider,
    epoch: Epoch,
) -> Result<ValidationPoint> {
    system.set_epoch(epoch);

    let computed = system.body_position(body);

    // Convert Body to SolarSystemBody for ephemeris query
    let eph_body = match body {
        Body::Sun => universe_data::SolarSystemBody::Sun,
        Body::Mercury => universe_data::SolarSystemBody::Mercury,
        Body::Venus => universe_data::SolarSystemBody::Venus,
        Body::Earth => universe_data::SolarSystemBody::Earth,
        Body::Moon => universe_data::SolarSystemBody::Moon,
        Body::Mars => universe_data::SolarSystemBody::Mars,
        Body::Jupiter => universe_data::SolarSystemBody::Jupiter,
        Body::Saturn => universe_data::SolarSystemBody::Saturn,
        Body::Uranus => universe_data::SolarSystemBody::Uranus,
        Body::Neptune => universe_data::SolarSystemBody::Neptune,
        Body::Pluto => universe_data::SolarSystemBody::Pluto,
    };

    let eph_pos = ephemeris.body_position(eph_body, epoch)?;

    // Compute error
    let dx = computed.x - eph_pos.x;
    let dy = computed.y - eph_pos.y;
    let dz = computed.z - eph_pos.z;
    let error_m = (dx*dx + dy*dy + dz*dz).sqrt();
    let error_km = error_m / 1000.0;

    let eph_dist = eph_pos.magnitude();
    let error_percent = if eph_dist > 0.0 {
        (error_m / eph_dist) * 100.0
    } else {
        0.0
    };

    Ok(ValidationPoint {
        body,
        epoch,
        computed_pos: (computed.x / AU, computed.y / AU, computed.z / AU),
        ephemeris_pos: (eph_pos.x / AU, eph_pos.y / AU, eph_pos.z / AU),
        error_km,
        error_percent,
    })
}

/// Validate all planets over a time range
pub fn validate_range(
    system: &mut SolarSystem,
    ephemeris: &EphemerisProvider,
    start: Epoch,
    end: Epoch,
    step: Duration,
) -> Result<Vec<ValidationPoint>> {
    let mut results = Vec::new();
    let mut epoch = start;

    while epoch <= end {
        for body in Body::planets() {
            if let Ok(point) = validate_body(*body, system, ephemeris, epoch) {
                results.push(point);
            }
        }
        epoch = epoch + step;
    }

    Ok(results)
}

/// Summary statistics for validation
#[derive(Debug)]
pub struct ValidationSummary {
    pub body: Body,
    pub num_points: usize,
    pub mean_error_km: f64,
    pub max_error_km: f64,
    pub mean_error_percent: f64,
}

/// Compute summary statistics per body
pub fn summarize_validation(results: &[ValidationPoint]) -> Vec<ValidationSummary> {
    use std::collections::HashMap;

    let mut by_body: HashMap<Body, Vec<&ValidationPoint>> = HashMap::new();

    for point in results {
        by_body.entry(point.body).or_default().push(point);
    }

    by_body.into_iter().map(|(body, points)| {
        let n = points.len();
        let mean_km = points.iter().map(|p| p.error_km).sum::<f64>() / n as f64;
        let max_km = points.iter().map(|p| p.error_km).fold(0.0, f64::max);
        let mean_pct = points.iter().map(|p| p.error_percent).sum::<f64>() / n as f64;

        ValidationSummary {
            body,
            num_points: n,
            mean_error_km: mean_km,
            max_error_km: max_km,
            mean_error_percent: mean_pct,
        }
    }).collect()
}
