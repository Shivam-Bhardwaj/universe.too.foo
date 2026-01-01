use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Landmark kind (matches TypeScript enum)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum LandmarkKind {
    Star,
    Planet,
    DwarfPlanet,
    Galaxy,
    Cluster,
    Nebula,
    Region,
    Spacecraft,
    Other,
}

/// Position in ecliptic coordinates (meters)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Landmark definition (matches TypeScript interface)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Landmark {
    pub id: String,
    pub name: String,
    pub kind: LandmarkKind,
    pub pos_meters: Position,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub radius_hint: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl Landmark {
    /// Get the magnitude (distance from origin) of this landmark
    pub fn distance(&self) -> f64 {
        (self.pos_meters.x.powi(2) + self.pos_meters.y.powi(2) + self.pos_meters.z.powi(2)).sqrt()
    }

    /// Get visual radius for rendering (uses radius_hint if available, otherwise kind-based default)
    pub fn visual_radius(&self) -> f64 {
        if let Some(r) = self.radius_hint {
            // Clamp to reasonable range (1 km to 1 Gpc)
            r.clamp(1e3, 1e27)
        } else {
            self.kind_default_radius()
        }
    }

    /// Get default visual radius based on kind
    fn kind_default_radius(&self) -> f64 {
        match self.kind {
            LandmarkKind::Star => 1e9,           // ~1 solar radius
            LandmarkKind::Planet => 1e7,         // ~Earth-sized
            LandmarkKind::DwarfPlanet => 1e6,    // ~1000 km
            LandmarkKind::Galaxy => 3e21,        // ~100,000 ly diameter
            LandmarkKind::Cluster => 1e17,       // ~30 ly
            LandmarkKind::Nebula => 5e16,        // ~15 ly
            LandmarkKind::Region => 1e18,        // ~300 ly
            LandmarkKind::Spacecraft => 1e4,     // 10 km (scaled for visibility)
            LandmarkKind::Other => 1e9,
        }
    }

    /// Get color and opacity for rendering based on kind
    pub fn visual_appearance(&self) -> ([f32; 3], f32) {
        match self.kind {
            LandmarkKind::Star => ([1.0, 0.95, 0.85], 1.0),              // Yellowish-white, opaque
            LandmarkKind::Planet => ([0.7, 0.7, 0.8], 1.0),              // Grayish-blue, opaque
            LandmarkKind::DwarfPlanet => ([0.6, 0.6, 0.65], 0.9),        // Gray, slightly transparent
            LandmarkKind::Galaxy => ([0.9, 0.85, 0.95], 0.3),            // Pale purple, transparent
            LandmarkKind::Cluster => ([0.95, 0.95, 1.0], 0.7),           // White-blue, semi-transparent
            LandmarkKind::Nebula => ([1.0, 0.4, 0.6], 0.5),              // Pinkish (emission), semi-transparent
            LandmarkKind::Region => ([0.8, 0.8, 0.9], 0.2),              // Faint blue, very transparent
            LandmarkKind::Spacecraft => ([1.0, 1.0, 0.9], 1.0),          // Bright white-yellow, opaque
            LandmarkKind::Other => ([0.8, 0.8, 0.8], 0.6),
        }
    }
}

/// Load landmarks from a JSON file
pub fn load_landmarks_json(path: &Path) -> Result<Vec<Landmark>> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("Failed to read landmarks file: {}", path.display()))?;

    let landmarks: Vec<Landmark> = serde_json::from_str(&contents)
        .with_context(|| format!("Failed to parse landmarks JSON: {}", path.display()))?;

    Ok(landmarks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_landmark_distance() {
        let landmark = Landmark {
            id: "test".to_string(),
            name: "Test".to_string(),
            kind: LandmarkKind::Star,
            pos_meters: Position { x: 3.0, y: 4.0, z: 0.0 },
            radius_hint: None,
            description: None,
        };
        assert_eq!(landmark.distance(), 5.0);
    }

    #[test]
    fn test_visual_radius_with_hint() {
        let landmark = Landmark {
            id: "test".to_string(),
            name: "Test".to_string(),
            kind: LandmarkKind::Galaxy,
            pos_meters: Position { x: 0.0, y: 0.0, z: 1e20 },
            radius_hint: Some(2e21),
            description: None,
        };
        assert_eq!(landmark.visual_radius(), 2e21);
    }

    #[test]
    fn test_visual_radius_default() {
        let landmark = Landmark {
            id: "test".to_string(),
            name: "Test".to_string(),
            kind: LandmarkKind::Nebula,
            pos_meters: Position { x: 0.0, y: 0.0, z: 1e17 },
            radius_hint: None,
            description: None,
        };
        assert_eq!(landmark.visual_radius(), 5e16);
    }

    #[test]
    fn test_visual_appearance() {
        let landmark = Landmark {
            id: "test".to_string(),
            name: "Test".to_string(),
            kind: LandmarkKind::Galaxy,
            pos_meters: Position { x: 0.0, y: 0.0, z: 1e20 },
            radius_hint: None,
            description: None,
        };
        let (color, opacity) = landmark.visual_appearance();
        assert_eq!(color, [0.9, 0.85, 0.95]);
        assert_eq!(opacity, 0.3);
    }
}
