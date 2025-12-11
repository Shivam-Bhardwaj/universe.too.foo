//! Input event protocol for remote control

use serde::{Deserialize, Serialize};
use glam::DVec3;

/// Input event from client
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InputEvent {
    /// Mouse movement (delta)
    MouseMove { dx: f32, dy: f32 },

    /// Mouse scroll (for zoom)
    Scroll { delta: f32 },

    /// Mouse button
    MouseButton { button: u8, pressed: bool },

    /// Key press/release
    Key { code: String, pressed: bool },

    /// Set absolute time (Julian Date)
    SetTime { jd: f64 },

    /// Set time rate (years per second)
    SetTimeRate { rate: f64 },

    /// Teleport camera to position
    Teleport { x: f64, y: f64, z: f64 },

    /// Look at target
    LookAt { x: f64, y: f64, z: f64 },

    /// Ping for latency measurement
    Ping { client_time: u64 },
}

/// Output event to client
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputEvent {
    /// Pong response
    Pong { client_time: u64, server_time: u64 },

    /// Current simulation state
    State {
        epoch_jd: f64,
        time_rate: f64,
        camera_x: f64,
        camera_y: f64,
        camera_z: f64,
        fps: f32,
        /// Number of connected viewers (best-effort)
        clients: u32,
    },

    /// Jump budget status for this connection
    JumpStatus {
        remaining: u32,
        max: u32,
        registered: bool,
    },

    /// Error message
    Error { message: String },
}

/// Parse input event from JSON
pub fn parse_input(json: &str) -> Result<InputEvent, serde_json::Error> {
    serde_json::from_str(json)
}

/// Serialize output event to JSON
pub fn serialize_output(event: &OutputEvent) -> String {
    serde_json::to_string(event).unwrap_or_else(|_| "{}".to_string())
}

/// Apply input event to renderer state
pub fn apply_input(
    event: InputEvent,
    camera: &mut universe_render::Camera,
    time_controller: &mut universe_sim::TimeController,
) -> Option<OutputEvent> {
    match event {
        InputEvent::MouseMove { dx, dy } => {
            camera.rotate(dx, dy);
            None
        }

        InputEvent::MouseButton { .. } => {
            None
        }

        InputEvent::Scroll { delta } => {
            // Exponential zoom - multiply distance by factor
            let factor = if delta > 0.0 { 0.85 } else { 1.18 };
            let dir = camera.forward();
            let dist = camera.position.length();
            camera.position = camera.position + dir * dist * (1.0 - factor);
            None
        }

        InputEvent::Key { code, pressed } => {
            if pressed {
                let dt = 1.0 / 60.0;
                match code.as_str() {
                    "KeyW" => camera.translate(1.0, 0.0, 0.0, dt),
                    "KeyS" => camera.translate(-1.0, 0.0, 0.0, dt),
                    "KeyA" => camera.translate(0.0, -1.0, 0.0, dt),
                    "KeyD" => camera.translate(0.0, 1.0, 0.0, dt),
                    "Space" => camera.translate(0.0, 0.0, 1.0, dt),
                    "ShiftLeft" => camera.translate(0.0, 0.0, -1.0, dt),
                    "KeyP" => time_controller.toggle_pause(),
                    "Comma" => time_controller.set_rate(time_controller.rate() * 0.5),
                    "Period" => time_controller.set_rate(time_controller.rate() * 2.0),
                    _ => {}
                }
            }
            None
        }

        InputEvent::SetTime { jd } => {
            let epoch = universe_sim::jc_to_epoch((jd - 2451545.0) / 36525.0);
            time_controller.set_time(epoch);
            None
        }

        InputEvent::SetTimeRate { rate } => {
            time_controller.set_rate_years_per_second(rate);
            None
        }

        InputEvent::Teleport { x, y, z } => {
            camera.set_position(DVec3::new(x, y, z));
            None
        }

        InputEvent::LookAt { x, y, z } => {
            camera.look_at(DVec3::new(x, y, z));
            None
        }

        InputEvent::Ping { client_time } => {
            let server_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            Some(OutputEvent::Pong { client_time, server_time })
        }
    }
}
