//! Camera math with floating origin and logarithmic depth
//! Phase 1.2: True camera math implementation

use glam::{DVec3, DQuat, Vec3, Mat4};
use std::f64::consts::PI;

/// Camera using f64 for position (astronomical precision)
/// Converts to f32 for GPU with floating origin
pub struct Camera {
    /// Position in world space (meters, f64 for precision)
    pub position: DVec3,
    /// Orientation quaternion
    pub orientation: DQuat,
    /// Vertical field of view (radians)
    pub fov_y: f32,
    /// Near plane (meters)
    pub near: f32,
    /// Far plane (meters) - can be huge with log depth
    pub far: f32,
    /// Movement speed (meters per second)
    pub speed: f64,
    /// Mouse sensitivity
    pub sensitivity: f32,

    // Euler angles for FPS-style control
    yaw: f64,   // Radians
    pitch: f64, // Radians
}

impl Camera {
    pub fn new() -> Self {
        Self {
            position: DVec3::new(0.0, 0.0, 1.5e11), // Start at 1 AU on Z
            orientation: DQuat::IDENTITY,
            fov_y: 60.0_f32.to_radians(),
            near: 1.0,           // 1 meter
            far: 1e20,           // 10 billion AU
            speed: 1e10,         // 10 million km/s default (faster for solar system scale)
            sensitivity: 0.003,  // Increased sensitivity
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    /// Forward direction (f64)
    pub fn forward(&self) -> DVec3 {
        self.orientation * DVec3::NEG_Z
    }

    /// Right direction (f64)
    pub fn right(&self) -> DVec3 {
        self.orientation * DVec3::X
    }

    /// Up direction (f64)
    pub fn up(&self) -> DVec3 {
        self.orientation * DVec3::Y
    }

    /// Update orientation from mouse delta
    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.yaw += dx as f64 * self.sensitivity as f64;
        self.pitch += dy as f64 * self.sensitivity as f64;

        // Clamp pitch to avoid gimbal lock
        self.pitch = self.pitch.clamp(-PI / 2.0 + 0.01, PI / 2.0 - 0.01);

        // Rebuild quaternion from Euler angles
        let yaw_quat = DQuat::from_rotation_y(self.yaw);
        let pitch_quat = DQuat::from_rotation_x(self.pitch);
        self.orientation = yaw_quat * pitch_quat;
    }

    /// Move camera (FPS style)
    pub fn translate(&mut self, forward: f64, right: f64, up: f64, dt: f64) {
        let movement = self.forward() * forward + self.right() * right + self.up() * up;
        self.position += movement * self.speed * dt;
    }

    /// View matrix (world to camera), using floating origin
    /// The origin is shifted to camera position to avoid precision loss
    pub fn view_matrix(&self) -> Mat4 {
        let forward = self.orientation * DVec3::NEG_Z;
        let up = self.orientation * DVec3::Y;

        // View matrix with camera at origin (floating origin technique)
        Mat4::look_to_rh(
            Vec3::ZERO,
            forward.as_vec3(),
            up.as_vec3(),
        )
    }

    /// Projection matrix with infinite far plane for reverse-Z
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        // Reverse-Z infinite far plane projection
        reverse_z_infinite_projection(self.fov_y, aspect, self.near)
    }

    /// Convert world position to camera-relative (for floating origin)
    pub fn world_to_camera_relative(&self, world_pos: DVec3) -> Vec3 {
        (world_pos - self.position).as_vec3()
    }

    /// Set speed based on current distance to origin
    pub fn auto_speed(&mut self) {
        let dist = self.position.length();
        // Speed = 1% of distance from origin per second, clamped
        self.speed = (dist * 0.01).clamp(1e3, 1e15);
    }

    /// Teleport to position
    pub fn set_position(&mut self, pos: DVec3) {
        self.position = pos;
        self.auto_speed();
    }

    /// Look at target
    pub fn look_at(&mut self, target: DVec3) {
        let dir = (target - self.position).normalize();

        self.pitch = (-dir.y).asin();
        self.yaw = dir.x.atan2(-dir.z);

        let yaw_quat = DQuat::from_rotation_y(self.yaw);
        let pitch_quat = DQuat::from_rotation_x(self.pitch);
        self.orientation = yaw_quat * pitch_quat;
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

/// Reverse-Z infinite far plane projection matrix
fn reverse_z_infinite_projection(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    let f = 1.0 / (fov_y / 2.0).tan();

    // Reverse-Z maps near to 1.0, infinity to 0.0
    Mat4::from_cols_array(&[
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, 0.0, -1.0,
        0.0, 0.0, near, 0.0,
    ])
}

/// Camera uniform buffer data for GPU
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub position: [f32; 3],
    pub _pad0: f32,
    pub near: f32,
    pub far: f32,
    pub fov_y: f32,
    pub log_depth_c: f32, // Logarithmic depth constant
}

impl CameraUniform {
    pub fn from_camera(camera: &Camera, aspect: f32) -> Self {
        let view = camera.view_matrix();
        let proj = camera.projection_matrix(aspect);
        let view_proj = proj * view;

        // Phase 1.2: Compute log depth constant for astronomical scales
        let log_depth_c = 1.0 / camera.near.max(1.0);

        Self {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            view_proj: view_proj.to_cols_array_2d(),
            position: [0.0, 0.0, 0.0], // Always zero with floating origin
            _pad0: 0.0,
            near: camera.near,
            far: camera.far,
            fov_y: camera.fov_y,
            log_depth_c,
        }
    }
}




