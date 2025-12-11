//! Camera model for rendering splats

use glam::{Mat4, Vec3, Vec4};
use std::f32::consts::PI;

/// Perspective camera for rendering
#[derive(Clone, Debug)]
pub struct Camera {
    /// Position in world space
    pub position: Vec3,
    /// Look-at target
    pub target: Vec3,
    /// Up vector
    pub up: Vec3,
    /// Vertical field of view in radians
    pub fov_y: f32,
    /// Aspect ratio (width/height)
    pub aspect: f32,
    /// Near plane distance
    pub near: f32,
    /// Far plane distance
    pub far: f32,
}

impl Camera {
    pub fn new(position: Vec3, target: Vec3, aspect: f32) -> Self {
        Self {
            position,
            target,
            up: Vec3::Y,
            fov_y: 60.0_f32.to_radians(),
            aspect,
            near: 0.1,
            far: 1e12, // Astronomical distances
        }
    }

    /// View matrix (world to camera)
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    /// Projection matrix (camera to clip)
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.near, self.far)
    }

    /// Combined view-projection matrix
    pub fn view_projection(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// Forward direction (normalized)
    pub fn forward(&self) -> Vec3 {
        (self.target - self.position).normalize()
    }

    /// Right direction (normalized)
    pub fn right(&self) -> Vec3 {
        self.forward().cross(self.up).normalize()
    }

    /// Project world point to screen coordinates [0,1]
    pub fn project(&self, world_pos: Vec3) -> Option<(f32, f32, f32)> {
        let clip = self.view_projection() * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);

        if clip.w <= 0.0 {
            return None; // Behind camera
        }

        let ndc = clip.truncate() / clip.w;

        // NDC to screen [0, 1]
        let u = (ndc.x + 1.0) * 0.5;
        let v = (1.0 - ndc.y) * 0.5; // Flip Y for image coordinates
        let depth = ndc.z;

        if u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 {
            return None; // Outside frustum
        }

        Some((u, v, depth))
    }
}

/// Generate camera views around a bounding box for training
pub fn generate_training_cameras(
    center: Vec3,
    radius: f32,
    num_views: usize,
    image_size: (u32, u32),
) -> Vec<Camera> {
    let aspect = image_size.0 as f32 / image_size.1 as f32;
    let mut cameras = Vec::with_capacity(num_views);

    // Distribute cameras on a sphere around the center
    let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;

    for i in 0..num_views {
        let t = i as f32 / num_views as f32;

        // Fibonacci sphere distribution
        let theta = 2.0 * PI * t * golden_ratio;
        let phi = (1.0 - 2.0 * (i as f32 + 0.5) / num_views as f32).acos();

        let x = phi.sin() * theta.cos();
        let y = phi.cos();
        let z = phi.sin() * theta.sin();

        let position = center + Vec3::new(x, y, z) * radius * 2.0;

        cameras.push(Camera::new(position, center, aspect));
    }

    cameras
}
