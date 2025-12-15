//! Renderer core (platform-agnostic)
//! Phase 2.1: Shared rendering logic

use crate::camera::{Camera, CameraUniform};
use universe_data::GaussianSplat;
use glam::DVec3;

/// Renderer state (platform-agnostic core)
pub struct RendererCore {
    pub camera: Camera,
    pub splats: Vec<GpuSplat>,
}

/// GPU splat representation
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSplat {
    pub pos: [f32; 3],
    pub _pad0: f32,
    pub scale: [f32; 3],
    pub _pad1: f32,
    pub rotation: [f32; 4],
    pub color: [f32; 3],
    pub opacity: f32,
}

impl RendererCore {
    pub fn new() -> Self {
        Self {
            camera: Camera::new(),
            splats: Vec::new(),
        }
    }

    /// Update camera
    pub fn update_camera(&mut self, _dt: f64) {
        self.camera.auto_speed();
    }

    /// Convert world splats to camera-relative GPU splats
    pub fn prepare_splats(&mut self, world_splats: &[(DVec3, GaussianSplat)]) {
        self.splats.clear();
        self.splats.reserve(world_splats.len());

        for (world_pos, splat) in world_splats {
            let camera_rel = self.camera.world_to_camera_relative(*world_pos);

            self.splats.push(GpuSplat {
                pos: camera_rel.into(),
                _pad0: 0.0,
                scale: splat.scale,
                _pad1: 0.0,
                rotation: splat.rotation,
                color: splat.color,
                opacity: splat.opacity,
            });
        }
    }

    /// Get camera uniform for GPU
    pub fn camera_uniform(&self, aspect: f32) -> CameraUniform {
        CameraUniform::from_camera(&self.camera, aspect)
    }
}

impl Default for RendererCore {
    fn default() -> Self {
        Self::new()
    }
}



