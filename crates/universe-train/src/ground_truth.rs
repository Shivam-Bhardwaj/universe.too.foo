//! Ground truth renderer for training targets
//!
//! Generates reference images from raw astronomical data

use universe_data::GaussianSplat;
use image::{Rgb, RgbImage};

use crate::camera::Camera;
use glam::Vec3;

/// Simple CPU-based renderer for ground truth generation
pub struct GroundTruthRenderer {
    width: u32,
    height: u32,
}

impl GroundTruthRenderer {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Render splats to image (simple point-based, no splatting)
    pub fn render(&self, splats: &[GaussianSplat], camera: &Camera) -> RgbImage {
        let mut image = RgbImage::new(self.width, self.height);

        // Sort splats by distance (back to front for correct alpha)
        let mut sorted: Vec<_> = splats
            .iter()
            .filter_map(|s| {
                let pos = Vec3::new(s.pos[0], s.pos[1], s.pos[2]);
                let depth = (pos - camera.position).length();
                Some((s, depth))
            })
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Back to front

        for (splat, _depth) in sorted {
            let pos = Vec3::new(splat.pos[0], splat.pos[1], splat.pos[2]);

            if let Some((u, v, _z)) = camera.project(pos) {
                let px = (u * self.width as f32) as i32;
                let py = (v * self.height as f32) as i32;

                // Calculate screen-space radius based on scale and distance
                let distance = (pos - camera.position).length();
                let max_scale = splat.scale[0].max(splat.scale[1]).max(splat.scale[2]);
                let focal = self.height as f32 / (2.0 * (camera.fov_y / 2.0).tan());
                let radius = ((max_scale * focal / distance) as i32).max(1).min(50);

                // Draw Gaussian blob
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let x = px + dx;
                        let y = py + dy;

                        if x < 0 || x >= self.width as i32 || y < 0 || y >= self.height as i32 {
                            continue;
                        }

                        // Gaussian falloff
                        let dist_sq = (dx * dx + dy * dy) as f32;
                        let sigma = radius as f32 / 2.0;
                        let weight = (-dist_sq / (2.0 * sigma * sigma)).exp();

                        if weight < 0.01 {
                            continue;
                        }

                        let alpha = splat.opacity * weight;

                        // Get existing pixel
                        let existing = image.get_pixel(x as u32, y as u32);

                        // Alpha blend
                        let r = (existing[0] as f32 * (1.0 - alpha)
                            + splat.color[0] * 255.0 * alpha) as u8;
                        let g = (existing[1] as f32 * (1.0 - alpha)
                            + splat.color[1] * 255.0 * alpha) as u8;
                        let b = (existing[2] as f32 * (1.0 - alpha)
                            + splat.color[2] * 255.0 * alpha) as u8;

                        image.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
                    }
                }
            }
        }

        image
    }

    /// Convert RgbImage to tensor [H, W, 3] normalized to [0, 1]
    pub fn image_to_tensor<B: burn::prelude::Backend>(
        &self,
        image: &RgbImage,
        device: &B::Device,
    ) -> burn::tensor::Tensor<B, 3> {
        let (w, h) = image.dimensions();
        let data: Vec<f32> = image
            .pixels()
            .flat_map(|p| {
                [
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                ]
            })
            .collect();

        burn::tensor::Tensor::<B, 1>::from_floats(&data[..], device).reshape([h as usize, w as usize, 3])
    }
}
