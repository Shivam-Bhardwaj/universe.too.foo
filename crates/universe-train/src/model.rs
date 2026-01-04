//! Differentiable Gaussian Splatting model using Burn

use burn::prelude::*;
use burn::tensor::{backend::AutodiffBackend, Tensor};
use burn::module::{Module, Param};

/// Learnable Gaussian Cloud parameters
#[derive(Module, Debug)]
pub struct GaussianCloud<B: Backend> {
    /// Positions [N, 3]
    pub positions: Param<Tensor<B, 2>>,
    /// Log-scales [N, 3] (log for numerical stability)
    pub log_scales: Param<Tensor<B, 2>>,
    /// Rotation quaternions [N, 4]
    pub rotations: Param<Tensor<B, 2>>,
    /// RGB colors [N, 3]
    pub colors: Param<Tensor<B, 2>>,
    /// Logit opacities [N] (sigmoid applied during render)
    pub logit_opacities: Param<Tensor<B, 1>>,
}

impl<B: Backend> GaussianCloud<B> {
    /// Create from initial splat data
    pub fn from_splats(
        device: &B::Device,
        positions: Vec<[f32; 3]>,
        scales: Vec<[f32; 3]>,
        rotations: Vec<[f32; 4]>,
        colors: Vec<[f32; 3]>,
        opacities: Vec<f32>,
    ) -> Self {
        let n = positions.len();

        // Flatten and convert to tensors
        let pos_data: Vec<f32> = positions.iter().flat_map(|p| p.iter().copied()).collect();
        let scale_data: Vec<f32> = scales
            .iter()
            .flat_map(|s| s.iter().map(|x| x.max(1e-6).ln())) // Log-scale
            .collect();
        let rot_data: Vec<f32> = rotations.iter().flat_map(|r| r.iter().copied()).collect();
        let color_data: Vec<f32> = colors.iter().flat_map(|c| c.iter().copied()).collect();
        let opacity_data: Vec<f32> = opacities
            .iter()
            .map(|o| inverse_sigmoid(o.clamp(0.01, 0.99))) // Logit
            .collect();

        Self {
            positions: Param::from_tensor(
                Tensor::<B, 1>::from_floats(pos_data.as_slice(), device).reshape([n, 3]),
            ),
            log_scales: Param::from_tensor(
                Tensor::<B, 1>::from_floats(scale_data.as_slice(), device).reshape([n, 3]),
            ),
            rotations: Param::from_tensor(
                Tensor::<B, 1>::from_floats(rot_data.as_slice(), device).reshape([n, 4]),
            ),
            colors: Param::from_tensor(
                Tensor::<B, 1>::from_floats(color_data.as_slice(), device).reshape([n, 3]),
            ),
            logit_opacities: Param::from_tensor(Tensor::<B, 1>::from_floats(
                opacity_data.as_slice(),
                device,
            )),
        }
    }

    /// Number of gaussians
    pub fn num_gaussians(&self) -> usize {
        self.positions.dims()[0]
    }

    /// Get actual scales (exp of log_scales)
    pub fn scales(&self) -> Tensor<B, 2> {
        self.log_scales.val().exp()
    }

    /// Get actual opacities (sigmoid of logits)
    pub fn opacities(&self) -> Tensor<B, 1> {
        burn::tensor::activation::sigmoid(self.logit_opacities.val())
    }

    /// Normalize rotation quaternions
    pub fn normalized_rotations(&self) -> Tensor<B, 2> {
        let r = self.rotations.val();
        let norm = r.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        // `sum_dim(1)` keeps the reduced dimension, so `norm` is [N,1].
        // Some backends have issues broadcasting with an extra unsqueeze here; expand explicitly.
        let norm4 = norm.repeat(&[1, 4]); // [N,1] -> [N,4]
        r / norm4
    }

    /// Export to GaussianSplat vec
    pub fn to_splats(&self) -> Vec<universe_data::GaussianSplat> {
        let n = self.num_gaussians();

        let pos: Vec<f32> = self.positions.val().to_data().to_vec().unwrap();
        let scales: Vec<f32> = self.scales().to_data().to_vec().unwrap();
        let rots: Vec<f32> = self.normalized_rotations().to_data().to_vec().unwrap();
        let colors: Vec<f32> = self
            .colors
            .val()
            .clamp(0.0, 1.0)
            .to_data()
            .to_vec()
            .unwrap();
        let opacities: Vec<f32> = self.opacities().to_data().to_vec().unwrap();

        (0..n)
            .map(|i| universe_data::GaussianSplat {
                pos: [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]],
                scale: [scales[i * 3], scales[i * 3 + 1], scales[i * 3 + 2]],
                rotation: [rots[i * 4], rots[i * 4 + 1], rots[i * 4 + 2], rots[i * 4 + 3]],
                color: [colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]],
                opacity: opacities[i],
            })
            .collect()
    }
}

fn inverse_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}
