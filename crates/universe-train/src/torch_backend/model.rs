//! TorchGaussianCloud - Learnable Gaussian parameters as tch::Tensor

use tch::{Device, Kind, Tensor};
use universe_data::GaussianSplat;

/// Gaussian cloud parameters as tch::Tensor with requires_grad
pub struct TorchGaussianCloud {
    /// Positions [N, 3] - requires_grad = true
    pub positions: Tensor,
    /// Log-scales [N, 3] - requires_grad = true
    pub log_scales: Tensor,
    /// Rotation quaternions [N, 4] - requires_grad = true
    pub rotations: Tensor,
    /// RGB colors [N, 3] - requires_grad = true
    pub colors: Tensor,
    /// Logit opacities [N] - requires_grad = true
    pub logit_opacities: Tensor,
    /// Device for tensors
    device: Device,
}

impl TorchGaussianCloud {
    /// Create from splat data, moving tensors to specified device
    pub fn from_splats(splats: &[GaussianSplat], device: Device) -> Self {
        let n = splats.len() as i64;

        // Collect data into flat Vec<f32>
        let pos_data: Vec<f32> = splats.iter().flat_map(|s| s.pos.iter().copied()).collect();
        let scale_data: Vec<f32> = splats
            .iter()
            .flat_map(|s| s.scale.iter().map(|x| x.max(1e-6).ln()))
            .collect();
        let rot_data: Vec<f32> = splats
            .iter()
            .flat_map(|s| s.rotation.iter().copied())
            .collect();
        let color_data: Vec<f32> = splats
            .iter()
            .flat_map(|s| s.color.iter().copied())
            .collect();
        let opacity_data: Vec<f32> = splats
            .iter()
            .map(|s| inverse_sigmoid(s.opacity.clamp(0.01, 0.99)))
            .collect();

        Self {
            positions: Tensor::from_slice(&pos_data)
                .view([n, 3])
                .to_device(device)
                .set_requires_grad(true),
            log_scales: Tensor::from_slice(&scale_data)
                .view([n, 3])
                .to_device(device)
                .set_requires_grad(true),
            rotations: Tensor::from_slice(&rot_data)
                .view([n, 4])
                .to_device(device)
                .set_requires_grad(true),
            colors: Tensor::from_slice(&color_data)
                .view([n, 3])
                .to_device(device)
                .set_requires_grad(true),
            logit_opacities: Tensor::from_slice(&opacity_data)
                .to_device(device)
                .set_requires_grad(true),
            device,
        }
    }

    /// Get all trainable parameters for optimizer
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![
            self.positions.shallow_clone(),
            self.log_scales.shallow_clone(),
            self.rotations.shallow_clone(),
            self.colors.shallow_clone(),
            self.logit_opacities.shallow_clone(),
        ]
    }

    /// Number of gaussians
    pub fn num_gaussians(&self) -> i64 {
        self.positions.size()[0]
    }

    /// Get device
    pub fn device(&self) -> Device {
        self.device
    }

    /// Actual scales (exp of log_scales)
    pub fn scales(&self) -> Tensor {
        self.log_scales.exp()
    }

    /// Actual opacities (sigmoid of logits)
    pub fn opacities(&self) -> Tensor {
        self.logit_opacities.sigmoid()
    }

    /// Normalized rotation quaternions
    pub fn normalized_rotations(&self) -> Tensor {
        let r = &self.rotations;
        let norm = r.pow_tensor_scalar(2).sum_dim_intlist([1i64].as_slice(), true, Kind::Float).sqrt();
        r / &norm
    }

    /// Export to GaussianSplat vec
    pub fn to_splats(&self) -> Vec<GaussianSplat> {
        let n = self.num_gaussians() as usize;

        // Move to CPU for extraction (no_grad to avoid tracking)
        let pos: Vec<f32> = Vec::try_from(
            self.positions.detach().to_device(Device::Cpu)
        ).unwrap();
        let scales: Vec<f32> = Vec::try_from(
            self.scales().detach().to_device(Device::Cpu)
        ).unwrap();
        let rots: Vec<f32> = Vec::try_from(
            self.normalized_rotations().detach().to_device(Device::Cpu)
        ).unwrap();
        let colors: Vec<f32> = Vec::try_from(
            self.colors.clamp(0.0, 1.0).detach().to_device(Device::Cpu)
        ).unwrap();
        let opacities: Vec<f32> = Vec::try_from(
            self.opacities().detach().to_device(Device::Cpu)
        ).unwrap();

        (0..n)
            .map(|i| GaussianSplat {
                pos: [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]],
                scale: [scales[i * 3], scales[i * 3 + 1], scales[i * 3 + 2]],
                rotation: [
                    rots[i * 4],
                    rots[i * 4 + 1],
                    rots[i * 4 + 2],
                    rots[i * 4 + 3],
                ],
                color: [colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]],
                opacity: opacities[i],
            })
            .collect()
    }
}

fn inverse_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}
