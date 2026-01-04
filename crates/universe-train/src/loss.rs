//! Loss functions for Gaussian Splatting training
//!
//! Includes 3D geometry regularization to prevent Z-collapse during training.

use burn::prelude::*;
use burn::tensor::Tensor;

/// Combined loss: L1 + λ * D-SSIM
pub fn combined_loss<B: Backend>(
    rendered: Tensor<B, 3>, // [H, W, 3]
    target: Tensor<B, 3>,   // [H, W, 3]
    lambda_dssim: f32,
) -> Tensor<B, 1> {
    let l1 = l1_loss(rendered.clone(), target.clone());
    let dssim = dssim_loss(rendered, target);

    l1 + dssim * lambda_dssim
}

/// Full loss with 3D geometry regularization to prevent Z-collapse
///
/// For astronomical point sources (stars), we want spherical splats.
/// This loss adds:
/// - Isotropy regularization: penalizes deviation from spherical shape
/// - Collapse prevention: penalizes any axis from shrinking too small
pub fn full_loss_with_regularization<B: Backend>(
    rendered: Tensor<B, 3>,  // [H, W, 3]
    target: Tensor<B, 3>,    // [H, W, 3]
    scales: Tensor<B, 2>,    // [N, 3] - actual scale values (not log)
    log_scales: Tensor<B, 2>, // [N, 3] - log of scale values
    lambda_dssim: f32,
    lambda_isotropy: f32,
    lambda_collapse: f32,
    min_scale_ratio: f32,
) -> (Tensor<B, 1>, LossComponentsBurn) {
    // Image-space losses
    let l1 = l1_loss(rendered.clone(), target.clone());
    let dssim = dssim_loss(rendered, target);
    let image_loss = l1.clone() + dssim.clone() * lambda_dssim;

    // 3D geometry regularization
    let isotropy = log_scale_variance_loss(log_scales);
    let collapse = scale_collapse_loss(scales, min_scale_ratio);

    // Total loss
    let total = image_loss
        + isotropy.clone() * lambda_isotropy
        + collapse.clone() * lambda_collapse;

    let components = LossComponentsBurn {
        l1: l1.clone().into_scalar().elem::<f32>(),
        dssim: dssim.clone().into_scalar().elem::<f32>(),
        isotropy: isotropy.clone().into_scalar().elem::<f32>(),
        collapse: collapse.clone().into_scalar().elem::<f32>(),
        total: total.clone().into_scalar().elem::<f32>(),
    };

    (total, components)
}

/// Isotropy regularization: penalizes deviation from spherical shape
///
/// For point sources like stars, all three scale values should be equal.
/// This computes variance of log-scales across axes.
pub fn log_scale_variance_loss<B: Backend>(log_scales: Tensor<B, 2>) -> Tensor<B, 1> {
    // log_scales: [N, 3]
    // Compute mean log scale per splat [N, 1]
    let mean_log = log_scales.clone().mean_dim(1);

    // Broadcast mean to [N, 3] for subtraction
    // Burn's `mean_dim` keeps the reduced dimension, so `mean_log` is already [N, 1].
    // Avoid `unsqueeze_dim` here (can trigger rank-mismatch issues on some backends).
    let mean_expanded = mean_log.repeat_dim(1, 3);

    // Variance from mean
    let deviation = log_scales - mean_expanded;
    let variance = deviation.powf_scalar(2.0).mean();

    variance
}

/// Scale collapse prevention: penalizes any axis from becoming too small
///
/// Computes: mean(ReLU(min_ratio - min_scale/max_scale))
pub fn scale_collapse_loss<B: Backend>(scales: Tensor<B, 2>, min_ratio: f32) -> Tensor<B, 1> {
    // scales: [N, 3]
    // Get min and max scale per splat
    // min_dim(1) returns [N, 1], squeeze to [N]
    let min_scale: Tensor<B, 1> = scales.clone().min_dim(1).squeeze();
    let max_scale: Tensor<B, 1> = scales.max_dim(1).squeeze();

    // Compute ratio with small epsilon to avoid division by zero
    let ratio = min_scale / (max_scale + 1e-8);

    // Penalize if ratio < min_ratio: ReLU(min_ratio - ratio)
    // Use subtraction and clamp instead of Tensor::full for simplicity
    let violation = (ratio * -1.0 + min_ratio).clamp_min(0.0);

    violation.mean()
}

/// Breakdown of loss components for logging (Burn backend)
#[derive(Debug, Clone, Copy)]
pub struct LossComponentsBurn {
    pub l1: f32,
    pub dssim: f32,
    pub isotropy: f32,
    pub collapse: f32,
    pub total: f32,
}

impl std::fmt::Display for LossComponentsBurn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "total={:.4} (L1={:.4} DSSIM={:.4} iso={:.4} col={:.4})",
            self.total, self.l1, self.dssim, self.isotropy, self.collapse)
    }
}

/// L1 (Mean Absolute Error) loss
pub fn l1_loss<B: Backend>(rendered: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 1> {
    (rendered - target).abs().mean()
}

/// L2 (Mean Squared Error) loss
#[allow(dead_code)]
pub fn l2_loss<B: Backend>(rendered: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 1> {
    (rendered - target).powf_scalar(2.0).mean()
}

/// D-SSIM loss: (1 - SSIM) / 2
/// Simplified SSIM over the whole image
pub fn dssim_loss<B: Backend>(
    rendered: Tensor<B, 3>, // [H, W, 3]
    target: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let c1 = 0.01_f32.powi(2);
    let c2 = 0.03_f32.powi(2);

    // Compute means (returns scalar 0-D tensor)
    let mu_x = rendered.clone().mean();
    let mu_y = target.clone().mean();

    // Compute variances and covariance
    // Subtract scalar from 3D tensor, then compute variance
    let diff_x = rendered.clone() - mu_x.clone().unsqueeze::<3>().repeat(&[rendered.dims()[0], rendered.dims()[1], rendered.dims()[2]]);
    let diff_y = target.clone() - mu_y.clone().unsqueeze::<3>().repeat(&[target.dims()[0], target.dims()[1], target.dims()[2]]);

    let sigma_x_sq = diff_x.clone().powf_scalar(2.0).mean();
    let sigma_y_sq = diff_y.clone().powf_scalar(2.0).mean();
    let sigma_xy = (diff_x * diff_y).mean();

    // SSIM formula
    let numerator = (mu_x.clone() * mu_y.clone() * 2.0 + c1) * (sigma_xy * 2.0 + c2);
    let denominator =
        (mu_x.powf_scalar(2.0) + mu_y.powf_scalar(2.0) + c1) * (sigma_x_sq + sigma_y_sq + c2);

    let ssim = numerator / denominator;

    // D-SSIM - expand to 1D tensor
    let one: Tensor<B, 1> = Tensor::ones([1], &ssim.device());
    let dssim = (one - ssim.unsqueeze::<1>()) / 2.0;
    dssim
}

/// PSNR metric (not differentiable, for logging)
#[allow(dead_code)]
pub fn psnr<B: Backend>(rendered: Tensor<B, 3>, target: Tensor<B, 3>) -> f32 {
    let mse = (rendered - target)
        .powf_scalar(2.0)
        .mean()
        .into_scalar()
        .elem::<f32>();
    if mse < 1e-10 {
        return 100.0;
    }
    10.0 * (1.0 / mse).log10()
}
