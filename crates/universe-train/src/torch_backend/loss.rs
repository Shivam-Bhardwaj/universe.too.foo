//! Loss functions for Gaussian splatting optimization
//!
//! Includes regularization terms to prevent Z-collapse during training.

use tch::{Kind, Tensor};

/// L1 (Mean Absolute Error) loss
pub fn l1_loss(rendered: &Tensor, target: &Tensor) -> Tensor {
    (rendered - target).abs().mean(Kind::Float)
}

/// D-SSIM (Structural Dissimilarity) loss
///
/// Computes (1 - SSIM) / 2 as a loss value.
/// Uses simplified global SSIM computation.
pub fn dssim_loss(rendered: &Tensor, target: &Tensor) -> Tensor {
    let c1: f64 = 0.01_f64.powi(2);
    let c2: f64 = 0.03_f64.powi(2);

    // Global means
    let mu_x = rendered.mean(Kind::Float);
    let mu_y = target.mean(Kind::Float);

    // Centered signals
    let diff_x = rendered - &mu_x;
    let diff_y = target - &mu_y;

    // Variances and covariance
    let sigma_x_sq = diff_x.pow_tensor_scalar(2).mean(Kind::Float);
    let sigma_y_sq = diff_y.pow_tensor_scalar(2).mean(Kind::Float);
    let sigma_xy = (&diff_x * &diff_y).mean(Kind::Float);

    // SSIM formula
    let numerator = (&mu_x * &mu_y * 2.0 + c1) * (&sigma_xy * 2.0 + c2);
    let denominator =
        (mu_x.pow_tensor_scalar(2) + mu_y.pow_tensor_scalar(2) + c1) * (&sigma_x_sq + &sigma_y_sq + c2);

    let ssim = &numerator / &denominator;

    // D-SSIM = (1 - SSIM) / 2
    (1.0 - ssim) / 2.0
}

/// Isotropy regularization loss for point sources (stars)
///
/// Penalizes deviation from spherical shape by measuring variance between scale axes.
/// For stars (point sources), all three scale values should be equal.
///
/// Loss = mean(variance of scales per splat) = mean((sx-μ)² + (sy-μ)² + (sz-μ)²)
/// where μ = (sx + sy + sz) / 3
pub fn isotropy_loss(scales: &Tensor) -> Tensor {
    // scales: [N, 3] - scale values per splat
    // Compute mean scale per splat [N, 1]
    let mean_scale = scales.mean_dim([1i64].as_slice(), true, Kind::Float);

    // Compute variance from mean (deviation from spherical)
    let deviation = scales - &mean_scale;
    let variance = deviation.pow_tensor_scalar(2).mean(Kind::Float);

    variance
}

/// Scale collapse prevention loss
///
/// Penalizes any scale axis from becoming too small relative to the others.
/// This prevents Z-collapse where one axis shrinks to near-zero.
///
/// Loss = mean(max(0, min_ratio - min_scale/max_scale))
/// where min_ratio is the minimum allowed ratio (e.g., 0.1 = 10% of max)
pub fn scale_collapse_loss(scales: &Tensor, min_ratio: f64) -> Tensor {
    // scales: [N, 3]
    // Get min and max scale per splat
    let min_scale = scales.min_dim(1, false).0; // [N]
    let max_scale = scales.max_dim(1, false).0; // [N]

    // Compute ratio (add small epsilon to prevent division by zero)
    let ratio = &min_scale / (&max_scale + 1e-8);

    // Penalize if ratio < min_ratio
    // Loss = ReLU(min_ratio - ratio) = max(0, min_ratio - ratio)
    let violation = (min_ratio - ratio).relu();

    violation.mean(Kind::Float)
}

/// Log-scale variance loss (alternative to isotropy)
///
/// Works in log-space to handle the wide range of astronomical scales.
/// Penalizes variance in log(scale) across axes.
pub fn log_scale_variance_loss(log_scales: &Tensor) -> Tensor {
    // log_scales: [N, 3] - log of scale values
    let mean_log = log_scales.mean_dim([1i64].as_slice(), true, Kind::Float);
    let deviation = log_scales - &mean_log;
    let variance = deviation.pow_tensor_scalar(2).mean(Kind::Float);

    variance
}

/// Combined loss with regularization: L1 + lambda_dssim * D-SSIM + regularization
pub fn combined_loss(rendered: &Tensor, target: &Tensor, lambda_dssim: f32) -> Tensor {
    let l1 = l1_loss(rendered, target);
    let dssim = dssim_loss(rendered, target);
    &l1 + &dssim * (lambda_dssim as f64)
}

/// Full loss with 3D geometry regularization
///
/// Parameters:
/// - rendered: Rendered image tensor [H, W, 3]
/// - target: Ground truth image tensor [H, W, 3]
/// - scales: Splat scales tensor [N, 3] (actual scales, not log)
/// - log_scales: Log of scales [N, 3]
/// - lambda_dssim: Weight for D-SSIM loss (typically 0.2)
/// - lambda_isotropy: Weight for isotropy regularization (typically 0.01)
/// - lambda_collapse: Weight for collapse prevention (typically 0.1)
/// - min_scale_ratio: Minimum allowed ratio of min/max scale (typically 0.1-0.2)
pub fn full_loss_with_regularization(
    rendered: &Tensor,
    target: &Tensor,
    scales: &Tensor,
    log_scales: &Tensor,
    lambda_dssim: f32,
    lambda_isotropy: f32,
    lambda_collapse: f32,
    min_scale_ratio: f32,
) -> (Tensor, LossComponents) {
    // Image-space losses
    let l1 = l1_loss(rendered, target);
    let dssim = dssim_loss(rendered, target);
    let image_loss = &l1 + &dssim * (lambda_dssim as f64);

    // 3D geometry regularization
    let isotropy = log_scale_variance_loss(log_scales);
    let collapse = scale_collapse_loss(scales, min_scale_ratio as f64);

    // Total loss
    let total = &image_loss
        + &isotropy * (lambda_isotropy as f64)
        + &collapse * (lambda_collapse as f64);

    let components = LossComponents {
        l1: l1.double_value(&[]) as f32,
        dssim: dssim.double_value(&[]) as f32,
        isotropy: isotropy.double_value(&[]) as f32,
        collapse: collapse.double_value(&[]) as f32,
        total: total.double_value(&[]) as f32,
    };

    (total, components)
}

/// Breakdown of loss components for logging
#[derive(Debug, Clone, Copy)]
pub struct LossComponents {
    pub l1: f32,
    pub dssim: f32,
    pub isotropy: f32,
    pub collapse: f32,
    pub total: f32,
}

impl std::fmt::Display for LossComponents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "total={:.4} (L1={:.4} DSSIM={:.4} iso={:.4} col={:.4})",
            self.total, self.l1, self.dssim, self.isotropy, self.collapse)
    }
}
