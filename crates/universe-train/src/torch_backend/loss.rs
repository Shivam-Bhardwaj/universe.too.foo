//! Loss functions for Gaussian splatting optimization

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

/// Combined loss: L1 + lambda * D-SSIM
pub fn combined_loss(rendered: &Tensor, target: &Tensor, lambda_dssim: f32) -> Tensor {
    let l1 = l1_loss(rendered, target);
    let dssim = dssim_loss(rendered, target);
    &l1 + &dssim * (lambda_dssim as f64)
}
