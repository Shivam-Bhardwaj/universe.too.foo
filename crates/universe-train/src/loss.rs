//! Loss functions for Gaussian Splatting training

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
