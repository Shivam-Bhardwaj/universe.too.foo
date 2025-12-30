//! Differentiable Gaussian Splatting rasterizer using tch-rs
//!
//! This implements the same rasterization pipeline as the Burn version
//! but using LibTorch tensor operations for CUDA acceleration.

use crate::camera::Camera;
use glam::Mat4;
use tch::{Device, Kind, Tensor};

/// Convert glam::Mat4 to tch::Tensor [4, 4]
/// Note: glam uses column-major, but we need row-major for matmul with row vectors
fn mat4_to_tensor(mat: &Mat4, device: Device) -> Tensor {
    let data: Vec<f32> = mat.to_cols_array().to_vec();
    Tensor::from_slice(&data)
        .view([4, 4])
        .transpose(0, 1)  // Convert column-major to row-major
        .to_device(device)
}

/// Quaternion to rotation matrix (batched)
/// Converts [N, 4] normalized quaternions to [N, 3, 3] rotation matrices
fn quaternion_to_rotation_matrix(q: &Tensor) -> Tensor {
    let device = q.device();
    let n = q.size()[0];

    // Extract quaternion components (x, y, z, w)
    let x = q.select(1, 0); // [N]
    let y = q.select(1, 1);
    let z = q.select(1, 2);
    let w = q.select(1, 3);

    // Precompute common terms
    let xx = &x * &x;
    let yy = &y * &y;
    let zz = &z * &z;
    let xy = &x * &y;
    let xz = &x * &z;
    let yz = &y * &z;
    let wx = &w * &x;
    let wy = &w * &y;
    let wz = &w * &z;

    // Build rotation matrix elements
    let one = Tensor::ones([n], (Kind::Float, device));
    let r00 = &one - (&yy + &zz) * 2.0;
    let r01 = (&xy - &wz) * 2.0;
    let r02 = (&xz + &wy) * 2.0;
    let r10 = (&xy + &wz) * 2.0;
    let r11 = &one - (&xx + &zz) * 2.0;
    let r12 = (&yz - &wx) * 2.0;
    let r20 = (&xz - &wy) * 2.0;
    let r21 = (&yz + &wx) * 2.0;
    let r22 = &one - (&xx + &yy) * 2.0;

    // Stack into [N, 3, 3]
    let row0 = Tensor::stack(&[r00, r01, r02], 1); // [N, 3]
    let row1 = Tensor::stack(&[r10, r11, r12], 1);
    let row2 = Tensor::stack(&[r20, r21, r22], 1);

    Tensor::stack(&[row0, row1, row2], 1) // [N, 3, 3]
}

/// Build 3D covariance matrices: Σ = R · S · S^T · R^T
fn build_covariance_3d(scales: &Tensor, rot_matrices: &Tensor) -> Tensor {
    let n = scales.size()[0];

    // Scale each column of R by the corresponding scale
    let sx = scales.select(1, 0).view([n, 1, 1]);
    let sy = scales.select(1, 1).view([n, 1, 1]);
    let sz = scales.select(1, 2).view([n, 1, 1]);

    let col0 = rot_matrices.slice(2, 0, 1, 1) * &sx; // [N, 3, 1]
    let col1 = rot_matrices.slice(2, 1, 2, 1) * &sy;
    let col2 = rot_matrices.slice(2, 2, 3, 1) * &sz;

    let rs = Tensor::cat(&[col0, col1, col2], 2); // [N, 3, 3]

    // Σ = RS @ RS^T
    rs.matmul(&rs.transpose(1, 2))
}

/// Project 3D covariance to 2D via Jacobian
fn project_covariance_2d(
    cov_3d: &Tensor,    // [N, 3, 3]
    view_pos: &Tensor,  // [N, 3]
    focal_x: f64,
    focal_y: f64,
) -> Tensor {
    let n = view_pos.size()[0];
    let device = view_pos.device();

    // Extract view space coordinates
    let x = view_pos.select(1, 0);
    let y = view_pos.select(1, 1);
    let z_view = view_pos.select(1, 2);

    // Use -z for positive depth
    let z = -&z_view;
    let z2 = &z * &z;

    // Jacobian elements
    let ones = Tensor::ones_like(&z);
    let j00 = &ones * focal_x / &z;
    let j02 = -&x * focal_x / &z2;
    let j11 = &ones * focal_y / &z;
    let j12 = -&y * focal_y / &z2;

    // Extract covariance elements (symmetric)
    let s00 = cov_3d.select(1, 0).select(1, 0);
    let s01 = cov_3d.select(1, 0).select(1, 1);
    let s02 = cov_3d.select(1, 0).select(1, 2);
    let s11 = cov_3d.select(1, 1).select(1, 1);
    let s12 = cov_3d.select(1, 1).select(1, 2);
    let s22 = cov_3d.select(1, 2).select(1, 2);

    // Compute J * Σ
    let t00 = &j00 * &s00 + &j02 * &s02;
    let t01 = &j00 * &s01 + &j02 * &s12;
    let t02 = &j00 * &s02 + &j02 * &s22;
    let t11 = &j11 * &s11 + &j12 * &s12;
    let t12 = &j11 * &s12 + &j12 * &s22;

    // Compute (J * Σ) * J^T
    let cov_2d_00 = &t00 * &j00 + &t02 * &j02 + 0.3; // Low-pass filter
    let cov_2d_01 = &t01 * &j11 + &t02 * &j12;
    let cov_2d_11 = &t11 * &j11 + &t12 * &j12 + 0.3;

    // Stack into [N, 2, 2]
    let row0 = Tensor::stack(&[cov_2d_00, cov_2d_01.shallow_clone()], 1);
    let row1 = Tensor::stack(&[cov_2d_01, cov_2d_11], 1);

    Tensor::stack(&[row0, row1], 1)
}

/// Invert 2D covariance to conic form [N, 3] = (a, b, c)
fn invert_to_conic(cov_2d: &Tensor) -> Tensor {
    let n = cov_2d.size()[0];

    let a = cov_2d.select(1, 0).select(1, 0);
    let b = cov_2d.select(1, 0).select(1, 1);
    let c = cov_2d.select(1, 1).select(1, 1);

    let det = &a * &c - &b * &b;
    let valid = det.gt(1e-6).to_kind(Kind::Float);

    let inv_a = &c / &det * &valid;
    let inv_b = -&b / &det * &valid;
    let inv_c = &a / &det * &valid;

    Tensor::stack(&[inv_a, inv_b, inv_c], 1)
}

/// Generate pixel coordinate grid [H, W, 2]
fn generate_pixel_grid(h: i64, w: i64, device: Device) -> Tensor {
    let x = Tensor::arange(w, (Kind::Float, device))
        .view([1, w])
        .repeat([h, 1]); // [H, W]
    let y = Tensor::arange(h, (Kind::Float, device))
        .view([h, 1])
        .repeat([1, w]); // [H, W]

    Tensor::stack(&[x, y], 2) // [H, W, 2]
}

/// Evaluate Gaussian contribution at each pixel
/// Returns alpha values [H, W, N]
fn evaluate_gaussians(
    pixel_coords: &Tensor,   // [H, W, 2]
    screen_pos: &Tensor,     // [N, 2]
    conics: &Tensor,         // [N, 3]
    opacities: &Tensor,      // [N]
) -> Tensor {
    let h = pixel_coords.size()[0];
    let w = pixel_coords.size()[1];
    let n = screen_pos.size()[0];

    // Broadcast: [H, W, 1, 2] - [1, 1, N, 2] -> [H, W, N, 2]
    let pixels = pixel_coords.unsqueeze(2);      // [H, W, 1, 2]
    let centers = screen_pos.view([1, 1, n, 2]); // [1, 1, N, 2]
    let dx = &pixels - &centers;                 // [H, W, N, 2]

    let dx_x = dx.select(3, 0); // [H, W, N]
    let dx_y = dx.select(3, 1);

    // Conic coefficients [1, 1, N]
    let a = conics.select(1, 0).view([1, 1, n]);
    let b = conics.select(1, 1).view([1, 1, n]);
    let c = conics.select(1, 2).view([1, 1, n]);

    // Mahalanobis distance squared
    let dist_sq = &a * &dx_x * &dx_x + &b * &dx_x * &dx_y * 2.0 + &c * &dx_y * &dx_y;

    // Gaussian falloff with 3σ cutoff
    let gaussian = (-&dist_sq * 0.5).exp();
    let cutoff_mask = dist_sq.lt(9.0).to_kind(Kind::Float);
    let gaussian = &gaussian * &cutoff_mask;

    // Apply opacity [1, 1, N]
    let opacities_exp = opacities.view([1, 1, n]);
    (&gaussian * &opacities_exp).clamp(0.0, 1.0)
}

/// Alpha blending with depth sorting
fn alpha_blend(
    alphas: &Tensor,  // [H, W, N]
    colors: &Tensor,  // [N, 3]
    depths: &Tensor,  // [N]
) -> Tensor {
    let h = alphas.size()[0];
    let w = alphas.size()[1];
    let n = alphas.size()[2];
    let device = alphas.device();

    // Sort by depth (back to front) - non-differentiable
    let (_, indices) = depths.sort(0, false); // Ascending
    let indices_vec: Vec<i64> = Vec::try_from(&indices).unwrap();

    // Gather sorted alphas and colors
    let mut alphas_sorted_list = Vec::new();
    let mut colors_sorted_list = Vec::new();
    for &idx in &indices_vec {
        alphas_sorted_list.push(alphas.select(2, idx).unsqueeze(2));
        colors_sorted_list.push(colors.select(0, idx).unsqueeze(0));
    }
    let alphas_sorted = Tensor::cat(&alphas_sorted_list, 2);
    let colors_sorted = Tensor::cat(&colors_sorted_list, 0);

    // Cumulative transmittance
    let one_minus_alpha = Tensor::ones_like(&alphas_sorted) - &alphas_sorted;

    let mut transmittance_list = Vec::new();
    let mut cum_t = Tensor::ones([h, w, 1], (Kind::Float, device));
    for i in 0..(n as i64) {
        transmittance_list.push(cum_t.shallow_clone());
        if i < n as i64 - 1 {
            let oma_i = one_minus_alpha.select(2, i).unsqueeze(2);
            cum_t = &cum_t * &oma_i;
        }
    }
    let transmittance = Tensor::cat(&transmittance_list, 2);

    // Weighted contributions
    let weights = &alphas_sorted * &transmittance; // [H, W, N]

    // Expand and blend
    let weights_exp = weights.unsqueeze(3);              // [H, W, N, 1]
    let colors_exp = colors_sorted.view([1, 1, n, 3]);   // [1, 1, N, 3]
    let weighted = &weights_exp * &colors_exp;           // [H, W, N, 3]

    weighted
        .sum_dim_intlist([2i64].as_slice(), false, Kind::Float)
        .clamp(0.0, 1.0)
}

/// Project 3D positions to 2D screen space
fn project_to_screen(
    positions: &Tensor, // [N, 3]
    camera: &Camera,
    width: i64,
    height: i64,
) -> (Tensor, Tensor, Tensor) {
    let device = positions.device();
    let n = positions.size()[0];

    // Camera matrices
    let view_matrix = mat4_to_tensor(&camera.view_matrix(), device);
    let proj_matrix = mat4_to_tensor(&camera.projection_matrix(), device);

    // Homogeneous coordinates [N, 4]
    let ones = Tensor::ones([n, 1], (Kind::Float, device));
    let pos_homo = Tensor::cat(&[positions.shallow_clone(), ones], 1);

    // View space
    let view_pos_homo = pos_homo.matmul(&view_matrix);

    // Clip space
    let clip_pos = view_pos_homo.matmul(&proj_matrix);

    // Perspective divide
    let w_clip = clip_pos.select(1, 3).unsqueeze(1);
    let ndc = clip_pos.slice(1, 0, 3, 1) / &w_clip;

    // NDC to screen
    let ndc_x = ndc.select(1, 0);
    let ndc_y = ndc.select(1, 1);
    let x_screen = (&ndc_x + 1.0) * (width as f64 / 2.0);
    let y_screen = (1.0 - &ndc_y) * (height as f64 / 2.0); // Flip Y
    let screen_pos = Tensor::stack(&[x_screen, y_screen], 1); // [N, 2]

    // View positions and depths
    let view_pos = view_pos_homo.slice(1, 0, 3, 1);
    let depths = view_pos.select(1, 2);

    (screen_pos, depths, view_pos)
}

/// Render Gaussians to an image using differentiable splatting
pub fn render_gaussians(
    positions: &Tensor,  // [N, 3]
    scales: &Tensor,     // [N, 3]
    rotations: &Tensor,  // [N, 4]
    colors: &Tensor,     // [N, 3]
    opacities: &Tensor,  // [N]
    camera: &Camera,
    image_size: (u32, u32),
) -> Tensor {
    let device = positions.device();
    let (w, h) = (image_size.0 as i64, image_size.1 as i64);

    // Phase 1: Project to screen
    let (screen_pos, depths, view_pos) = project_to_screen(positions, camera, w, h);

    // Phase 2: Covariance transformation
    let rot_matrices = quaternion_to_rotation_matrix(rotations);
    let cov_3d = build_covariance_3d(scales, &rot_matrices);
    let focal_x = w as f64 / 2.0;
    let focal_y = h as f64 / 2.0;
    let cov_2d = project_covariance_2d(&cov_3d, &view_pos, focal_x, focal_y);
    let conics = invert_to_conic(&cov_2d);

    // Phase 3: Per-pixel Gaussian evaluation
    let pixel_coords = generate_pixel_grid(h, w, device);
    let alphas = evaluate_gaussians(&pixel_coords, &screen_pos, &conics, opacities);

    // Phase 4: Alpha blending
    alpha_blend(&alphas, colors, &depths)
}
