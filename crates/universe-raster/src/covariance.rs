//! 3D Gaussian to 2D screen-space projection mathematics
//!
//! This module implements the core covariance transformations needed for
//! projecting 3D Gaussians onto the image plane as 2D ellipses.

use glam::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};

/// Convert a rotation quaternion to a 3x3 rotation matrix
#[inline]
pub fn quat_to_rotation_matrix(q: Quat) -> Mat3 {
    Mat3::from_quat(q)
}

/// Build a 3D covariance matrix from scale and rotation
///
/// The covariance matrix is computed as: Σ = R * S * S^T * R^T
/// where R is the rotation matrix and S is the diagonal scale matrix.
///
/// # Arguments
/// * `scale` - The (sx, sy, sz) axis scales of the Gaussian
/// * `rotation` - The rotation quaternion (must be normalized)
///
/// # Returns
/// A symmetric 3x3 covariance matrix
pub fn build_covariance_3d(scale: Vec3, rotation: Quat) -> Mat3 {
    let r = Mat3::from_quat(rotation);

    // S is diagonal, so R*S is just scaling the columns of R
    let rs = Mat3::from_cols(
        r.col(0) * scale.x,
        r.col(1) * scale.y,
        r.col(2) * scale.z,
    );

    // Σ = RS * RS^T
    rs * rs.transpose()
}

/// Project a 3D covariance matrix to 2D screen space
///
/// Uses the Jacobian of the perspective projection to transform the
/// 3D covariance to a 2D covariance in image coordinates.
///
/// # Arguments
/// * `cov_3d` - The 3D covariance matrix in view space
/// * `view_pos` - Position of the Gaussian center in view space (z should be negative, pointing into screen)
/// * `focal_x` - Focal length in pixels (width / (2 * tan(fov_x/2)))
/// * `focal_y` - Focal length in pixels (height / (2 * tan(fov_y/2)))
///
/// # Returns
/// A 2x2 covariance matrix in screen pixel coordinates
pub fn project_covariance(cov_3d: Mat3, view_pos: Vec3, focal_x: f32, focal_y: f32) -> [[f32; 2]; 2] {
    // In view space, z is negative (OpenGL convention), so we use -z for the positive depth
    let z = -view_pos.z;
    let z2 = z * z;

    if z < 0.001 {
        // Behind camera or too close - return identity as fallback
        return [[1.0, 0.0], [0.0, 1.0]];
    }

    // Jacobian of perspective projection:
    // J = | fx/z    0    -fx*x/z^2 |
    //     |  0    fy/z   -fy*y/z^2 |
    //
    // Note: we're projecting the 3D covariance, so we need J * Σ * J^T

    let j00 = focal_x / z;
    let j02 = -focal_x * view_pos.x / z2;
    let j11 = focal_y / z;
    let j12 = -focal_y * view_pos.y / z2;

    // Extract covariance elements (symmetric matrix)
    let s00 = cov_3d.col(0).x;
    let s01 = cov_3d.col(1).x;
    let s02 = cov_3d.col(2).x;
    let s11 = cov_3d.col(1).y;
    let s12 = cov_3d.col(2).y;
    let s22 = cov_3d.col(2).z;

    // Compute J * Σ (2x3 * 3x3 = 2x3)
    // Row 0: [j00*s00 + j02*s02, j00*s01 + j02*s12, j00*s02 + j02*s22]
    // Row 1: [j11*s01 + j12*s02, j11*s11 + j12*s12, j11*s12 + j12*s22]
    let t00 = j00 * s00 + j02 * s02;
    let t01 = j00 * s01 + j02 * s12;
    let t02 = j00 * s02 + j02 * s22;
    let _t10 = j11 * s01 + j12 * s02;
    let t11 = j11 * s11 + j12 * s12;
    let t12 = j11 * s12 + j12 * s22;

    // Compute (J * Σ) * J^T (2x3 * 3x2 = 2x2)
    // J^T = | j00  0   |
    //       |  0  j11  |
    //       | j02 j12  |
    let cov_2d_00 = t00 * j00 + t02 * j02;
    let cov_2d_01 = t01 * j11 + t02 * j12;
    let cov_2d_11 = t11 * j11 + t12 * j12;

    [[cov_2d_00, cov_2d_01], [cov_2d_01, cov_2d_11]]
}

/// Convert a 2D covariance matrix to conic form for efficient Gaussian evaluation
///
/// The conic form (a, b, c) allows evaluating the Gaussian as:
/// G(dx, dy) = exp(-0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2))
///
/// where (dx, dy) is the offset from the Gaussian center.
///
/// # Arguments
/// * `cov` - 2x2 covariance matrix
///
/// # Returns
/// * `Some((a, b, c))` - The conic coefficients (inverse covariance)
/// * `None` - If the covariance is degenerate (determinant <= 0)
pub fn covariance_to_conic(cov: [[f32; 2]; 2]) -> Option<Vec3> {
    let det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];

    if det <= 1e-6 {
        return None;
    }

    let det_inv = 1.0 / det;

    // Inverse of 2x2 matrix: [d, -b; -c, a] / det
    // But we have symmetric matrix [a, b; b, c], so inverse is [c, -b; -b, a] / det
    Some(Vec3::new(
        cov[1][1] * det_inv,  // a (inverse cov[0][0])
        -cov[0][1] * det_inv, // b (inverse cov[0][1])
        cov[0][0] * det_inv,  // c (inverse cov[1][1])
    ))
}

/// Compute eigenvalues of a 2x2 symmetric matrix
///
/// For a symmetric matrix [[a, b], [b, c]], the eigenvalues are:
/// λ = (a + c) / 2 ± sqrt(((a - c) / 2)^2 + b^2)
///
/// # Returns
/// (λ_max, λ_min) - The larger and smaller eigenvalues
pub fn eigenvalues_2x2(cov: [[f32; 2]; 2]) -> (f32, f32) {
    let a = cov[0][0];
    let b = cov[0][1];
    let c = cov[1][1];

    let trace = a + c;
    let det = a * c - b * b;

    let discriminant = (trace * trace * 0.25 - det).max(0.0);
    let sqrt_disc = discriminant.sqrt();

    let half_trace = trace * 0.5;
    (half_trace + sqrt_disc, half_trace - sqrt_disc)
}

/// Compute the bounding radius for a 2D Gaussian
///
/// Returns the radius (in pixels) that contains 99.7% of the Gaussian's mass
/// (3 standard deviations along the major axis).
///
/// # Arguments
/// * `cov` - 2x2 covariance matrix in screen space
///
/// # Returns
/// The bounding radius in pixels
pub fn compute_bounding_radius(cov: [[f32; 2]; 2]) -> f32 {
    let (lambda_max, _) = eigenvalues_2x2(cov);
    3.0 * lambda_max.max(0.0).sqrt()
}

/// Add a low-pass filter to the covariance to prevent aliasing
///
/// This adds a small identity component to ensure the Gaussian
/// covers at least a fraction of a pixel.
pub fn apply_low_pass_filter(cov: [[f32; 2]; 2], filter_size: f32) -> [[f32; 2]; 2] {
    [
        [cov[0][0] + filter_size, cov[0][1]],
        [cov[1][0], cov[1][1] + filter_size],
    ]
}

/// Full projection pipeline: 3D Gaussian parameters to 2D rendering parameters
///
/// # Arguments
/// * `position` - World-space position of Gaussian center
/// * `scale` - (sx, sy, sz) axis scales
/// * `rotation` - Rotation quaternion (normalized)
/// * `view_matrix` - Camera view matrix (world to view transform)
/// * `focal_x`, `focal_y` - Focal lengths in pixels
/// * `screen_width`, `screen_height` - Screen dimensions in pixels
///
/// # Returns
/// * `Some((screen_pos, conic, radius))` - Screen position, conic coefficients, bounding radius
/// * `None` - If the Gaussian should be culled
pub fn project_gaussian(
    position: Vec3,
    scale: Vec3,
    rotation: Quat,
    view_matrix: Mat4,
    proj_matrix: Mat4,
    focal_x: f32,
    focal_y: f32,
    screen_width: f32,
    screen_height: f32,
) -> Option<(Vec2, Vec3, f32)> {
    // Transform to view space
    let view_pos = view_matrix.transform_point3(position);

    // Cull behind camera (with margin for Gaussian extent)
    let max_scale = scale.x.max(scale.y).max(scale.z);
    if view_pos.z > -0.1 + max_scale * 3.0 {
        return None;
    }

    // Build and project covariance
    let cov_3d = build_covariance_3d(scale, rotation);

    // Transform covariance to view space
    // Σ_view = R_view * Σ_world * R_view^T
    // Since we're dealing with positions already in view space relative to camera,
    // and rotations are world-space, we need to transform the rotation part
    let view_rot = Mat3::from_mat4(view_matrix);
    let cov_3d_view = view_rot * cov_3d * view_rot.transpose();

    let cov_2d = project_covariance(cov_3d_view, view_pos, focal_x, focal_y);

    // Apply low-pass filter (0.3 pixels minimum variance)
    let cov_2d_filtered = apply_low_pass_filter(cov_2d, 0.3);

    // Convert to conic
    let conic = covariance_to_conic(cov_2d_filtered)?;

    // Compute bounding radius
    let radius = compute_bounding_radius(cov_2d_filtered);

    // Cull tiny splats
    if radius < 0.25 {
        return None;
    }

    // Project center to screen
    let clip = proj_matrix * Vec4::new(view_pos.x, view_pos.y, view_pos.z, 1.0);
    if clip.w <= 0.0 {
        return None;
    }

    let ndc = clip.truncate() / clip.w;
    let screen_x = (ndc.x * 0.5 + 0.5) * screen_width;
    let screen_y = (1.0 - (ndc.y * 0.5 + 0.5)) * screen_height;

    // Cull off-screen
    if screen_x + radius < 0.0 || screen_x - radius > screen_width ||
       screen_y + radius < 0.0 || screen_y - radius > screen_height {
        return None;
    }

    Some((Vec2::new(screen_x, screen_y), conic, radius))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity_covariance() {
        // Unit sphere Gaussian
        let scale = Vec3::ONE;
        let rotation = Quat::IDENTITY;
        let cov = build_covariance_3d(scale, rotation);

        // Should be identity matrix
        assert_relative_eq!(cov.col(0).x, 1.0, epsilon = 1e-6);
        assert_relative_eq!(cov.col(1).y, 1.0, epsilon = 1e-6);
        assert_relative_eq!(cov.col(2).z, 1.0, epsilon = 1e-6);
        assert_relative_eq!(cov.col(0).y, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_scaled_covariance() {
        let scale = Vec3::new(2.0, 1.0, 0.5);
        let rotation = Quat::IDENTITY;
        let cov = build_covariance_3d(scale, rotation);

        // Diagonal should be scale^2
        assert_relative_eq!(cov.col(0).x, 4.0, epsilon = 1e-6);
        assert_relative_eq!(cov.col(1).y, 1.0, epsilon = 1e-6);
        assert_relative_eq!(cov.col(2).z, 0.25, epsilon = 1e-6);
    }

    #[test]
    fn test_conic_inversion() {
        let cov = [[4.0, 0.0], [0.0, 1.0]];
        let conic = covariance_to_conic(cov).unwrap();

        // Inverse should be [0.25, 0, 1]
        assert_relative_eq!(conic.x, 0.25, epsilon = 1e-6);
        assert_relative_eq!(conic.y, 0.0, epsilon = 1e-6);
        assert_relative_eq!(conic.z, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_eigenvalues() {
        // Identity matrix has eigenvalues (1, 1)
        let cov = [[1.0, 0.0], [0.0, 1.0]];
        let (l1, l2) = eigenvalues_2x2(cov);
        assert_relative_eq!(l1, 1.0, epsilon = 1e-6);
        assert_relative_eq!(l2, 1.0, epsilon = 1e-6);

        // [[4, 0], [0, 1]] has eigenvalues (4, 1)
        let cov2 = [[4.0, 0.0], [0.0, 1.0]];
        let (l1, l2) = eigenvalues_2x2(cov2);
        assert_relative_eq!(l1, 4.0, epsilon = 1e-6);
        assert_relative_eq!(l2, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_bounding_radius() {
        // Unit variance -> 3 sigma radius = 3
        let cov = [[1.0, 0.0], [0.0, 1.0]];
        let radius = compute_bounding_radius(cov);
        assert_relative_eq!(radius, 3.0, epsilon = 1e-6);

        // 4x variance -> 3 * 2 = 6
        let cov2 = [[4.0, 0.0], [0.0, 1.0]];
        let radius2 = compute_bounding_radius(cov2);
        assert_relative_eq!(radius2, 6.0, epsilon = 1e-6);
    }
}
