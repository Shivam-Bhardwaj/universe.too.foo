// Compute shader: Project 3D Gaussians to 2D screen-space ellipses
//
// This shader transforms 3D Gaussian splats into 2D rendering parameters,
// including computing the 2D covariance matrix from the 3D covariance and
// camera projection.

// Input: 3D Gaussian splat data (camera-relative positions)
struct SplatInput {
    pos: vec3<f32>,
    _pad0: f32,
    scale: vec3<f32>,
    _pad1: f32,
    rotation: vec4<f32>,
    color: vec3<f32>,
    opacity: f32,
}

// Output: 2D projected splat for rendering (64 bytes)
struct Splat2D {
    center: vec2<f32>,     // 8 bytes (offset 0)
    conic: vec3<f32>,      // 12 bytes (offset 8)
    depth: f32,            // 4 bytes (offset 20)
    color: vec3<f32>,      // 12 bytes (offset 24)
    opacity: f32,          // 4 bytes (offset 36)
    radius: f32,           // 4 bytes (offset 40)
    _pad0: f32,            // 4 bytes (offset 44)
    _pad1: vec4<f32>,      // 16 bytes (offset 48)
    // Total: 64 bytes
}

// Camera uniform with extended parameters for rasterization
struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    _pad0: f32,
    near: f32,
    far: f32,
    fov_y: f32,
    log_depth_c: f32,
    focal_x: f32,
    focal_y: f32,
    width: f32,
    height: f32,
}

@group(0) @binding(0) var<storage, read> splats_in: array<SplatInput>;
@group(0) @binding(1) var<storage, read_write> splats_2d: array<Splat2D>;
@group(0) @binding(2) var<uniform> camera: Camera;
@group(0) @binding(3) var<storage, read_write> visible_count: atomic<u32>;

// Convert quaternion to 3x3 rotation matrix
fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;

    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;

    let xx2 = x * x2;
    let xy2 = x * y2;
    let xz2 = x * z2;
    let yy2 = y * y2;
    let yz2 = y * z2;
    let zz2 = z * z2;
    let wx2 = w * x2;
    let wy2 = w * y2;
    let wz2 = w * z2;

    return mat3x3<f32>(
        vec3<f32>(1.0 - yy2 - zz2, xy2 + wz2, xz2 - wy2),
        vec3<f32>(xy2 - wz2, 1.0 - xx2 - zz2, yz2 + wx2),
        vec3<f32>(xz2 + wy2, yz2 - wx2, 1.0 - xx2 - yy2)
    );
}

// Build 3D covariance matrix from scale and rotation
// Σ = R * S * S^T * R^T where S is diagonal scale matrix
fn build_covariance_3d(scale: vec3<f32>, rotation: vec4<f32>) -> mat3x3<f32> {
    let R = quat_to_mat3(rotation);

    // RS = R * diag(scale)
    let RS = mat3x3<f32>(
        R[0] * scale.x,
        R[1] * scale.y,
        R[2] * scale.z
    );

    // Σ = RS * RS^T
    return RS * transpose(RS);
}

// Check if a Gaussian is approximately isotropic (spherical)
fn is_isotropic(sigma_3d: mat3x3<f32>) -> bool {
    // For an isotropic Gaussian, all diagonal elements should be equal
    // and off-diagonal elements should be near zero
    let diag_avg = (sigma_3d[0][0] + sigma_3d[1][1] + sigma_3d[2][2]) / 3.0;
    if diag_avg < 1e-10 {
        return false;
    }

    let tolerance = 0.1; // 10% tolerance
    let d0_diff = abs(sigma_3d[0][0] - diag_avg) / diag_avg;
    let d1_diff = abs(sigma_3d[1][1] - diag_avg) / diag_avg;
    let d2_diff = abs(sigma_3d[2][2] - diag_avg) / diag_avg;

    // Check diagonal elements are similar
    if d0_diff > tolerance || d1_diff > tolerance || d2_diff > tolerance {
        return false;
    }

    // Check off-diagonal elements are small relative to diagonal
    let off_diag_max = max(abs(sigma_3d[0][1]), max(abs(sigma_3d[0][2]), abs(sigma_3d[1][2])));
    if off_diag_max / diag_avg > tolerance {
        return false;
    }

    return true;
}

// Simple projection for isotropic (spherical) Gaussians
// For a sphere, the screen-space covariance is just a circle
fn project_isotropic(scale: f32, depth: f32) -> mat2x2<f32> {
    // Angular size in pixels: scale / depth * focal_length
    // Variance is (angular_size)^2
    let angular_size_x = scale / depth * camera.focal_x;
    let angular_size_y = scale / depth * camera.focal_y;

    return mat2x2<f32>(
        vec2<f32>(angular_size_x * angular_size_x, 0.0),
        vec2<f32>(0.0, angular_size_y * angular_size_y)
    );
}

// Project 3D covariance to 2D using Jacobian of perspective projection
fn project_covariance(sigma_3d: mat3x3<f32>, view_pos: vec3<f32>) -> mat2x2<f32> {
    let z = -view_pos.z; // Positive depth (view space z is negative)

    if z < 0.001 {
        return mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
    }

    // Use reciprocal for numerical stability
    let z_inv = 1.0 / z;
    let z_inv2 = z_inv * z_inv;

    // Jacobian of perspective projection
    // J = | fx/z    0    -fx*x/z^2 |
    //     |  0    fy/z   -fy*y/z^2 |
    let j00 = camera.focal_x * z_inv;
    let j02 = -camera.focal_x * view_pos.x * z_inv2;
    let j11 = camera.focal_y * z_inv;
    let j12 = -camera.focal_y * view_pos.y * z_inv2;

    // Compute Σ_2d = J * Σ_3d * J^T directly without forming full matrices
    // This avoids numerical issues with extreme scale differences

    // Extract covariance elements
    let s00 = sigma_3d[0][0]; let s01 = sigma_3d[0][1]; let s02 = sigma_3d[0][2];
    let s11 = sigma_3d[1][1]; let s12 = sigma_3d[1][2];
    let s22 = sigma_3d[2][2];

    // Compute J * Σ (2x3 result, but we only need certain elements)
    // Row 0: [j00*s00 + j02*s02, j00*s01 + j02*s12, j00*s02 + j02*s22]
    // Row 1: [j11*s01 + j12*s02, j11*s11 + j12*s12, j11*s12 + j12*s22]

    // Then (J*Σ) * J^T where J^T has columns [j00, 0], [0, j11], [j02, j12]

    // Result[0][0] = row0 · col0 of J^T = (j00*s00 + j02*s02)*j00 + (j00*s02 + j02*s22)*j02
    let cov_00 = j00*j00*s00 + 2.0*j00*j02*s02 + j02*j02*s22;

    // Result[1][1] = row1 · col1 of J^T = (j11*s11 + j12*s12)*j11 + (j11*s12 + j12*s22)*j12
    let cov_11 = j11*j11*s11 + 2.0*j11*j12*s12 + j12*j12*s22;

    // Result[0][1] = row0 · col1 of J^T = (j00*s01 + j02*s12)*j11 + (j00*s02 + j02*s22)*j12
    let cov_01 = j00*j11*s01 + j00*j12*s02 + j02*j11*s12 + j02*j12*s22;

    return mat2x2<f32>(
        vec2<f32>(cov_00, cov_01),
        vec2<f32>(cov_01, cov_11)
    );
}

// Compute eigenvalues of 2x2 symmetric matrix
fn eigenvalues_2x2(m: mat2x2<f32>) -> vec2<f32> {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[1][1];

    let trace = a + c;
    let det = a * c - b * b;

    let discriminant = max(0.0, trace * trace * 0.25 - det);
    let sqrt_disc = sqrt(discriminant);

    let half_trace = trace * 0.5;
    return vec2<f32>(half_trace + sqrt_disc, half_trace - sqrt_disc);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&splats_in) {
        return;
    }

    let splat = splats_in[idx];

    // Skip zero-opacity splats
    if splat.opacity <= 0.0 {
        splats_2d[idx].opacity = 0.0;
        return;
    }

    // Position is already camera-relative (floating origin)
    // Transform to view space
    let view_pos = (camera.view * vec4<f32>(splat.pos, 1.0)).xyz;

    // Compute distance to splat (positive value)
    let distance = length(view_pos);
    let depth = -view_pos.z; // Positive depth along view direction

    // Cull behind camera (with margin for Gaussian extent)
    let max_scale = max(splat.scale.x, max(splat.scale.y, splat.scale.z));
    if view_pos.z > -camera.near + max_scale * 3.0 {
        splats_2d[idx].opacity = 0.0;
        return;
    }

    // Guard against division by zero
    if distance < 0.001 {
        splats_2d[idx].opacity = 0.0;
        return;
    }

    // Build 3D covariance in world space
    let sigma_3d_world = build_covariance_3d(splat.scale, splat.rotation);

    // Transform covariance to view space
    // For a covariance matrix: Σ_view = R_view * Σ_world * R_view^T
    let view_rot = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    );
    let sigma_3d_view = view_rot * sigma_3d_world * transpose(view_rot);

    // Project to 2D covariance
    var cov_2d: mat2x2<f32>;
    if is_isotropic(sigma_3d_view) {
        // Fast path for spheres (planets) - avoids numerical precision issues
        // at astronomical scales by computing angular size directly
        let avg_scale = sqrt(sigma_3d_view[0][0]); // Extract scale from variance
        cov_2d = project_isotropic(avg_scale, depth);
    } else {
        // General path for anisotropic Gaussians using Jacobian projection
        cov_2d = project_covariance(sigma_3d_view, view_pos);
    }

    // DEBUG: Force circular covariance to test if issue is in projection or rasterization
    // ALSO force large variance to make it very visible
    // Remove this after testing!
    cov_2d = mat2x2<f32>(
        vec2<f32>(10000.0, 0.0),
        vec2<f32>(0.0, 10000.0)
    );

    // Apply low-pass filter to prevent aliasing (minimum 0.3 pixel variance)
    cov_2d[0][0] += 0.3;
    cov_2d[1][1] += 0.3;

    // Compute determinant for inversion
    let det = cov_2d[0][0] * cov_2d[1][1] - cov_2d[0][1] * cov_2d[1][0];
    if det <= 1e-6 {
        splats_2d[idx].opacity = 0.0;
        return;
    }

    // Invert covariance to get conic form
    let det_inv = 1.0 / det;
    let conic = vec3<f32>(
        cov_2d[1][1] * det_inv,  // a
        -cov_2d[0][1] * det_inv, // b
        cov_2d[0][0] * det_inv   // c
    );

    // Compute bounding radius (3 sigma)
    let eigenvalues = eigenvalues_2x2(cov_2d);
    let radius = 3.0 * sqrt(max(eigenvalues.x, eigenvalues.y));

    // Cull tiny splats (less than 0.25 pixel radius)
    if radius < 0.25 {
        splats_2d[idx].opacity = 0.0;
        return;
    }

    // Project center to screen coordinates
    let clip = camera.proj * vec4<f32>(view_pos, 1.0);
    if clip.w <= 0.0 {
        splats_2d[idx].opacity = 0.0;
        return;
    }

    let ndc = clip.xyz / clip.w;
    let screen_x = (ndc.x * 0.5 + 0.5) * camera.width;
    let screen_y = (1.0 - (ndc.y * 0.5 + 0.5)) * camera.height; // Flip Y

    // Cull off-screen splats
    if screen_x + radius < 0.0 || screen_x - radius > camera.width ||
       screen_y + radius < 0.0 || screen_y - radius > camera.height {
        splats_2d[idx].opacity = 0.0;
        return;
    }

    // Compute logarithmic depth for sorting (reverse-Z compatible)
    let fcoef = 2.0 / log2(camera.far * camera.log_depth_c + 1.0);
    let log_depth = log2(max(1e-6, depth * camera.log_depth_c + 1.0)) * fcoef;

    // Write output
    splats_2d[idx].center = vec2<f32>(screen_x, screen_y);
    splats_2d[idx].conic = conic;
    splats_2d[idx].depth = log_depth;
    splats_2d[idx].color = splat.color;
    splats_2d[idx].opacity = splat.opacity;
    splats_2d[idx].radius = radius;
    splats_2d[idx]._pad0 = 0.0;
    splats_2d[idx]._pad1 = vec4<f32>(0.0);

    // Increment visible count
    atomicAdd(&visible_count, 1u);
}
