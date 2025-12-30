// Milky Way Background Shader
// Procedural galactic band visible at extreme distances

struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    _pad0: f32,
    near: f32,
    far: f32,
    fov_y: f32,
    log_depth_c: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ray_dir: vec3<f32>,
}

// Fullscreen triangle
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    // Generate fullscreen triangle
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);

    var out: VertexOutput;
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);

    // Compute ray direction in world space
    let ndc = vec2<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0);

    // Inverse projection to get view-space ray
    let tan_half_fov = tan(camera.fov_y * 0.5);
    let aspect = camera.proj[1][1] / camera.proj[0][0];
    let view_ray = vec3<f32>(
        ndc.x * tan_half_fov * aspect,
        ndc.y * tan_half_fov,
        -1.0
    );

    // Transform to world space
    let inv_view = transpose(mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    ));
    out.ray_dir = normalize(inv_view * view_ray);

    return out;
}

// Simple hash for procedural noise
fn hash13(p3: vec3<f32>) -> f32 {
    var p = fract(p3 * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

// Perlin-like noise (simplified)
fn noise3d(p: vec3<f32>) -> f32 {
    let pi = floor(p);
    let pf = fract(p);

    // Smoothstep
    let w = pf * pf * (3.0 - 2.0 * pf);

    // 8 corner samples (simplified to 2 for performance)
    let n000 = hash13(pi);
    let n100 = hash13(pi + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash13(pi + vec3<f32>(0.0, 1.0, 0.0));
    let n001 = hash13(pi + vec3<f32>(0.0, 0.0, 1.0));

    let nx0 = mix(n000, n100, w.x);
    let nx1 = mix(n010, n001, w.x);
    return mix(nx0, nx1, w.y);
}

// Fractal Brownian Motion
fn fbm(p: vec3<f32>, octaves: i32) -> f32 {
    var sum: f32 = 0.0;
    var amp: f32 = 0.5;
    var freq: f32 = 1.0;
    var pp = p;

    for (var i = 0; i < octaves; i++) {
        sum += amp * noise3d(pp * freq);
        freq *= 2.0;
        amp *= 0.5;
    }

    return sum;
}

// Milky Way spiral arm density
fn spiral_density(dir: vec3<f32>) -> f32 {
    // Convert direction to galactic coordinates (simplified)
    // Galactic center approximately at RA=266°, Dec=-29°
    let gc_dir = vec3<f32>(0.0, -0.49, -0.87);  // Approximate

    // Distance to galactic plane
    let dist_to_plane = abs(dot(dir, vec3<f32>(0.0, 1.0, 0.0)));
    let plane_density = exp(-dist_to_plane * dist_to_plane * 20.0);

    // Add turbulence with noise
    let turbulence = fbm(dir * 3.0, 3);

    // Combine
    return plane_density * (0.7 + 0.3 * turbulence);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ray = normalize(in.ray_dir);

    // Calculate Milky Way band intensity
    let mw_density = spiral_density(ray);

    // Color: yellowish-white for starlight, bluish in dark regions
    let base_color = vec3<f32>(0.9, 0.95, 1.0);
    let dust_color = vec3<f32>(0.6, 0.7, 0.9);

    let color = mix(dust_color, base_color, mw_density);
    let intensity = mw_density * 0.15;  // Subtle background

    // Fade in only at galactic scales (> 1 Em = 1e18 m)
    let cam_dist = length(camera.position);
    let fade_threshold = 1e18;
    let fade = smoothstep(fade_threshold * 0.5, fade_threshold * 2.0, cam_dist);

    return vec4<f32>(color * intensity * fade, intensity * fade * 0.5);
}
