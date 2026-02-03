// Gaussian Splat Shader with Logarithmic Depth

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
}

@group(0) @binding(0) var<uniform> camera: Camera;

// Raw float array (16 floats/instance) so we can keep vec3 fields aligned in a GPU-friendly layout.
// Layout:
// 0-2: pos (x,y,z)
// 3:   pad
// 4-6: scale (x,y,z)
// 7:   pad
// 8-11: rotation (quat xyzw)
// 12-14: color (rgb)
// 15: opacity
@group(0) @binding(1) var<storage, read> splats: array<f32>;

struct Splat {
    pos: vec3<f32>,
    scale: vec3<f32>,
    rotation: vec4<f32>,
    color: vec3<f32>,
    opacity: f32,
}

fn fetch_splat(idx: u32) -> Splat {
    let offset = idx * 16u;
    
    let pos = vec3<f32>(splats[offset + 0u], splats[offset + 1u], splats[offset + 2u]);
    let scale = vec3<f32>(splats[offset + 4u], splats[offset + 5u], splats[offset + 6u]);
    let rotation = vec4<f32>(splats[offset + 8u], splats[offset + 9u], splats[offset + 10u], splats[offset + 11u]);
    let color = vec3<f32>(splats[offset + 12u], splats[offset + 13u], splats[offset + 14u]);
    let opacity = splats[offset + 15u];

    return Splat(pos, scale, rotation, color, opacity);
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) opacity: f32,
    @location(3) @interpolate(flat) log_depth_w: f32,
}

// Quad vertices for billboard
const QUAD_VERTICES: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let splat = fetch_splat(instance_idx);
    let quad_pos = QUAD_VERTICES[vertex_idx];

    // Transform splat center to view space
    let view_pos = camera.view * vec4<f32>(splat.pos, 1.0);

    // Calculate billboard size based on max scale
    let max_scale = max(splat.scale.x, max(splat.scale.y, splat.scale.z));

    // Screen-space sizing (perspective, clamp angular size).
    // NOTE: star splats encode a *visual* radius that grows with distance; this keeps them sane on screen.
    let dist = max(length(view_pos.xyz), 1.0);
    let angular_size = max_scale / dist;

    // Treat objects within ~100 AU as "nearby" (planets/spacecraft), allow a larger cap + minimum size.
    let nearby_threshold = 1.5e13; // ~100 AU
    let is_nearby = dist < nearby_threshold;

    // Min angular size for nearby objects (~5px), max angular size:
    // - stars capped at ~4px
    // - nearby objects capped at ~40px
    let min_angular = select(0.0, 0.003, is_nearby);
    let max_angular = select(0.002, 0.02, is_nearby);

    let effective_angular = clamp(angular_size, min_angular, max_angular);
    let screen_scale = effective_angular * dist;

    // Expand billboard in view space (screen-aligned)
    let billboard_offset = vec3<f32>(quad_pos * screen_scale, 0.0);
    let expanded_pos = view_pos.xyz + billboard_offset;

    // Project
    var clip_pos = camera.proj * vec4<f32>(expanded_pos, 1.0);

    // LOGARITHMIC DEPTH for astronomical scales (WebGPU depth range is 0..1)
    // We compute a log depth in [0, 1] based on view-space distance, then reverse it:
    // near -> 1.0, far -> 0.0 (for reverse-Z + Greater depth test).
    let z = max(1e-6, -expanded_pos.z); // view-space forward depth (meters)
    let denom = log2(camera.far * camera.log_depth_c + 1.0);
    let log_depth = log2(z * camera.log_depth_c + 1.0) / denom; // 0..1
    let depth = clamp(1.0 - log_depth, 0.0, 1.0);              // reverse-Z
    clip_pos.z = depth * clip_pos.w;

    var out: VertexOutput;
    out.clip_position = clip_pos;
    out.uv = quad_pos * 0.5 + 0.5;
    out.color = splat.color;
    out.opacity = splat.opacity;
    out.log_depth_w = clip_pos.w;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Gaussian falloff
    let d = in.uv - vec2<f32>(0.5);
    let dist_sq = dot(d, d);

    // Discard outside circle
    if dist_sq > 0.25 {
        discard;
    }

    // Gaussian intensity (sharper falloff => less "blurry discs")
    let gaussian = exp(-32.0 * dist_sq);

    // Boost visibility (treat stars as emissive-ish sprites)
    // This is effectively an exposure control for the current dataset.
    let alpha = clamp(gaussian * in.opacity * 8.0, 0.0, 1.0);

    // Premultiplied alpha output
    return vec4<f32>(in.color * alpha, alpha);
}