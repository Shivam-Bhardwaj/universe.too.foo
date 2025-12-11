// Simple fullscreen blit shader
// Copies a texture to the screen using a fullscreen triangle

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle trick: 3 vertices cover entire screen
@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    var out: VertexOutput;

    // Generate fullscreen triangle
    let x = f32((vertex_idx << 1u) & 2u);
    let y = f32(vertex_idx & 2u);

    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    // UVs should go from 0 to 1, clamp within the visible region
    out.uv = vec2<f32>(x * 0.5, y * 0.5);

    return out;
}

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Clamp UVs to valid range
    let uv = clamp(in.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    return textureSample(source_texture, source_sampler, uv);
}
