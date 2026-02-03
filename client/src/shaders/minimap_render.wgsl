// Minimap Render Shader
// Renders the density grid to a quad

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;
    // Fullscreen quad from 2 triangles
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

struct RenderParams {
    grid_size: u32,
    max_density: f32,
}

@group(0) @binding(0) var<storage, read> density_grid: array<u32>;
@group(0) @binding(1) var<uniform> params: RenderParams;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let gx = u32(in.uv.x * f32(params.grid_size));
    let gy = u32(in.uv.y * f32(params.grid_size)); // Flip Y if needed

    let idx = gy * params.grid_size + gx;
    
    let count = density_grid[idx];
    
    if (count == 0u) {
        discard;
    }

    // Logarithmic mapping
    let intensity = log2(f32(count) + 1.0) / log2(params.max_density + 1.0);
    
    // Color map: Dark blue -> Cyan -> White
    let color = mix(
        vec3<f32>(0.0, 0.1, 0.3),
        vec3<f32>(0.5, 1.0, 1.0),
        intensity
    );

    // Circular mask
    let dx = in.uv.x - 0.5;
    let dy = in.uv.y - 0.5;
    if (dx*dx + dy*dy > 0.25) {
        discard;
    }

    return vec4<f32>(color, 0.8 * intensity + 0.2);
}


