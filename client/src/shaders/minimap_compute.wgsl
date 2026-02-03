// Minimap Compute Shader
// Accumulates splat density into a grid

struct MinimapParams {
    grid_size: u32,
    // Axis-aligned bounds for the minimap (e.g. heliosphere bounds)
    min_x: f32,
    min_z: f32,
    size_x: f32,
    size_z: f32,
    // Offset from splat coordinates (camera-relative) to world coordinates
    offset_x: f32,
    offset_z: f32,
}

@group(0) @binding(0) var<storage, read> splats: array<f32>;
@group(0) @binding(1) var<storage, read_write> density_grid: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: MinimapParams;

// Splat layout (16 floats, matches `finalGpuSplats` instance layout):
// 0-2: pos (x, y, z)
// 3:   pad
// 4-6: scale
// 7:   pad
// 8-11: rot (quat)
// 12-14: color
// 15: opacity

const SPLAT_STRIDE: u32 = 16u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Safety check: ensure we don't read past end
    // (Note: arrayLength returns number of f32s, divide by stride)
    if (idx * SPLAT_STRIDE >= arrayLength(&splats)) {
        return;
    }

    let base_idx = idx * SPLAT_STRIDE;
    
    let px = splats[base_idx];
    let py = splats[base_idx + 1u]; // Y is up/down, we project XZ
    let pz = splats[base_idx + 2u];

    // World position = relative_pos + offset
    // Grid relative = World position - min_pos
    // Combined: grid_rel = px + offset_x - min_x
    // Actually, params.offset_x should be (camera_origin.x), so world_x = px + offset_x
    // rel_x = world_x - min_x = px + offset_x - min_x
    
    // We can precompute (offset_x - min_x) on CPU? 
    // Let's assume params.offset_x is exactly that: (camera_origin.x - grid.min_x)
    
    let rel_x = px + params.offset_x;
    let rel_z = pz + params.offset_z;

    if (rel_x < 0.0 || rel_x >= params.size_x || rel_z < 0.0 || rel_z >= params.size_z) {
        return;
    }

    // Map to grid coordinates
    let u = rel_x / params.size_x;
    let v = rel_z / params.size_z;

    let grid_x = u32(u * f32(params.grid_size));
    let grid_y = u32(v * f32(params.grid_size));

    // Clamp for safety
    let gx = min(grid_x, params.grid_size - 1u);
    let gy = min(grid_y, params.grid_size - 1u);

    let grid_idx = gy * params.grid_size + gx;

    // Add density
    atomicAdd(&density_grid[grid_idx], 1u);
}
