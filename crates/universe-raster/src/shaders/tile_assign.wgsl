// Compute shader: Assign 2D projected splats to tiles
//
// Each splat may overlap multiple tiles based on its bounding radius.
// This shader writes (tile_id, depth, splat_idx) keys for sorting.

const TILE_SIZE: u32 = 16u;

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

struct TileKey {
    key_high: u32,  // tile_id
    key_low: u32,   // depth bits
    splat_idx: u32,
    _pad: u32,
}

struct TileParams {
    num_tiles_x: u32,
    num_tiles_y: u32,
    screen_width: f32,
    screen_height: f32,
}

@group(0) @binding(0) var<storage, read> splats_2d: array<Splat2D>;
@group(0) @binding(1) var<storage, read_write> tile_keys: array<TileKey>;
@group(0) @binding(2) var<storage, read_write> key_count: atomic<u32>;
@group(0) @binding(3) var<uniform> params: TileParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&splats_2d) {
        return;
    }

    let splat = splats_2d[idx];

    // Skip culled splats (opacity <= 0)
    if splat.opacity <= 0.0 {
        return;
    }

    // Compute tile range this splat overlaps
    let min_x = splat.center.x - splat.radius;
    let max_x = splat.center.x + splat.radius;
    let min_y = splat.center.y - splat.radius;
    let max_y = splat.center.y + splat.radius;

    let min_tile_x = max(0, i32(floor(min_x / f32(TILE_SIZE))));
    let max_tile_x = min(i32(params.num_tiles_x) - 1, i32(floor(max_x / f32(TILE_SIZE))));
    let min_tile_y = max(0, i32(floor(min_y / f32(TILE_SIZE))));
    let max_tile_y = min(i32(params.num_tiles_y) - 1, i32(floor(max_y / f32(TILE_SIZE))));

    // Convert depth to u32 for sorting
    // We negate so that closer splats (higher depth in reverse-Z) sort first
    let depth_bits = bitcast<u32>(splat.depth);

    // Add entry for each overlapping tile
    for (var ty = min_tile_y; ty <= max_tile_y; ty++) {
        for (var tx = min_tile_x; tx <= max_tile_x; tx++) {
            let tile_id = u32(ty) * params.num_tiles_x + u32(tx);

            // Atomically allocate slot in output buffer
            let write_idx = atomicAdd(&key_count, 1u);

            // Write key-value pair
            tile_keys[write_idx].key_high = tile_id;
            tile_keys[write_idx].key_low = depth_bits;
            tile_keys[write_idx].splat_idx = idx;
            tile_keys[write_idx]._pad = 0u;
        }
    }
}
