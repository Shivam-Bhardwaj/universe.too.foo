// Compute shader: Per-tile Gaussian rasterization with alpha blending
//
// Each workgroup (16x16 threads) processes one tile. Splats are loaded
// in batches to shared memory, then each thread composites its pixel
// using front-to-back alpha blending with early termination.

const TILE_SIZE: u32 = 16u;
const BATCH_SIZE: u32 = 256u;
const ALPHA_THRESHOLD: f32 = 0.99;

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
    key_high: u32,
    key_low: u32,
    splat_idx: u32,
    _pad: u32,
}

struct TileRange {
    start: u32,
    count: u32,
}

@group(0) @binding(0) var<storage, read> splats_2d: array<Splat2D>;
@group(0) @binding(1) var<storage, read> sorted_keys: array<TileKey>;
@group(0) @binding(2) var<storage, read> tile_ranges: array<TileRange>;
@group(0) @binding(3) var output: texture_storage_2d<rgba8unorm, write>;

// Shared memory for batch loading splats
var<workgroup> shared_splats: array<Splat2D, BATCH_SIZE>;
// Track how many threads are done (for early termination optimization)
var<workgroup> done_count: atomic<u32>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let output_dims = textureDimensions(output);
    let num_tiles_x = (output_dims.x + TILE_SIZE - 1u) / TILE_SIZE;

    // Compute tile ID and pixel coordinates
    let tile_id = wg_id.y * num_tiles_x + wg_id.x;
    let pixel_x = wg_id.x * TILE_SIZE + local_id.x;
    let pixel_y = wg_id.y * TILE_SIZE + local_id.y;

    // Pixel center in continuous coordinates
    let pixel = vec2<f32>(f32(pixel_x) + 0.5, f32(pixel_y) + 0.5);

    // Get tile range
    let range = tile_ranges[tile_id];

    // Initialize per-thread accumulator
    var color_accum = vec3<f32>(0.0);
    var alpha_accum: f32 = 0.0;
    var is_done = false;

    // Process splats in batches
    var batch_start = range.start;
    let batch_end = range.start + range.count;

    while batch_start < batch_end {
        // Reset done count for this batch
        if local_idx == 0u {
            atomicStore(&done_count, 0u);
        }
        workgroupBarrier();

        // Collaboratively load batch into shared memory
        let batch_size = min(BATCH_SIZE, batch_end - batch_start);
        if local_idx < batch_size {
            let key_idx = batch_start + local_idx;
            let splat_idx = sorted_keys[key_idx].splat_idx;
            shared_splats[local_idx] = splats_2d[splat_idx];
        }
        workgroupBarrier();

        // Process batch (front-to-back, sorted by depth)
        if !is_done {
            for (var i = 0u; i < batch_size; i++) {
                if alpha_accum >= ALPHA_THRESHOLD {
                    is_done = true;
                    atomicAdd(&done_count, 1u);
                    break;
                }

                let splat = shared_splats[i];

                // Distance from pixel to splat center
                let d = pixel - splat.center;

                // Evaluate 2D Gaussian using conic form
                // G(dx, dy) = exp(-0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2))
                let power = -0.5 * (
                    splat.conic.x * d.x * d.x +
                    2.0 * splat.conic.y * d.x * d.y +
                    splat.conic.z * d.y * d.y
                );

                // Cutoff at ~3 sigma (power < -4.5 means < 0.01 contribution)
                if power > -4.5 {
                    let gaussian = exp(power);
                    let alpha = min(0.99, gaussian * splat.opacity);

                    if alpha > 1.0 / 255.0 { // Skip negligible contributions
                        // Front-to-back alpha blending
                        // C_out = C_out + (1 - alpha_out) * alpha * C_in
                        // alpha_out = alpha_out + (1 - alpha_out) * alpha
                        let weight = alpha * (1.0 - alpha_accum);
                        color_accum += splat.color * weight;
                        alpha_accum += weight;
                    }
                }
            }
        }

        workgroupBarrier();

        // Early termination: if all threads are done, skip remaining batches
        if atomicLoad(&done_count) == TILE_SIZE * TILE_SIZE {
            break;
        }

        batch_start += BATCH_SIZE;
    }

    // Write output pixel
    if pixel_x < output_dims.x && pixel_y < output_dims.y {
        // Background color (dark blue for space)
        let bg_color = vec3<f32>(0.0, 0.0, 0.02);

        // Blend with background
        let final_color = color_accum + bg_color * (1.0 - alpha_accum);

        // Gamma correction (sRGB)
        let gamma_color = pow(final_color, vec3<f32>(1.0 / 2.2));

        // DEBUG: Draw a perfect circle at screen center to test aspect ratio
        let center = vec2<f32>(f32(output_dims.x) * 0.5, f32(output_dims.y) * 0.5);
        let dist = length(pixel - center);
        var debug_color = gamma_color;
        if dist < 100.0 {
            debug_color = vec3<f32>(1.0, 0.0, 0.0); // Red circle
        } else if dist < 102.0 {
            debug_color = vec3<f32>(1.0, 1.0, 1.0); // White outline
        }

        textureStore(
            output,
            vec2<i32>(i32(pixel_x), i32(pixel_y)),
            vec4<f32>(debug_color, 1.0)
        );
    }
}
