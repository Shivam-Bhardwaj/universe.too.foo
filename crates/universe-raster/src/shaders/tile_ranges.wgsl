// Compute shader: Compute start/count for each tile after sorting
//
// After the tile keys are sorted by (tile_id, depth), this shader
// scans through and records where each tile's data begins and how
// many entries it has.

struct TileKey {
    key_high: u32,  // tile_id
    key_low: u32,   // depth bits
    splat_idx: u32,
    _pad: u32,
}

struct TileRange {
    start: u32,
    count: u32,
}

struct Params {
    num_keys: u32,
    num_tiles: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> sorted_keys: array<TileKey>;
@group(0) @binding(1) var<storage, read_write> tile_ranges: array<TileRange>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_keys {
        return;
    }

    let tile_id = sorted_keys[idx].key_high;

    // Check if this is the start of a new tile group
    var is_start = false;
    if idx == 0u {
        is_start = true;
    } else {
        let prev_tile_id = sorted_keys[idx - 1u].key_high;
        if tile_id != prev_tile_id {
            is_start = true;
        }
    }

    if is_start {
        tile_ranges[tile_id].start = idx;
    }

    // Check if this is the end of a tile group
    var is_end = false;
    if idx == params.num_keys - 1u {
        is_end = true;
    } else {
        let next_tile_id = sorted_keys[idx + 1u].key_high;
        if tile_id != next_tile_id {
            is_end = true;
        }
    }

    if is_end {
        // Count = end_idx - start_idx + 1
        // Since we write start when we see the first element, we need to
        // compute count carefully. We know this is the last element for this tile.
        // The count is: current_idx - start + 1
        // But we need to read the start we wrote, which could race.
        // Instead, we compute count in a second pass or use atomics.

        // Simple approach: compute count based on position
        // Find where this tile started by looking backwards
        var start_idx = idx;
        while start_idx > 0u && sorted_keys[start_idx - 1u].key_high == tile_id {
            start_idx -= 1u;
        }
        tile_ranges[tile_id].count = idx - start_idx + 1u;
    }
}

// Alternative: Single-pass with atomics for counting
// This is simpler but may have more contention
@compute @workgroup_size(256)
fn main_atomic(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_keys {
        return;
    }

    let tile_id = sorted_keys[idx].key_high;

    // Check if this is the start of a new tile group
    if idx == 0u || sorted_keys[idx - 1u].key_high != tile_id {
        tile_ranges[tile_id].start = idx;
    }

    // Increment count for this tile
    // Note: This requires tile_ranges.count to be atomic, which it isn't in this struct
    // So we use the scan-based approach in main() instead
}
