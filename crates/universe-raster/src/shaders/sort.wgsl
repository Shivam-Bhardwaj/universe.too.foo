// Compute shader: Bitonic sort for tile keys
//
// This implements a parallel bitonic merge sort on the GPU.
// Sorts by (tile_id, depth) to group splats by tile and order by depth within.
//
// Bitonic sort has O(n log^2 n) work but is highly parallel and predictable,
// making it efficient on GPUs for moderate sizes (up to ~1M elements).

struct TileKey {
    key_high: u32,  // tile_id (primary sort key)
    key_low: u32,   // depth bits (secondary sort key)
    splat_idx: u32,
    _pad: u32,
}

struct SortParams {
    num_elements: u32,
    stage: u32,      // Current stage of bitonic sort
    substage: u32,   // Current substage within stage
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> keys: array<TileKey>;
@group(0) @binding(1) var<uniform> params: SortParams;

// Compare function: returns true if a should come before b
fn compare(a: TileKey, b: TileKey) -> bool {
    // Primary: sort by tile_id ascending
    if a.key_high != b.key_high {
        return a.key_high < b.key_high;
    }
    // Secondary: sort by depth ascending (front-to-back for reverse-Z)
    return a.key_low < b.key_low;
}

@compute @workgroup_size(256)
fn bitonic_sort_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    // Compute partner index for this stage/substage
    let stage = params.stage;
    let substage = params.substage;

    // Distance to partner element
    let offset = 1u << substage;

    // Block size for this stage
    let block_size = 1u << (stage + 1u);

    // Position within block
    let block_idx = idx / (block_size / 2u);
    let local_idx = idx % (block_size / 2u);

    // Determine sort direction (ascending or descending for this block)
    let ascending = (block_idx % 2u) == 0u;

    // Compute actual indices
    let base = block_idx * block_size;

    // In the bitonic merge, we compare elements offset apart
    let pos_in_substage = local_idx % offset;
    let group = local_idx / offset;

    let i = base + group * 2u * offset + pos_in_substage;
    let j = i + offset;

    if j >= params.num_elements {
        return;
    }

    let a = keys[i];
    let b = keys[j];

    // Determine if we should swap
    let a_less_b = compare(a, b);
    let should_swap = (ascending && !a_less_b) || (!ascending && a_less_b);

    if should_swap {
        keys[i] = b;
        keys[j] = a;
    }
}

// Alternative: Local bitonic sort within workgroup using shared memory
// This is more efficient for the initial sorting passes

var<workgroup> local_keys: array<TileKey, 512>;

@compute @workgroup_size(256)
fn bitonic_sort_local(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let global_idx = gid.x;
    let local_idx = lid.x;

    // Load two elements per thread into shared memory
    let idx0 = global_idx * 2u;
    let idx1 = global_idx * 2u + 1u;

    if idx0 < params.num_elements {
        local_keys[local_idx * 2u] = keys[idx0];
    }
    if idx1 < params.num_elements {
        local_keys[local_idx * 2u + 1u] = keys[idx1];
    }

    workgroupBarrier();

    // Perform bitonic sort within shared memory
    for (var stage = 0u; stage < 9u; stage++) { // log2(512) = 9
        for (var substage = stage; substage >= 0u; substage--) {
            let offset = 1u << substage;
            let block_size = 1u << (stage + 1u);

            let local_block_idx = (local_idx * 2u) / block_size;
            let ascending = (local_block_idx % 2u) == 0u;

            // Each thread handles one comparison
            let pos = local_idx;
            let group = pos / offset;
            let pos_in_group = pos % offset;

            let i = group * 2u * offset + pos_in_group;
            let j = i + offset;

            if j < 512u {
                let a = local_keys[i];
                let b = local_keys[j];

                let a_less_b = compare(a, b);
                let block_asc = ((i / block_size) % 2u) == 0u;
                let should_swap = (block_asc && !a_less_b) || (!block_asc && a_less_b);

                if should_swap {
                    local_keys[i] = b;
                    local_keys[j] = a;
                }
            }

            workgroupBarrier();

            if substage == 0u {
                break;
            }
        }
    }

    // Write back to global memory
    if idx0 < params.num_elements {
        keys[idx0] = local_keys[local_idx * 2u];
    }
    if idx1 < params.num_elements {
        keys[idx1] = local_keys[local_idx * 2u + 1u];
    }
}
