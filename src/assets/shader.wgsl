// Neural Planetarium Shader (Production)
//
// Group 0: Orbit Params (Uniform) + Residuals (Storage) + Neural Weights (Storage)
// Group 1: Global Scene Data (Uniform)

// --- STRUCTS ---
struct KeplerParams {
    semi_major_axis: f32,
    eccentricity: f32,
    inclination: f32,
    arg_periapsis: f32,
    long_asc_node: f32,
    mean_anomaly_0: f32,
    residual_scale: f32,
    count: u32,
}

struct GlobalUniforms {
    view_proj: mat4x4<f32>,
    time: f32,
    _pad0: vec3<f32>,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @builtin(instance_index) instance_idx: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

// --- BINDINGS ---
@group(0) @binding(0) var<uniform> orbit: KeplerParams;
@group(0) @binding(1) var<storage, read> residuals: array<u32>;
@group(0) @binding(2) var<storage, read> neural_weights: array<f32>; // The Brain

@group(1) @binding(0) var<uniform> globals: GlobalUniforms;

// --- CONSTANTS ---
const PI: f32 = 3.14159265;

// --- NEURAL NETWORK LOGIC ---

fn relu(x: f32) -> f32 {
    return max(0.0, x);
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// Manual MLP Inference: Input(35) -> L1(64) -> L2(64) -> Out(4)
// Offsets must match 'data_compiler.py' flattening order
const L1_W_START: u32 = 0u;
const L1_B_START: u32 = 2240u; // 35 * 64
const L2_W_START: u32 = 2304u; // 2240 + 64
const L2_B_START: u32 = 6400u; // 2304 + (64*64)
const OUT_W_START: u32 = 6464u;
const OUT_B_START: u32 = 6720u;

fn run_neural_net(pos: vec3<f32>) -> vec4<f32> {
    // 1. LAYER 1: Input (35) -> Hidden (64)
    // We strictly use the first 3 inputs (x,y,z) as the feature vector for now.
    // In the future, inputs 3..34 would be the "Instance Latent Code".
    var h1: array<f32, 64>;

    for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        var sum = neural_weights[L1_B_START + i]; // Bias

        // Weight * Input (Unrolled for first 3 inputs)
        let w_base = L1_W_START + (i * 35u);
        sum += pos.x * neural_weights[w_base + 0u];
        sum += pos.y * neural_weights[w_base + 1u];
        sum += pos.z * neural_weights[w_base + 2u];
        // Note: Inputs 3..34 are assumed 0.0 for this prototype

        h1[i] = relu(sum);
    }

    // 2. LAYER 2: Hidden (64) -> Hidden (64)
    var h2: array<f32, 64>;
    for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        var sum = neural_weights[L2_B_START + i];
        let w_base = L2_W_START + (i * 64u);

        for (var j: u32 = 0u; j < 64u; j = j + 1u) {
            sum += h1[j] * neural_weights[w_base + j];
        }
        h2[i] = relu(sum);
    }

    // 3. OUTPUT LAYER: Hidden (64) -> Output (4)
    // RGB (3) + Displacement (1)
    var output: vec4<f32> = vec4<f32>(0.0);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        var sum = neural_weights[OUT_B_START + i];
        let w_base = OUT_W_START + (i * 64u);

        for (var j: u32 = 0u; j < 64u; j = j + 1u) {
            sum += h2[j] * neural_weights[w_base + j];
        }

        // Output activation
        if (i < 3u) {
            output[i] = sigmoid(sum); // RGB must be 0.0-1.0
        } else {
            output[i] = sum; // Displacement is unbounded
        }
    }

    return output;
}

// --- PHYSICS LOGIC ---

fn sign_extend_i16(x: u32) -> i32 {
    return (bitcast<i32>(x << 16u)) >> 16;
}

fn unpack_residual(packed_val: u32) -> vec2<f32> {
    let r_u = packed_val & 0xFFFFu;
    let t_u = (packed_val >> 16u) & 0xFFFFu;
    let r_q = sign_extend_i16(r_u);
    let t_q = sign_extend_i16(t_u);
    return vec2<f32>(f32(r_q), f32(t_q)) * orbit.residual_scale;
}

fn solve_kepler(M: f32, e: f32) -> f32 {
    var E: f32 = M;
    for (var it: i32 = 0; it < 5; it = it + 1) {
        E = E - (E - e * sin(E) - M) / (1.0 - e * cos(E));
    }
    return E;
}

fn get_orbit_pos_with_residuals(t: f32) -> vec3<f32> {
    // A. Kepler
    let n = 0.017202 / pow(orbit.semi_major_axis, 1.5);
    let M = orbit.mean_anomaly_0 + n * t;
    let E = solve_kepler(M, orbit.eccentricity);

    let a = orbit.semi_major_axis;
    let e = orbit.eccentricity;
    let x_orb = a * (cos(E) - e);
    let y_orb = a * sqrt(1.0 - e * e) * sin(E);

    // B. Rotate to 3D (prototype: planar)
    let w = orbit.arg_periapsis;
    let x_w = x_orb * cos(w) - y_orb * sin(w);
    let y_w = x_orb * sin(w) + y_orb * cos(w);
    let base_pos = vec3<f32>(x_w, y_w, 0.0);

    // C. Apply Shannon residuals
    if (orbit.count == 0u) {
        return base_pos;
    }

    let idx_f = t;
    let idx_0 = u32(floor(idx_f)) % orbit.count;
    let idx_1 = (idx_0 + 1u) % orbit.count;
    let mix_factor = fract(idx_f);

    let res0 = unpack_residual(residuals[idx_0]);
    let res1 = unpack_residual(residuals[idx_1]);
    let res = mix(res0, res1, mix_factor);

    let r_dir = normalize(base_pos);
    let t_dir = vec3<f32>(-r_dir.y, r_dir.x, 0.0);
    return base_pos + (r_dir * res.x) + (t_dir * res.y);
}

// --- MAIN ---

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // 1. NEURAL DEFORMATION
    let neural_data = run_neural_net(in.position);
    let rgb = neural_data.xyz;
    let displacement = neural_data.w;

    // Displace vertex along its normal
    let local_pos_displaced = in.position + (normalize(in.position) * displacement * 0.1);

    // 2. ORBITAL POSITION
    let t = globals.time + f32(in.instance_idx) * 50.0;
    let world_center = get_orbit_pos_with_residuals(t);

    // 3. FINAL TRANSFORM
    let world_pos = (local_pos_displaced * 0.02) + world_center;

    out.clip_position = globals.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = rgb;
    out.normal = normalize(local_pos_displaced);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let diff = max(dot(in.normal, light_dir), 0.2);
    return vec4<f32>(in.color * diff, 1.0);
}
