// Neural Planetarium prototype shader
//
// Uses:
// - orbit metadata (KeplerParams) as a uniform
// - packed residuals (u32 stream) as a storage buffer
// - flattened neural weights (f32 stream) as a storage buffer
//
// NOTE: This is a prototype WGSL to validate the IO + binding pipeline.

// --- BINDINGS ---

// Group 0: Physics & Neural Data
struct KeplerParams {
    semi_major_axis: f32,
    eccentricity: f32,
    inclination: f32,
    arg_periapsis: f32,

    long_asc_node: f32,
    mean_anomaly_0: f32,
    residual_scale: f32,
    count: u32,

    // Match the Rust struct's 16-byte aligned size (48 bytes).
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> orbit: KeplerParams;
@group(0) @binding(1) var<storage, read> residuals: array<u32>;
@group(0) @binding(2) var<storage, read> neural_weights: array<f32>;

// Group 1: Global Scene Data (Camera, Time)
struct GlobalUniforms {
    view_proj: mat4x4<f32>,
    time: f32,
}
@group(1) @binding(0) var<uniform> globals: GlobalUniforms;

// --- CONSTANTS ---
const PI: f32 = 3.14159265359;

// --- 1. PHYSICS ENGINE: KEPLER SOLVER ---

fn solve_eccentric_anomaly(M: f32, e: f32) -> f32 {
    // Newton-Raphson Iteration
    // M = E - e*sin(E)
    var E: f32 = M;
    for (var it: i32 = 0; it < 5; it = it + 1) {
        let f = E - e * sin(E) - M;
        let df = 1.0 - e * cos(E);
        E = E - f / df;
    }
    return E;
}

struct OrbitalState {
    pos: vec3<f32>,
    vel: vec3<f32>,
}

fn rotate_orbital_to_world(v: vec3<f32>) -> vec3<f32> {
    let w = orbit.arg_periapsis;
    let O = orbit.long_asc_node;
    let i = orbit.inclination;

    let cw = cos(w);
    let sw = sin(w);
    let cO = cos(O);
    let sO = sin(O);
    let ci = cos(i);
    let si = sin(i);

    // Rotate by w (about +Z)
    let x1 = v.x * cw - v.y * sw;
    let y1 = v.x * sw + v.y * cw;

    // Rotate by i (about +X)
    let x2 = x1;
    let y2 = y1 * ci;
    let z2 = y1 * si;

    // Rotate by O (about +Z)
    let x3 = x2 * cO - y2 * sO;
    let y3 = x2 * sO + y2 * cO;

    return vec3<f32>(x3, y3, z2);
}

fn get_orbital_state(t_days: f32) -> OrbitalState {
    // Mean Anomaly
    // n approx for Solar System (assuming a in AU, t in Days)
    let a = orbit.semi_major_axis;
    let e = orbit.eccentricity;
    let n = 0.017202 / pow(a, 1.5);

    let M = orbit.mean_anomaly_0 + n * t_days;

    // Eccentric Anomaly
    let E = solve_eccentric_anomaly(M, e);

    let cE = cos(E);
    let sE = sin(E);
    let sqrt1me2 = sqrt(1.0 - e * e);

    // Perifocal position (AU)
    let x_orb = a * (cE - e);
    let y_orb = a * sqrt1me2 * sE;

    // Perifocal velocity (AU/day)
    // E_dot = n / (1 - e cos E)
    let Edot = n / max(1e-6, (1.0 - e * cE));
    let vx_orb = -a * sE * Edot;
    let vy_orb = a * sqrt1me2 * cE * Edot;

    let pos = rotate_orbital_to_world(vec3<f32>(x_orb, y_orb, 0.0));
    let vel = rotate_orbital_to_world(vec3<f32>(vx_orb, vy_orb, 0.0));

    return OrbitalState(pos, vel);
}

// --- 2. PHYSICS ENGINE: SHANNON DECOMPRESSOR ---

fn sign_extend_i16(x: u32) -> i32 {
    // Convert a 16-bit two's-complement value (stored in low bits) to i32.
    // This uses arithmetic right shift after bitcast.
    return (bitcast<i32>(x << 16u)) >> 16;
}

fn unpack_residual(packed_val: u32) -> vec2<f32> {
    // Our binary format packs two int16 into one u32 (little-endian):
    // low 16 bits = radial, high 16 bits = transverse.
    let r_u = packed_val & 0xFFFFu;
    let t_u = (packed_val >> 16u) & 0xFFFFu;

    let r_q = sign_extend_i16(r_u);
    let t_q = sign_extend_i16(t_u);

    // Scale back to world units. `residual_scale` is interpreted as units per LSB.
    return vec2<f32>(f32(r_q), f32(t_q)) * orbit.residual_scale;
}

fn apply_residuals(state: OrbitalState, t_days: f32) -> vec3<f32> {
    if (orbit.count == 0u) {
        return state.pos;
    }

    // Linear interpolation between residual samples.
    // Assumes residuals are sampled at 1 unit per day. Adjust by changing idx_f.
    let idx_f = t_days;
    let idx_0 = u32(floor(idx_f)) % orbit.count;
    let idx_1 = (idx_0 + 1u) % orbit.count;
    let mix_factor = fract(idx_f);

    let res_0 = unpack_residual(residuals[idx_0]);
    let res_1 = unpack_residual(residuals[idx_1]);
    let res_rt = mix(res_0, res_1, mix_factor);

    // Reconstruct residual vector from radial + transverse components.
    let r_hat = normalize(state.pos);
    let v_proj = state.vel - dot(state.vel, r_hat) * r_hat;
    let t_hat = normalize(v_proj);

    return state.pos + r_hat * res_rt.x + t_hat * res_rt.y;
}

// --- 3. VISUAL ENGINE: NEURAL DECODER (MLP) ---

fn relu(x: f32) -> f32 {
    return max(0.0, x);
}

// Runs the MLP for a single instance
// Input: Position (3 floats) + Code (32 floats, hardcoded 0.0 for now)
fn run_asteroid_decoder(pos: vec3<f32>) -> vec4<f32> {
    // Prototype: just prove we can read the buffer.
    let w0 = neural_weights[0];

    let r = sin(pos.x * 10.0 + w0) * 0.5 + 0.5;
    let g = sin(pos.y * 10.0) * 0.5 + 0.5;
    let b = sin(pos.z * 10.0) * 0.5 + 0.5;

    return vec4<f32>(r, g, b, 1.0);
}

// --- 4. MAIN SHADER STAGES ---

struct VertexInput {
    @location(0) position: vec3<f32>, // Local mesh position (sphere)
    @builtin(instance_index) instance_idx: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // A. CALCULATE ORBIT
    // Add instance offset to time to desync asteroids.
    let t = globals.time + f32(in.instance_idx) * 10.0;

    let state = get_orbital_state(t);
    let pos_final_center = apply_residuals(state, t);

    // B. NEURAL APPEARANCE
    let neural_color = run_asteroid_decoder(pos_final_center);

    // C. TRANSFORM
    let world_pos = in.position * 0.01 + pos_final_center;
    out.clip_position = globals.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = neural_color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
