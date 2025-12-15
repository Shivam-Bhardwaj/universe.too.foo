// Neural Planetarium Shader (prototype)
//
// Group 0: orbit params + packed residuals + neural weight buffer
// Group 1: global camera/time uniforms

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

@group(0) @binding(0) var<uniform> orbit: KeplerParams;
@group(0) @binding(1) var<storage, read> residuals: array<u32>;
@group(0) @binding(2) var<storage, read> neural_weights: array<f32>;

struct GlobalUniforms {
    view_proj: mat4x4<f32>,
    time: f32,
    _pad0: vec3<f32>,
}

@group(1) @binding(0) var<uniform> globals: GlobalUniforms;

const PI: f32 = 3.14159265359;

fn solve_eccentric_anomaly(M: f32, e: f32) -> f32 {
    var E: f32 = M;
    for (var it: i32 = 0; it < 5; it = it + 1) {
        let f = E - e * sin(E) - M;
        let df = 1.0 - e * cos(E);
        E = E - f / df;
    }
    return E;
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

fn get_orbital_position(t_days: f32) -> vec3<f32> {
    // Mean motion approximation (assuming a in AU and t in days)
    let a = orbit.semi_major_axis;
    let e = orbit.eccentricity;
    let n = 0.017202 / pow(a, 1.5);

    let M = orbit.mean_anomaly_0 + n * t_days;
    let E = solve_eccentric_anomaly(M, e);

    let x_orb = a * (cos(E) - e);
    let y_orb = a * sqrt(1.0 - e * e) * sin(E);

    return rotate_orbital_to_world(vec3<f32>(x_orb, y_orb, 0.0));
}

fn sign_extend_i16(x: u32) -> i32 {
    return (bitcast<i32>(x << 16u)) >> 16;
}

fn unpack_residual(packed_val: u32) -> vec2<f32> {
    // Matches Python `struct.pack('hh', radial, transverse)` on little-endian:
    // Low 16 = Radial, High 16 = Transverse
    let r_u = packed_val & 0xFFFFu;
    let t_u = (packed_val >> 16u) & 0xFFFFu;

    let r_q = sign_extend_i16(r_u);
    let t_q = sign_extend_i16(t_u);

    // Interpret residual_scale as units-per-LSB.
    return vec2<f32>(f32(r_q), f32(t_q)) * orbit.residual_scale;
}

fn apply_residuals(base_pos: vec3<f32>, t_days: f32) -> vec3<f32> {
    if (orbit.count == 0u) {
        return base_pos;
    }

    let idx_f = t_days;
    let idx_0 = u32(floor(idx_f)) % orbit.count;
    let idx_1 = (idx_0 + 1u) % orbit.count;
    let mix_factor = fract(idx_f);

    let res0 = unpack_residual(residuals[idx_0]);
    let res1 = unpack_residual(residuals[idx_1]);
    let res = mix(res0, res1, mix_factor);

    let r_dir = normalize(base_pos);
    // Simple transverse direction (perpendicular in XY); good enough for prototype
    let t_dir = normalize(vec3<f32>(-r_dir.y, r_dir.x, 0.0));

    return base_pos + r_dir * res.x + t_dir * res.y;
}

fn run_asteroid_decoder(pos: vec3<f32>) -> vec4<f32> {
    let w0 = neural_weights[0];
    let r = sin(pos.x * 10.0 + w0) * 0.5 + 0.5;
    let g = sin(pos.y * 10.0) * 0.5 + 0.5;
    let b = sin(pos.z * 10.0) * 0.5 + 0.5;
    return vec4<f32>(r, g, b, 1.0);
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @builtin(instance_index) instance_idx: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let t = globals.time + f32(in.instance_idx) * 10.0;

    let pos_kepler = get_orbital_position(t);
    let pos_center = apply_residuals(pos_kepler, t);

    let color = run_asteroid_decoder(pos_center);

    let world_pos = in.position * 0.01 + pos_center;
    out.clip_position = globals.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
