/// Simple procedural mesh generation utilities.

#[derive(Clone, Copy, Debug)]
pub struct SphereOptions {
    pub radius: f32,
    pub stacks: u32,
    pub slices: u32,
}

impl Default for SphereOptions {
    fn default() -> Self {
        Self {
            radius: 1.0,
            stacks: 32,
            slices: 64,
        }
    }
}

/// Generate a UV sphere.
///
/// Returns `(positions, indices)` where:
/// - `positions` is a list of `[x,y,z]` vertices
/// - `indices` is a triangle list (CCW winding)
pub fn generate_uv_sphere(opts: SphereOptions) -> (Vec<[f32; 3]>, Vec<u32>) {
    let stacks = opts.stacks.max(2);
    let slices = opts.slices.max(3);

    let mut positions = Vec::with_capacity(((stacks + 1) * (slices + 1)) as usize);

    for stack in 0..=stacks {
        let v = stack as f32 / stacks as f32;
        let phi = v * std::f32::consts::PI;

        let sin_phi = phi.sin();
        let cos_phi = phi.cos();

        for slice in 0..=slices {
            let u = slice as f32 / slices as f32;
            let theta = u * (2.0 * std::f32::consts::PI);

            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            let x = opts.radius * sin_phi * cos_theta;
            let y = opts.radius * cos_phi;
            let z = opts.radius * sin_phi * sin_theta;

            positions.push([x, y, z]);
        }
    }

    let ring = slices + 1;
    let mut indices = Vec::with_capacity((stacks * slices * 6) as usize);

    for stack in 0..stacks {
        for slice in 0..slices {
            let i0 = stack * ring + slice;
            let i1 = i0 + 1;
            let i2 = (stack + 1) * ring + slice;
            let i3 = i2 + 1;

            // Two triangles per quad (CCW)
            indices.push(i0);
            indices.push(i2);
            indices.push(i1);

            indices.push(i1);
            indices.push(i2);
            indices.push(i3);
        }
    }

    (positions, indices)
}



