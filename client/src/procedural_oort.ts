/**
 * Procedural Oort Cloud Generator
 *
 * Generates comet-like objects in the Oort Cloud (2,000 - 100,000 AU)
 * using deterministic seeding for consistent visualization.
 */

export interface OortCloudParams {
    innerRadiusAU: number;      // 2,000 AU
    outerRadiusAU: number;      // 100,000 AU
    seed: number;               // Deterministic seed
    objectCount: number;        // How many to generate
}

export interface ProceduralObject {
    position: { x: number; y: number; z: number };  // meters
    radius: number;                                   // meters
    brightness: number;                               // 0-1
    color: [number, number, number];                 // RGB
}

/**
 * Simple LCG random number generator for deterministic generation
 */
class SeededRandom {
    private seed: number;

    constructor(seed: number) {
        this.seed = seed % 2147483647;
        if (this.seed <= 0) this.seed += 2147483646;
    }

    next(): number {
        this.seed = (this.seed * 16807) % 2147483647;
        return (this.seed - 1) / 2147483646;
    }

    range(min: number, max: number): number {
        return min + this.next() * (max - min);
    }
}

/**
 * Generate Oort Cloud objects visible from camera position
 */
export function generateOortCloud(params: OortCloudParams): ProceduralObject[] {
    const AU_M = 1.496e11;
    const innerRadius = params.innerRadiusAU * AU_M;
    const outerRadius = params.outerRadiusAU * AU_M;

    const rng = new SeededRandom(params.seed);
    const objects: ProceduralObject[] = [];

    // Generate objects in spherical shell
    for (let i = 0; i < params.objectCount; i++) {
        // Random spherical coordinates (uniform on sphere)
        const u = rng.next();
        const v = rng.next();

        const theta = 2 * Math.PI * u;           // Azimuthal angle
        const phi = Math.acos(2 * v - 1);        // Polar angle (uniform distribution)

        // Random radius (power law distribution: more objects at outer edge)
        const rPower = rng.range(0, 1);
        const r = innerRadius * Math.pow(outerRadius / innerRadius, rPower);

        // Convert to Cartesian
        const sinPhi = Math.sin(phi);
        const x = r * sinPhi * Math.cos(theta);
        const y = r * sinPhi * Math.sin(theta);
        const z = r * Math.cos(phi);

        // Object size: 1-100 km (comets and planetesimals)
        const radius = rng.range(1e3, 1e5);

        // Brightness based on size and distance
        const apparentBrightness = Math.min(1.0, radius / 1e4);

        // Color: Icy bodies are bluish-white to gray
        const iceiness = rng.range(0.7, 1.0);
        const color: [number, number, number] = [
            iceiness,
            iceiness * 0.95,
            iceiness,
        ];

        objects.push({
            position: { x, y, z },
            radius,
            brightness: apparentBrightness * 0.5,  // Dim overall
            color,
        });
    }

    return objects;
}

/**
 * Importance sampling: generate more objects near camera view direction
 */
export function generateOortCloudViewBiased(
    params: OortCloudParams,
    _cameraPos: { x: number; y: number; z: number },
    cameraDir: { x: number; y: number; z: number },
    fovRadians: number
): ProceduralObject[] {
    const AU_M = 1.496e11;
    const innerRadius = params.innerRadiusAU * AU_M;
    const outerRadius = params.outerRadiusAU * AU_M;

    const rng = new SeededRandom(params.seed);
    const objects: ProceduralObject[] = [];

    // Normalize camera direction
    const dirLen = Math.sqrt(cameraDir.x ** 2 + cameraDir.y ** 2 + cameraDir.z ** 2);
    const dir = {
        x: cameraDir.x / dirLen,
        y: cameraDir.y / dirLen,
        z: cameraDir.z / dirLen,
    };

    // Generate with bias toward view direction
    for (let i = 0; i < params.objectCount; i++) {
        let accepted = false;
        let x = 0, y = 0, z = 0, r = 0;

        // Rejection sampling: prefer objects in view cone
        while (!accepted) {
            const u = rng.next();
            const v = rng.next();

            const theta = 2 * Math.PI * u;
            const phi = Math.acos(2 * v - 1);

            const rPower = rng.range(0, 1);
            r = innerRadius * Math.pow(outerRadius / innerRadius, rPower);

            const sinPhi = Math.sin(phi);
            x = r * sinPhi * Math.cos(theta);
            y = r * sinPhi * Math.sin(theta);
            z = r * Math.cos(phi);

            // Dot product with view direction
            const dot = (x * dir.x + y * dir.y + z * dir.z) / r;
            const angleToCam = Math.acos(Math.max(-1, Math.min(1, dot)));

            // Accept if within expanded FOV (3x for context)
            const acceptProbability = angleToCam < fovRadians * 3 ? 1.0 : 0.1;

            if (rng.next() < acceptProbability) {
                accepted = true;
            }
        }

        const radius = rng.range(1e3, 1e5);
        const apparentBrightness = Math.min(1.0, radius / 1e4);
        const iceiness = rng.range(0.7, 1.0);

        objects.push({
            position: { x, y, z },
            radius,
            brightness: apparentBrightness * 0.5,
            color: [iceiness, iceiness * 0.95, iceiness],
        });
    }

    return objects;
}
