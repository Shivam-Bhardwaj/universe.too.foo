
import { describe, it, expect } from 'vitest';

// We'll define the functions here for testing first, then move to a utility file/class
type Vec3 = { x: number; y: number; z: number };
type Spherical = { r: number; theta: number; phi: number };

function toSpherical(v: Vec3): Spherical {
    const r = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (r === 0) return { r: 0, theta: 0, phi: 0 };
    // theta = azimuth (angle in X-Z plane). Check atan2 logic.
    // If we use standard physics convention:
    // x = r sin(phi) cos(theta)
    // z = r sin(phi) sin(theta)? No, typical 3D graphics is Y-up.
    // Y is Up. X is Right. Z is Back? 
    // Let's match typical Y-up system:
    // r = dist
    // theta (azimuth) = atan2(x, z) or atan2(x, -z)?
    // phi (polar) = asin(y / r)  (Latitude-like, -PI/2 to PI/2)
    // OR phi = acos(y / r) (Co-latitude, 0 to PI)

    // Let's use Latitude-style for consistency with "Pitch":
    // polar (phi) = asin(y/r)
    // azimuth (theta) = atan2(x, z) 

    return {
        r,
        theta: Math.atan2(v.x, v.z), // Check conventions!
        phi: Math.asin(v.y / r)
    };
}

function toCartesian(s: Spherical): Vec3 {
    // y = r * sin(phi)
    // horizontal_r = r * cos(phi)
    // x = horizontal_r * sin(theta)
    // z = horizontal_r * cos(theta)

    const cosPhi = Math.cos(s.phi);
    const sinPhi = Math.sin(s.phi);
    const cosTheta = Math.cos(s.theta);
    const sinTheta = Math.sin(s.theta);

    return {
        x: s.r * cosPhi * sinTheta,
        y: s.r * sinPhi,
        z: s.r * cosPhi * cosTheta
    };
}

describe('Spherical Coordinates', () => {
    it('should convert (0,0,10) to spherical', () => {
        const c = { x: 0, y: 0, z: 10 };
        const s = toSpherical(c);
        // r=10, phi=0 (equator)
        // theta: atan2(0, 10). If atan2(y, x), here atan2(x, z) -> atan2(0, 10) = 0.
        expect(s.r).toBeCloseTo(10);
        expect(s.phi).toBeCloseTo(0);
        expect(s.theta).toBeCloseTo(0);
    });

    it('should convert (10,0,0) to spherical', () => {
        const c = { x: 10, y: 0, z: 0 };
        const s = toSpherical(c);
        // r=10
        // theta: atan2(10, 0) = PI/2
        expect(s.r).toBeCloseTo(10);
        expect(s.theta).toBeCloseTo(Math.PI / 2);
    });

    it('should round-trip Cartesian -> Spherical -> Cartesian', () => {
        const orig = { x: 100, y: 50, z: -25 };
        const s = toSpherical(orig);
        const res = toCartesian(s);

        expect(res.x).toBeCloseTo(orig.x);
        expect(res.y).toBeCloseTo(orig.y);
        expect(res.z).toBeCloseTo(orig.z);
    });
});
