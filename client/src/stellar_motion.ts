/**
 * Stellar Proper Motion System
 *
 * Simulates stellar motion over time based on:
 * - Proper motion (tangential velocity)
 * - Radial velocity (motion toward/away from Sun)
 * - Galactic rotation (for very long timescales)
 */

export interface StellarVelocity {
    vx: number;  // m/s in ecliptic X
    vy: number;  // m/s in ecliptic Y
    vz: number;  // m/s in ecliptic Z
}

/**
 * Estimate stellar velocity based on position and distance
 * Uses statistical models for typical stellar kinematics
 */
export function estimateStellarVelocity(
    posX: number,
    posY: number,
    posZ: number,
    seed: number
): StellarVelocity {
    const dist = Math.sqrt(posX * posX + posY * posY + posZ * posZ);

    // Simple LCG for deterministic "randomness" per star
    const hash = (seed * 16807) % 2147483647;
    const random = (hash - 1) / 2147483646;

    // Typical stellar velocities in solar neighborhood:
    // - Radial: ~20 km/s dispersion
    // - Tangential: ~25 km/s dispersion
    // Scale with distance (farther stars have less precise velocities)

    const velocityScale = Math.min(1.0, dist / 1e17);  // Normalize to ~3 ly

    // Random velocity components (km/s converted to m/s)
    const vRadial = (random - 0.5) * 40000 * velocityScale;        // -20 to +20 km/s
    const vTangent1 = ((hash % 1000) / 1000 - 0.5) * 50000 * velocityScale;  // Tangential
    const vTangent2 = ((hash % 100) / 100 - 0.5) * 50000 * velocityScale;

    // Convert to Cartesian velocity
    // Radial component: along position vector
    const r = Math.max(1, dist);
    const radialDir = { x: posX / r, y: posY / r, z: posZ / r };

    // Tangential components: perpendicular to radial
    // Use two orthogonal tangent directions
    const tangent1 = {
        x: -posY,
        y: posX,
        z: 0,
    };
    const t1Len = Math.sqrt(tangent1.x ** 2 + tangent1.y ** 2 + tangent1.z ** 2);
    if (t1Len > 0) {
        tangent1.x /= t1Len;
        tangent1.y /= t1Len;
        tangent1.z /= t1Len;
    }

    const tangent2 = {
        x: radialDir.y * tangent1.z - radialDir.z * tangent1.y,
        y: radialDir.z * tangent1.x - radialDir.x * tangent1.z,
        z: radialDir.x * tangent1.y - radialDir.y * tangent1.x,
    };

    return {
        vx: radialDir.x * vRadial + tangent1.x * vTangent1 + tangent2.x * vTangent2,
        vy: radialDir.y * vRadial + tangent1.y * vTangent1 + tangent2.y * vTangent2,
        vz: radialDir.z * vRadial + tangent1.z * vTangent1 + tangent2.z * vTangent2,
    };
}

/**
 * Propagate star position to a given epoch
 */
export function propagateStarPosition(
    basePos: { x: number; y: number; z: number },
    velocity: StellarVelocity,
    baseEpochJD: number,
    targetEpochJD: number
): { x: number; y: number; z: number } {
    const deltaYears = (targetEpochJD - baseEpochJD) / 365.25;
    const deltaSeconds = deltaYears * 365.25 * 86400;

    // Simple linear propagation (good enough for ±100,000 years at stellar distances)
    return {
        x: basePos.x + velocity.vx * deltaSeconds,
        y: basePos.y + velocity.vy * deltaSeconds,
        z: basePos.z + velocity.vz * deltaSeconds,
    };
}

/**
 * Apply galactic rotation for very long timescales (>10,000 years)
 */
export function applyGalacticRotation(
    pos: { x: number; y: number; z: number },
    deltaYears: number
): { x: number; y: number; z: number } {
    if (Math.abs(deltaYears) < 10000) {
        return pos;  // Too short for visible galactic rotation
    }

    // Galactic rotation period: ~225 million years
    const galacticPeriodYears = 225e6;
    const omega = (2 * Math.PI) / (galacticPeriodYears * 365.25 * 86400);  // rad/s

    // Rotate around galactic center (approximate: 8 kpc toward galactic center)
    // For simplicity, rotate around Z axis (not exact but visually reasonable)
    const angle = omega * deltaYears * 365.25 * 86400;

    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);

    return {
        x: pos.x * cosA - pos.y * sinA,
        y: pos.x * sinA + pos.y * cosA,
        z: pos.z,
    };
}

/**
 * Full propagation including proper motion and galactic rotation
 */
export function propagateStarFull(
    basePos: { x: number; y: number; z: number },
    velocity: StellarVelocity,
    baseEpochJD: number,
    targetEpochJD: number
): { x: number; y: number; z: number } {
    // Linear proper motion
    let pos = propagateStarPosition(basePos, velocity, baseEpochJD, targetEpochJD);

    // Add galactic rotation for long timescales
    const deltaYears = (targetEpochJD - baseEpochJD) / 365.25;
    pos = applyGalacticRotation(pos, deltaYears);

    return pos;
}
