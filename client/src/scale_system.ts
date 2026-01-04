/**
 * Scale System for HELIOS Universe Planetarium
 *
 * Manages scale regimes from planetary (km) to intergalactic (Mpc) distances,
 * providing regime-aware speed scaling and zoom constraints.
 */

// Physical constants (meters)
export const ASTRONOMICAL_UNIT = 1.496e11;          // 1 AU
export const LIGHT_YEAR = 9.461e15;                 // 1 ly
export const PARSEC = 3.086e16;                     // 1 pc
export const KILOPARSEC = 3.086e19;                 // 1 kpc
export const MEGAPARSEC = 3.086e22;                 // 1 Mpc
export const HELIOSPHERE_RADIUS = 1.8e13;           // ~120 AU

/**
 * Scale regimes for navigation and rendering
 */
export enum ScaleRegime {
    Planetary = 0,      // < 1 Gm (near planets, spacecraft)
    SolarSystem = 1,    // 1 Gm - 100 Tm (within heliosphere)
    Interstellar = 2,   // 100 Tm - 1 Em (nearby stars)
    Galactic = 3,       // 1 Em - 1 Zm (Milky Way)
    Intergalactic = 4,  // > 1 Zm (other galaxies)
}

/**
 * Speed scaling configuration per regime
 */
export interface RegimeSpeedConfig {
    min_speed_m_s: number;    // Minimum speed (m/s)
    max_speed_m_s: number;    // Maximum speed (m/s)
    ramp_time_s: number;      // Time to reach max speed (seconds)
}

/**
 * Speed scaling parameters for each regime
 */
export const REGIME_SPEEDS: Record<ScaleRegime, RegimeSpeedConfig> = {
    [ScaleRegime.Planetary]: {
        min_speed_m_s: 1e2,      // 100 m/s
        max_speed_m_s: 1e6,      // 1000 km/s
        ramp_time_s: 5,
    },
    [ScaleRegime.SolarSystem]: {
        min_speed_m_s: 1e6,      // 1000 km/s
        max_speed_m_s: 1e10,     // ~67 AU/s
        ramp_time_s: 8,
    },
    [ScaleRegime.Interstellar]: {
        min_speed_m_s: 1e10,     // ~67 AU/s
        max_speed_m_s: 1e14,     // ~10.6 ly/s
        ramp_time_s: 10,
    },
    [ScaleRegime.Galactic]: {
        min_speed_m_s: 1e14,     // ~10.6 ly/s
        max_speed_m_s: 1e18,     // ~32 kpc/s
        ramp_time_s: 12,
    },
    [ScaleRegime.Intergalactic]: {
        min_speed_m_s: 1e18,     // ~32 kpc/s
        max_speed_m_s: 1e21,     // ~32 Mpc/s
        ramp_time_s: 15,
    },
};

/**
 * Distance thresholds for each regime (meters)
 */
const REGIME_THRESHOLDS: number[] = [
    1e9,    // 1 Gm  (~0.007 AU) - Planetary/Solar System boundary
    1e14,   // 100 Tm (~670 AU)  - Solar System/Interstellar boundary
    1e18,   // 1 Em   (~6.7 ly)  - Interstellar/Galactic boundary
    1e21,   // 1 Zm   (~32 kpc)  - Galactic/Intergalactic boundary
];

/**
 * Determine the scale regime based on distance from origin
 */
export function determineRegime(distanceM: number): ScaleRegime {
    for (let i = 0; i < REGIME_THRESHOLDS.length; i++) {
        if (distanceM < REGIME_THRESHOLDS[i]) {
            return i as ScaleRegime;
        }
    }
    return ScaleRegime.Intergalactic;
}

/**
 * Compute blend factor for smooth regime transitions
 * Returns 0.0 at regime start, 1.0 at regime end
 */
export function computeRegimeBlendFactor(distanceM: number, regime: ScaleRegime): number {
    if (regime === ScaleRegime.Intergalactic) {
        return 1.0;  // No upper bound
    }

    const lowerBound = regime === ScaleRegime.Planetary ? 0 : REGIME_THRESHOLDS[regime - 1];
    const upperBound = REGIME_THRESHOLDS[regime];

    // Logarithmic interpolation for smoother transitions
    const logDist = Math.log10(Math.max(1, distanceM));
    const logLower = Math.log10(Math.max(1, lowerBound));
    const logUpper = Math.log10(upperBound);

    return Math.min(1.0, Math.max(0.0, (logDist - logLower) / (logUpper - logLower)));
}

/**
 * Compute smooth blend weight across regime boundary
 * Used for avoiding jarring speed changes at transitions
 */
export function computeRegimeBoundaryBlend(distanceM: number): number {
    // Check if we're near a boundary (within 0.5 octaves)
    for (const boundary of REGIME_THRESHOLDS) {
        const ratio = distanceM / boundary;
        if (ratio > 0.5 && ratio < 2.0) {
            // Within transition zone, smooth blend
            // Maps [0.5, 2.0] → [0.0, 1.0] → [0.0, 1.0] (cosine)
            const t = Math.log2(ratio);  // -1.0 to 1.0
            return 0.5 + 0.5 * Math.cos(Math.PI * t);
        }
    }

    return 1.0;  // Not near boundary, full weight
}

/**
 * Compute maximum camera distance based on FOV and screen size
 * such that heliosphere appears as targetPixels on screen
 */
export function computeMaxDistance(
    fovYRadians: number,
    viewportHeight: number,
    systemRadiusM: number = HELIOSPHERE_RADIUS,
    targetPixels: number = 10
): number {
    // Angular size = 2 * atan(radius / distance)
    // screen_pixels = (angular_size / fovY) * viewportHeight
    // Solving for distance:
    // angular_size = (targetPixels / viewportHeight) * fovY
    // distance = radius / tan(angular_size / 2)

    const angularSizeRad = (targetPixels / viewportHeight) * fovYRadians;
    const maxDist = systemRadiusM / Math.tan(angularSizeRad / 2);

    // Clamp to reasonable intergalactic limit (~300 Mpc)
    return Math.min(maxDist, 1e25);
}

/**
 * Compute speed for a given regime and ramp state
 */
export function computeSpeed(
    regime: ScaleRegime,
    speedRamp: number,  // 0.0 to 1.0
    blendFactor: number = 1.0
): number {
    const config = REGIME_SPEEDS[regime];

    // Logarithmic interpolation between min and max speed
    const logMin = Math.log10(config.min_speed_m_s);
    const logMax = Math.log10(config.max_speed_m_s);
    const logSpeed = logMin + (logMax - logMin) * speedRamp;

    return Math.pow(10, logSpeed) * blendFactor;
}

/**
 * Format distance for display based on scale
 */
export function formatDistance(distanceM: number): string {
    const regime = determineRegime(distanceM);

    switch (regime) {
        case ScaleRegime.Planetary: {
            if (distanceM < 1000) {
                return `${distanceM.toFixed(1)} m`;
            } else if (distanceM < 1e6) {
                return `${(distanceM / 1000).toFixed(2)} km`;
            } else {
                return `${(distanceM / 1e6).toFixed(2)} Mm`;
            }
        }
        case ScaleRegime.SolarSystem: {
            const au = distanceM / ASTRONOMICAL_UNIT;
            return `${au.toFixed(2)} AU`;
        }
        case ScaleRegime.Interstellar: {
            const ly = distanceM / LIGHT_YEAR;
            if (ly < 1000) {
                return `${ly.toFixed(2)} ly`;
            } else {
                const pc = distanceM / PARSEC;
                return `${pc.toFixed(1)} pc`;
            }
        }
        case ScaleRegime.Galactic: {
            const kpc = distanceM / KILOPARSEC;
            return `${kpc.toFixed(2)} kpc`;
        }
        case ScaleRegime.Intergalactic: {
            const mpc = distanceM / MEGAPARSEC;
            return `${mpc.toFixed(2)} Mpc`;
        }
        default:
            return `${distanceM.toExponential(2)} m`;
    }
}

/**
 * Format speed for display based on scale
 */
export function formatSpeed(speedMS: number): string {
    if (speedMS < 1000) {
        return `${speedMS.toFixed(1)} m/s`;
    } else if (speedMS < 1e6) {
        return `${(speedMS / 1000).toFixed(1)} km/s`;
    } else if (speedMS < ASTRONOMICAL_UNIT) {
        return `${(speedMS / 1e6).toFixed(1)} Mm/s`;
    } else if (speedMS < LIGHT_YEAR) {
        const auPerS = speedMS / ASTRONOMICAL_UNIT;
        return `${auPerS.toFixed(2)} AU/s`;
    } else if (speedMS < PARSEC) {
        const lyPerS = speedMS / LIGHT_YEAR;
        return `${lyPerS.toFixed(2)} ly/s`;
    } else if (speedMS < KILOPARSEC) {
        const pcPerS = speedMS / PARSEC;
        return `${pcPerS.toFixed(2)} pc/s`;
    } else {
        const kpcPerS = speedMS / KILOPARSEC;
        return `${kpcPerS.toFixed(2)} kpc/s`;
    }
}

/**
 * Get regime name for display
 */
export function getRegimeName(regime: ScaleRegime): string {
    switch (regime) {
        case ScaleRegime.Planetary:
            return 'Planetary';
        case ScaleRegime.SolarSystem:
            return 'Solar System';
        case ScaleRegime.Interstellar:
            return 'Interstellar';
        case ScaleRegime.Galactic:
            return 'Galactic';
        case ScaleRegime.Intergalactic:
            return 'Intergalactic';
        default:
            return 'Unknown';
    }
}
