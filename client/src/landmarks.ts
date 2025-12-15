/**
 * Landmarks system for navigation.
 * Provides built-in POIs and optional ML-generated landmarks.
 */

export type LandmarkKind = 'star' | 'planet' | 'galaxy' | 'cluster' | 'nebula' | 'region' | 'other';
export type LandmarkSource = 'builtin' | 'ml' | 'user';

export interface Landmark {
    id: string;
    name: string;
    kind: LandmarkKind;
    /** Position in meters (ecliptic coordinates, same frame as dataset) */
    pos_meters: { x: number; y: number; z: number };
    /** Optional radius hint for display sizing (meters) */
    radius_hint?: number;
    /** Where this landmark came from */
    source: LandmarkSource;
    /** Optional description */
    description?: string;
}

// Constants
const AU_M = 1.496e11;
const PC_M = 3.086e16;

/**
 * Convert RA/Dec/Distance to ecliptic XYZ (meters).
 * - RA: right ascension in degrees (0-360)
 * - Dec: declination in degrees (-90 to +90)
 * - Distance: in parsecs
 */
function radecToEcliptic(raDeg: number, decDeg: number, distPc: number): { x: number; y: number; z: number } {
    const ra = (raDeg * Math.PI) / 180;
    const dec = (decDeg * Math.PI) / 180;
    const dist = distPc * PC_M;

    // Equatorial to ecliptic obliquity (J2000)
    const obliquity = (23.4393 * Math.PI) / 180;
    const cosObl = Math.cos(obliquity);
    const sinObl = Math.sin(obliquity);

    // Equatorial XYZ
    const xEq = dist * Math.cos(dec) * Math.cos(ra);
    const yEq = dist * Math.cos(dec) * Math.sin(ra);
    const zEq = dist * Math.sin(dec);

    // Rotate to ecliptic
    return {
        x: xEq,
        y: yEq * cosObl + zEq * sinObl,
        z: -yEq * sinObl + zEq * cosObl,
    };
}

/**
 * Built-in landmarks: solar system bodies + famous astronomical objects.
 */
export const BUILTIN_LANDMARKS: Landmark[] = [
    // Solar system bodies (approximate positions along +X axis)
    {
        id: 'sun',
        name: 'Sun',
        kind: 'star',
        pos_meters: { x: 0, y: 0, z: 0 },
        radius_hint: 6.96e8,
        source: 'builtin',
        description: 'Our star',
    },
    {
        id: 'mercury',
        name: 'Mercury',
        kind: 'planet',
        pos_meters: { x: 0.39 * AU_M, y: 0, z: 0 },
        radius_hint: 2.4e6,
        source: 'builtin',
    },
    {
        id: 'venus',
        name: 'Venus',
        kind: 'planet',
        pos_meters: { x: 0.72 * AU_M, y: 0, z: 0 },
        radius_hint: 6.05e6,
        source: 'builtin',
    },
    {
        id: 'earth',
        name: 'Earth',
        kind: 'planet',
        pos_meters: { x: 1.0 * AU_M, y: 0, z: 0 },
        radius_hint: 6.37e6,
        source: 'builtin',
        description: 'Home',
    },
    {
        id: 'mars',
        name: 'Mars',
        kind: 'planet',
        pos_meters: { x: 1.52 * AU_M, y: 0, z: 0 },
        radius_hint: 3.39e6,
        source: 'builtin',
    },
    {
        id: 'jupiter',
        name: 'Jupiter',
        kind: 'planet',
        pos_meters: { x: 5.2 * AU_M, y: 0, z: 0 },
        radius_hint: 6.99e7,
        source: 'builtin',
    },
    {
        id: 'saturn',
        name: 'Saturn',
        kind: 'planet',
        pos_meters: { x: 9.5 * AU_M, y: 0, z: 0 },
        radius_hint: 5.82e7,
        source: 'builtin',
    },
    {
        id: 'uranus',
        name: 'Uranus',
        kind: 'planet',
        pos_meters: { x: 19.2 * AU_M, y: 0, z: 0 },
        radius_hint: 2.54e7,
        source: 'builtin',
    },
    {
        id: 'neptune',
        name: 'Neptune',
        kind: 'planet',
        pos_meters: { x: 30.0 * AU_M, y: 0, z: 0 },
        radius_hint: 2.46e7,
        source: 'builtin',
    },

    // Famous stars
    {
        id: 'proxima-centauri',
        name: 'Proxima Centauri',
        kind: 'star',
        pos_meters: radecToEcliptic(217.4, -62.7, 1.3),
        radius_hint: 1.08e8,
        source: 'builtin',
        description: 'Nearest star to the Sun (4.2 ly)',
    },
    {
        id: 'sirius',
        name: 'Sirius',
        kind: 'star',
        pos_meters: radecToEcliptic(101.3, -16.7, 2.64),
        radius_hint: 1.19e9,
        source: 'builtin',
        description: 'Brightest star in the night sky (8.6 ly)',
    },
    {
        id: 'betelgeuse',
        name: 'Betelgeuse',
        kind: 'star',
        pos_meters: radecToEcliptic(88.8, 7.4, 168),
        radius_hint: 6.17e11,
        source: 'builtin',
        description: 'Red supergiant in Orion (~550 ly)',
    },
    {
        id: 'polaris',
        name: 'Polaris',
        kind: 'star',
        pos_meters: radecToEcliptic(37.95, 89.26, 132),
        radius_hint: 2.66e10,
        source: 'builtin',
        description: 'North Star (~430 ly)',
    },

    // Galactic features
    {
        id: 'galactic-center',
        name: 'Galactic Center',
        kind: 'region',
        pos_meters: radecToEcliptic(266.4, -29.0, 8178),
        radius_hint: 1e18,
        source: 'builtin',
        description: 'Center of the Milky Way (~26,700 ly)',
    },

    // Nearby galaxies
    {
        id: 'andromeda',
        name: 'Andromeda Galaxy (M31)',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(10.68, 41.27, 778000),
        radius_hint: 1.1e21,
        source: 'builtin',
        description: 'Nearest large galaxy (2.5 Mly)',
    },
    {
        id: 'lmc',
        name: 'Large Magellanic Cloud',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(80.89, -69.76, 49970),
        radius_hint: 2.1e20,
        source: 'builtin',
        description: 'Satellite galaxy (163,000 ly)',
    },
    {
        id: 'smc',
        name: 'Small Magellanic Cloud',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(13.16, -72.8, 61000),
        radius_hint: 1e20,
        source: 'builtin',
        description: 'Satellite galaxy (200,000 ly)',
    },

    // Notable nebulae
    {
        id: 'orion-nebula',
        name: 'Orion Nebula (M42)',
        kind: 'nebula',
        pos_meters: radecToEcliptic(83.82, -5.39, 412),
        radius_hint: 1.13e17,
        source: 'builtin',
        description: 'Nearest massive star-forming region (1,340 ly)',
    },
    {
        id: 'crab-nebula',
        name: 'Crab Nebula (M1)',
        kind: 'nebula',
        pos_meters: radecToEcliptic(83.63, 22.01, 2000),
        radius_hint: 5.2e16,
        source: 'builtin',
        description: 'Supernova remnant (6,500 ly)',
    },

    // Star clusters
    {
        id: 'pleiades',
        name: 'Pleiades (M45)',
        kind: 'cluster',
        pos_meters: radecToEcliptic(56.87, 24.12, 136),
        radius_hint: 3.7e16,
        source: 'builtin',
        description: 'Open star cluster (444 ly)',
    },
];

/**
 * Fetch optional ML-generated landmarks from /universe/landmarks.json.
 * Returns empty array if not found or invalid.
 */
export async function fetchMLLandmarks(baseUrl: string = '/universe'): Promise<Landmark[]> {
    try {
        const response = await fetch(`${baseUrl}/landmarks.json`);
        if (!response.ok) {
            console.log('[LANDMARKS] No ML landmarks file found (optional)');
            return [];
        }

        const data = await response.json();
        if (!Array.isArray(data)) {
            console.warn('[LANDMARKS] Invalid landmarks format, expected array');
            return [];
        }

        // Validate and normalize landmarks
        const landmarks: Landmark[] = [];
        for (const item of data) {
            if (!item.id || !item.name || !item.pos_meters) {
                console.warn('[LANDMARKS] Skipping invalid landmark:', item);
                continue;
            }

            landmarks.push({
                id: item.id,
                name: item.name,
                kind: item.kind || 'other',
                pos_meters: {
                    x: item.pos_meters.x ?? 0,
                    y: item.pos_meters.y ?? 0,
                    z: item.pos_meters.z ?? 0,
                },
                radius_hint: item.radius_hint,
                source: 'ml',
                description: item.description,
            });
        }

        console.log(`[LANDMARKS] Loaded ${landmarks.length} ML landmarks`);
        return landmarks;
    } catch (e) {
        console.log('[LANDMARKS] Could not fetch ML landmarks:', e);
        return [];
    }
}

/**
 * LandmarksManager: holds all landmarks and provides lookup functions.
 */
export class LandmarksManager {
    private landmarks: Map<string, Landmark> = new Map();

    constructor() {
        // Initialize with built-in landmarks
        for (const lm of BUILTIN_LANDMARKS) {
            this.landmarks.set(lm.id, lm);
        }
    }

    /**
     * Load ML landmarks (async) and merge them.
     */
    async loadMLLandmarks(baseUrl: string = '/universe'): Promise<void> {
        const mlLandmarks = await fetchMLLandmarks(baseUrl);
        for (const lm of mlLandmarks) {
            // Don't overwrite built-in landmarks with same ID
            if (!this.landmarks.has(lm.id)) {
                this.landmarks.set(lm.id, lm);
            }
        }
    }

    /**
     * Get all landmarks.
     */
    getAll(): Landmark[] {
        return Array.from(this.landmarks.values());
    }

    /**
     * Get landmark by ID.
     */
    get(id: string): Landmark | undefined {
        return this.landmarks.get(id);
    }

    /**
     * Get landmarks by kind.
     */
    getByKind(kind: LandmarkKind): Landmark[] {
        return this.getAll().filter((lm) => lm.kind === kind);
    }

    /**
     * Get landmarks by source.
     */
    getBySource(source: LandmarkSource): Landmark[] {
        return this.getAll().filter((lm) => lm.source === source);
    }

    /**
     * Find nearest landmark to a position.
     */
    findNearest(pos: { x: number; y: number; z: number }, maxResults: number = 5): Array<{ landmark: Landmark; distance: number }> {
        const results: Array<{ landmark: Landmark; distance: number }> = [];

        for (const lm of this.landmarks.values()) {
            const dx = lm.pos_meters.x - pos.x;
            const dy = lm.pos_meters.y - pos.y;
            const dz = lm.pos_meters.z - pos.z;
            const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
            results.push({ landmark: lm, distance });
        }

        results.sort((a, b) => a.distance - b.distance);
        return results.slice(0, maxResults);
    }

    /**
     * Search landmarks by name (case-insensitive substring match).
     */
    search(query: string): Landmark[] {
        const q = query.toLowerCase();
        return this.getAll().filter((lm) => lm.name.toLowerCase().includes(q));
    }
}
