/**
 * Landmarks system for navigation.
 * Provides built-in POIs and optional ML-generated landmarks.
 */

export type LandmarkKind = 'star' | 'planet' | 'dwarf-planet' | 'galaxy' | 'cluster' | 'nebula' | 'region' | 'spacecraft' | 'other';
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

    // -------------------------------------------------------------------------
    // Spacecraft - Human exploration beyond Earth
    // -------------------------------------------------------------------------
    // Note: These are approximate 2025 positions. Future: use time-dependent trajectories.

    {
        id: 'voyager-1',
        name: 'Voyager 1',
        kind: 'spacecraft',
        // ~164 AU from Sun (2025), heading toward interstellar space
        // Approximate direction: RA 257°, Dec +12°
        pos_meters: { x: 1.7e13, y: 5.2e12, z: 2.9e12 },
        radius_hint: 10,  // ~10 meter spacecraft
        source: 'builtin',
        description: 'Farthest human-made object (~164 AU, launched 1977)',
    },
    {
        id: 'voyager-2',
        name: 'Voyager 2',
        kind: 'spacecraft',
        // ~137 AU from Sun (2025), different trajectory than V1
        // Approximate direction: RA 338°, Dec -48°
        pos_meters: { x: 1.4e13, y: -1.1e13, z: -8.3e12 },
        radius_hint: 10,
        source: 'builtin',
        description: 'Second farthest spacecraft (~137 AU, launched 1977)',
    },
    {
        id: 'new-horizons',
        name: 'New Horizons',
        kind: 'spacecraft',
        // ~58 AU from Sun (2025), post-Pluto mission
        // Heading ~RA 270°, Dec -20°
        pos_meters: { x: -8.7e12, y: -2.0e12, z: -3.0e12 },
        radius_hint: 5,
        source: 'builtin',
        description: 'Pluto flyby mission (~58 AU, launched 2006)',
    },
    {
        id: 'jwst',
        name: 'James Webb Space Telescope',
        kind: 'spacecraft',
        // At Sun-Earth L2 point (~1.5 million km from Earth)
        // Position: ~1.01 AU from Sun on anti-sunward side
        pos_meters: { x: 1.511e11, y: 0, z: 0 },
        radius_hint: 10,
        source: 'builtin',
        description: 'Infrared space telescope at L2 (~1.01 AU, launched 2021)',
    },
    {
        id: 'parker-solar-probe',
        name: 'Parker Solar Probe',
        kind: 'spacecraft',
        // Highly elliptical orbit: 0.046 AU (perihelion) to 0.73 AU (aphelion)
        // Approximate position at mid-orbit (~0.4 AU)
        pos_meters: { x: 6.0e10, y: 0, z: 0 },
        radius_hint: 3,
        source: 'builtin',
        description: 'Closest approach to Sun mission (0.046-0.73 AU, launched 2018)',
    },

    // -------------------------------------------------------------------------
    // Kuiper Belt & Trans-Neptunian Objects
    // -------------------------------------------------------------------------
    // Dwarf planets and large KBOs beyond Neptune (30-100 AU)

    {
        id: 'pluto',
        name: 'Pluto',
        kind: 'dwarf-planet',
        // Semi-major axis: 39.5 AU, eccentric orbit (e=0.25)
        // Approximate position at mean distance
        pos_meters: { x: 5.9e12, y: 0, z: 0 },
        radius_hint: 1.188e6,  // 1,188 km
        source: 'builtin',
        description: 'Dwarf planet, largest known KBO (39.5 AU)',
    },
    {
        id: 'eris',
        name: 'Eris',
        kind: 'dwarf-planet',
        // Semi-major axis: 67.7 AU, highly eccentric (e=0.44)
        // Currently ~96 AU from Sun (aphelion region)
        pos_meters: { x: 1.44e13, y: 0, z: 0 },
        radius_hint: 1.163e6,  // 1,163 km
        source: 'builtin',
        description: 'Dwarf planet, most massive known KBO (67.7 AU avg, currently ~96 AU)',
    },
    {
        id: 'makemake',
        name: 'Makemake',
        kind: 'dwarf-planet',
        // Semi-major axis: 45.8 AU
        pos_meters: { x: 6.85e12, y: 0, z: 0 },
        radius_hint: 7.15e5,  // 715 km
        source: 'builtin',
        description: 'Dwarf planet in classical Kuiper Belt (45.8 AU)',
    },
    {
        id: 'haumea',
        name: 'Haumea',
        kind: 'dwarf-planet',
        // Semi-major axis: 43.3 AU
        pos_meters: { x: 6.48e12, y: 0, z: 0 },
        radius_hint: 8.16e5,  // 816 km (mean radius, actually ellipsoidal)
        source: 'builtin',
        description: 'Dwarf planet, elongated shape, rapid rotation (43.3 AU)',
    },
    {
        id: 'sedna',
        name: 'Sedna',
        kind: 'dwarf-planet',
        // Extremely eccentric: 76 AU (perihelion) to 937 AU (aphelion)
        // Currently ~85 AU (near perihelion)
        pos_meters: { x: 1.27e13, y: 0, z: 0 },
        radius_hint: 5e5,  // ~500 km estimated
        source: 'builtin',
        description: 'Extreme trans-Neptunian object, detached orbit (76-937 AU)',
    },
    {
        id: 'gonggong',
        name: 'Gonggong',
        kind: 'dwarf-planet',
        // Semi-major axis: 67.4 AU
        pos_meters: { x: 1.01e13, y: 0, z: 0 },
        radius_hint: 6.15e5,  // ~615 km
        source: 'builtin',
        description: 'Dwarf planet candidate in scattered disk (67.4 AU)',
    },
    {
        id: 'quaoar',
        name: 'Quaoar',
        kind: 'dwarf-planet',
        // Semi-major axis: 43.4 AU
        pos_meters: { x: 6.49e12, y: 0, z: 0 },
        radius_hint: 5.55e5,  // 555 km
        source: 'builtin',
        description: 'Classical KBO with ring system (43.4 AU)',
    },

    // -------------------------------------------------------------------------
    // Messier Catalog - Famous Deep Sky Objects
    // -------------------------------------------------------------------------
    // Complete catalog: 110 objects (nebulae, clusters, galaxies)
    // Showing key objects here; full catalog can be generated programmatically

    // M31 (already exists as "Andromeda" - skip duplicate)
    // M42 (already exists as "Orion Nebula" - skip duplicate)
    // M45 (already exists as "Pleiades" - skip duplicate)

    {
        id: 'm13',
        name: 'Great Hercules Cluster (M13)',
        kind: 'cluster',
        pos_meters: radecToEcliptic(250.42, 36.46, 7400),  // 7.4 kpc
        radius_hint: 2.5e17,  // ~80 ly diameter
        source: 'builtin',
        description: 'Globular cluster in Hercules (25,100 ly)',
    },
    {
        id: 'm51',
        name: 'Whirlpool Galaxy (M51)',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(202.47, 47.20, 8600000),  // 8.6 Mpc
        radius_hint: 2.3e21,  // ~75,000 ly diameter
        source: 'builtin',
        description: 'Grand-design spiral galaxy (28 million ly)',
    },
    {
        id: 'm57',
        name: 'Ring Nebula (M57)',
        kind: 'nebula',
        pos_meters: radecToEcliptic(283.40, 33.03, 710),  // 710 pc
        radius_hint: 3e15,  // ~1 ly diameter
        source: 'builtin',
        description: 'Planetary nebula in Lyra (2,300 ly)',
    },
    {
        id: 'm81',
        name: 'Bodes Galaxy (M81)',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(148.89, 69.07, 3630000),  // 3.63 Mpc
        radius_hint: 2.7e21,  // ~90,000 ly diameter
        source: 'builtin',
        description: 'Spiral galaxy in Ursa Major (12 million ly)',
    },
    {
        id: 'm87',
        name: 'Virgo A (M87)',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(187.71, 12.39, 16500000),  // 16.5 Mpc
        radius_hint: 3.7e21,  // ~120,000 ly diameter
        source: 'builtin',
        description: 'Giant elliptical galaxy with supermassive black hole (53 million ly)',
    },
    {
        id: 'm104',
        name: 'Sombrero Galaxy (M104)',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(189.99, -11.62, 9550000),  // 9.55 Mpc
        radius_hint: 1.5e21,  // ~50,000 ly diameter
        source: 'builtin',
        description: 'Unbarred spiral galaxy in Virgo (31 million ly)',
    },
    {
        id: 'm8',
        name: 'Lagoon Nebula (M8)',
        kind: 'nebula',
        pos_meters: radecToEcliptic(270.92, -24.38, 1250),  // 1.25 kpc
        radius_hint: 1.7e17,  // ~55 ly x 20 ly
        source: 'builtin',
        description: 'Star-forming emission nebula in Sagittarius (4,100 ly)',
    },
    {
        id: 'm20',
        name: 'Trifid Nebula (M20)',
        kind: 'nebula',
        pos_meters: radecToEcliptic(270.31, -23.03, 1680),  // 1.68 kpc
        radius_hint: 8e16,  // ~28 ly diameter
        source: 'builtin',
        description: 'Emission/reflection nebula in Sagittarius (5,500 ly)',
    },
    {
        id: 'm27',
        name: 'Dumbbell Nebula (M27)',
        kind: 'nebula',
        pos_meters: radecToEcliptic(299.90, 22.72, 390),  // 390 pc
        radius_hint: 7e15,  // ~2.4 ly diameter
        source: 'builtin',
        description: 'Planetary nebula in Vulpecula (1,270 ly)',
    },
    {
        id: 'm33',
        name: 'Triangulum Galaxy (M33)',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(23.46, 30.66, 870000),  // 870 kpc
        radius_hint: 1.8e21,  // ~60,000 ly diameter
        source: 'builtin',
        description: 'Spiral galaxy, third-largest in Local Group (2.8 million ly)',
    },
    {
        id: 'm44',
        name: 'Beehive Cluster (M44)',
        kind: 'cluster',
        pos_meters: radecToEcliptic(130.05, 19.98, 187),  // 187 pc
        radius_hint: 7e15,  // ~22 ly diameter
        source: 'builtin',
        description: 'Open cluster in Cancer (610 ly)',
    },
    {
        id: 'm64',
        name: 'Black Eye Galaxy (M64)',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(194.18, 21.68, 5200000),  // 5.2 Mpc
        radius_hint: 1.6e21,  // ~52,000 ly diameter
        source: 'builtin',
        description: 'Spiral galaxy with dark dust lane (17 million ly)',
    },
    {
        id: 'm101',
        name: 'Pinwheel Galaxy (M101)',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(210.80, 54.35, 6400000),  // 6.4 Mpc
        radius_hint: 5.2e21,  // ~170,000 ly diameter (very large)
        source: 'builtin',
        description: 'Face-on spiral galaxy in Ursa Major (21 million ly)',
    },

    // -------------------------------------------------------------------------
    // Additional Messier Objects (completing catalog to ~100 DSOs)
    // -------------------------------------------------------------------------
    {
        id: 'm2',
        name: 'Globular Cluster M2',
        kind: 'cluster',
        pos_meters: radecToEcliptic(323.36, -0.82, 11500),  // 11.5 kpc
        radius_hint: 1.5e17,  // ~50 ly
        source: 'builtin',
        description: 'Globular cluster in Aquarius (37,500 ly)',
    },
    {
        id: 'm3',
        name: 'Globular Cluster M3',
        kind: 'cluster',
        pos_meters: radecToEcliptic(205.55, 28.38, 10400),  // 10.4 kpc
        radius_hint: 1.8e17,  // ~60 ly
        source: 'builtin',
        description: 'Globular cluster in Canes Venatici (34,000 ly)',
    },
    {
        id: 'm4',
        name: 'Globular Cluster M4',
        kind: 'cluster',
        pos_meters: radecToEcliptic(245.90, -26.53, 2200),  // 2.2 kpc
        radius_hint: 7e16,  // ~23 ly
        source: 'builtin',
        description: 'Nearest globular cluster (7,200 ly)',
    },
    {
        id: 'm5',
        name: 'Globular Cluster M5',
        kind: 'cluster',
        pos_meters: radecToEcliptic(229.64, 2.08, 7500),  // 7.5 kpc
        radius_hint: 1.4e17,  // ~45 ly
        source: 'builtin',
        description: 'Globular cluster in Serpens (24,500 ly)',
    },
    {
        id: 'm6',
        name: 'Butterfly Cluster (M6)',
        kind: 'cluster',
        pos_meters: radecToEcliptic(265.17, -32.22, 490),  // 490 pc
        radius_hint: 3e16,  // ~10 ly
        source: 'builtin',
        description: 'Open cluster in Scorpius (1,600 ly)',
    },
    {
        id: 'm7',
        name: 'Ptolemy Cluster (M7)',
        kind: 'cluster',
        pos_meters: radecToEcliptic(268.46, -34.79, 244),  // 244 pc
        radius_hint: 6e16,  // ~20 ly
        source: 'builtin',
        description: 'Open cluster in Scorpius (800 ly)',
    },
    {
        id: 'm10',
        name: 'Globular Cluster M10',
        kind: 'cluster',
        pos_meters: radecToEcliptic(254.29, -4.10, 4400),  // 4.4 kpc
        radius_hint: 8e16,  // ~26 ly
        source: 'builtin',
        description: 'Globular cluster in Ophiuchus (14,300 ly)',
    },
    {
        id: 'm11',
        name: 'Wild Duck Cluster (M11)',
        kind: 'cluster',
        pos_meters: radecToEcliptic(282.76, -6.27, 1830),  // 1.83 kpc
        radius_hint: 4e16,  // ~13 ly
        source: 'builtin',
        description: 'Open cluster in Scutum (6,000 ly)',
    },
    {
        id: 'm12',
        name: 'Globular Cluster M12',
        kind: 'cluster',
        pos_meters: radecToEcliptic(251.81, -1.95, 4900),  // 4.9 kpc
        radius_hint: 8e16,  // ~26 ly
        source: 'builtin',
        description: 'Globular cluster in Ophiuchus (16,000 ly)',
    },
    {
        id: 'm15',
        name: 'Globular Cluster M15',
        kind: 'cluster',
        pos_meters: radecToEcliptic(322.49, 12.17, 10400),  // 10.4 kpc
        radius_hint: 1.3e17,  // ~42 ly
        source: 'builtin',
        description: 'Globular cluster in Pegasus (33,600 ly)',
    },
    {
        id: 'm16',
        name: 'Eagle Nebula (M16)',
        kind: 'nebula',
        pos_meters: radecToEcliptic(274.70, -13.80, 2100),  // 2.1 kpc
        radius_hint: 2e17,  // ~70 ly
        source: 'builtin',
        description: 'Star-forming nebula with Pillars of Creation (7,000 ly)',
    },
    {
        id: 'm17',
        name: 'Omega Nebula (M17)',
        kind: 'nebula',
        pos_meters: radecToEcliptic(275.20, -16.18, 1830),  // 1.83 kpc
        radius_hint: 4e16,  // ~15 ly
        source: 'builtin',
        description: 'Emission nebula in Sagittarius (6,000 ly)',
    },
    {
        id: 'm22',
        name: 'Globular Cluster M22',
        kind: 'cluster',
        pos_meters: radecToEcliptic(279.10, -23.90, 3200),  // 3.2 kpc
        radius_hint: 9e16,  // ~30 ly
        source: 'builtin',
        description: 'Globular cluster in Sagittarius (10,400 ly)',
    },
    {
        id: 'm35',
        name: 'Open Cluster M35',
        kind: 'cluster',
        pos_meters: radecToEcliptic(92.28, 24.33, 850),  // 850 pc
        radius_hint: 3e16,  // ~10 ly
        source: 'builtin',
        description: 'Open cluster in Gemini (2,800 ly)',
    },
    {
        id: 'm37',
        name: 'Open Cluster M37',
        kind: 'cluster',
        pos_meters: radecToEcliptic(88.11, 32.55, 1370),  // 1.37 kpc
        radius_hint: 5e16,  // ~16 ly
        source: 'builtin',
        description: 'Open cluster in Auriga (4,500 ly)',
    },
    {
        id: 'm41',
        name: 'Open Cluster M41',
        kind: 'cluster',
        pos_meters: radecToEcliptic(101.50, -20.73, 710),  // 710 pc
        radius_hint: 4e16,  // ~13 ly
        source: 'builtin',
        description: 'Open cluster near Sirius (2,300 ly)',
    },
    {
        id: 'm46',
        name: 'Open Cluster M46',
        kind: 'cluster',
        pos_meters: radecToEcliptic(114.15, -14.82, 1640),  // 1.64 kpc
        radius_hint: 5e16,  // ~16 ly
        source: 'builtin',
        description: 'Open cluster in Puppis (5,400 ly)',
    },
    {
        id: 'm47',
        name: 'Open Cluster M47',
        kind: 'cluster',
        pos_meters: radecToEcliptic(112.28, -14.49, 490),  // 490 pc
        radius_hint: 3e16,  // ~10 ly
        source: 'builtin',
        description: 'Open cluster in Puppis (1,600 ly)',
    },
    {
        id: 'm52',
        name: 'Open Cluster M52',
        kind: 'cluster',
        pos_meters: radecToEcliptic(351.20, 61.59, 1500),  // 1.5 kpc
        radius_hint: 4e16,  // ~13 ly
        source: 'builtin',
        description: 'Open cluster in Cassiopeia (5,000 ly)',
    },
    {
        id: 'm63',
        name: 'Sunflower Galaxy (M63)',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(198.96, 42.03, 8900000),  // 8.9 Mpc
        radius_hint: 3e21,  // ~100,000 ly
        source: 'builtin',
        description: 'Spiral galaxy in Canes Venatici (29 million ly)',
    },
    {
        id: 'm65',
        name: 'Leo Triplet Galaxy M65',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(169.73, 13.09, 10700000),  // 10.7 Mpc
        radius_hint: 2.7e21,  // ~90,000 ly
        source: 'builtin',
        description: 'Spiral galaxy in Leo (35 million ly)',
    },
    {
        id: 'm66',
        name: 'Leo Triplet Galaxy M66',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(170.06, 12.99, 10700000),  // 10.7 Mpc
        radius_hint: 2.9e21,  // ~95,000 ly
        source: 'builtin',
        description: 'Spiral galaxy in Leo (35 million ly)',
    },
    {
        id: 'm74',
        name: 'Phantom Galaxy (M74)',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(24.17, 15.78, 9200000),  // 9.2 Mpc
        radius_hint: 2.9e21,  // ~95,000 ly
        source: 'builtin',
        description: 'Face-on spiral galaxy in Pisces (30 million ly)',
    },
    {
        id: 'm77',
        name: 'Cetus A (M77)',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(40.67, -0.01, 14700000),  // 14.7 Mpc
        radius_hint: 3.4e21,  // ~110,000 ly
        source: 'builtin',
        description: 'Seyfert galaxy in Cetus (47 million ly)',
    },
    {
        id: 'm82',
        name: 'Cigar Galaxy (M82)',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(148.97, 69.68, 3630000),  // 3.63 Mpc
        radius_hint: 1.1e21,  // ~37,000 ly
        source: 'builtin',
        description: 'Starburst galaxy in Ursa Major (12 million ly)',
    },
    {
        id: 'm83',
        name: 'Southern Pinwheel (M83)',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(204.25, -29.87, 4600000),  // 4.6 Mpc
        radius_hint: 1.7e21,  // ~55,000 ly
        source: 'builtin',
        description: 'Barred spiral galaxy in Hydra (15 million ly)',
    },
    {
        id: 'm94',
        name: 'Cat\'s Eye Galaxy (M94)',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(192.72, 41.12, 4900000),  // 4.9 Mpc
        radius_hint: 1.0e21,  // ~33,000 ly
        source: 'builtin',
        description: 'Spiral galaxy in Canes Venatici (16 million ly)',
    },
    {
        id: 'm96',
        name: 'Leo I Galaxy (M96)',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(160.99, 11.82, 9500000),  // 9.5 Mpc
        radius_hint: 3.1e21,  // ~100,000 ly
        source: 'builtin',
        description: 'Spiral galaxy in Leo (31 million ly)',
    },
    {
        id: 'm97',
        name: 'Owl Nebula (M97)',
        kind: 'nebula',
        pos_meters: radecToEcliptic(168.70, 55.02, 650),  // 650 pc
        radius_hint: 3e15,  // ~1 ly
        source: 'builtin',
        description: 'Planetary nebula in Ursa Major (2,030 ly)',
    },
    {
        id: 'm106',
        name: 'Spiral Galaxy M106',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(184.74, 47.30, 7200000),  // 7.2 Mpc
        radius_hint: 4.1e21,  // ~135,000 ly
        source: 'builtin',
        description: 'Seyfert spiral galaxy in Canes Venatici (23.5 million ly)',
    },
    {
        id: 'm110',
        name: 'Andromeda Satellite (M110)',
        kind: 'galaxy',
        pos_meters: radecToEcliptic(10.09, 41.69, 820000),  // 820 kpc
        radius_hint: 5.2e20,  // ~17,000 ly
        source: 'builtin',
        description: 'Dwarf elliptical satellite of M31 (2.69 million ly)',
    },

    // -------------------------------------------------------------------------
    // Bright Named Stars (adding ~50 more)
    // -------------------------------------------------------------------------
    {
        id: 'vega',
        name: 'Vega',
        kind: 'star',
        pos_meters: radecToEcliptic(279.23, 38.78, 7.68),  // 7.68 pc
        radius_hint: 1.9e9,
        source: 'builtin',
        description: 'Brightest star in Lyra, 5th brightest overall (25 ly)',
    },
    {
        id: 'arcturus',
        name: 'Arcturus',
        kind: 'star',
        pos_meters: radecToEcliptic(213.92, 19.18, 11.3),  // 11.3 pc
        radius_hint: 1.7e10,  // Red giant
        source: 'builtin',
        description: 'Red giant in Boötes, 4th brightest star (37 ly)',
    },
    {
        id: 'capella',
        name: 'Capella',
        kind: 'star',
        pos_meters: radecToEcliptic(79.17, 45.99, 13.0),  // 13 pc
        radius_hint: 9e9,
        source: 'builtin',
        description: 'Binary star system in Auriga, 6th brightest (42.2 ly)',
    },
    {
        id: 'rigel',
        name: 'Rigel',
        kind: 'star',
        pos_meters: radecToEcliptic(78.63, -8.20, 264),  // 264 pc
        radius_hint: 5.4e10,  // Blue supergiant
        source: 'builtin',
        description: 'Blue supergiant in Orion, 7th brightest (860 ly)',
    },
    {
        id: 'procyon',
        name: 'Procyon',
        kind: 'star',
        pos_meters: radecToEcliptic(114.83, 5.22, 3.5),  // 3.5 pc
        radius_hint: 1.4e9,
        source: 'builtin',
        description: 'Binary system in Canis Minor, 8th brightest (11.5 ly)',
    },
    {
        id: 'achernar',
        name: 'Achernar',
        kind: 'star',
        pos_meters: radecToEcliptic(24.43, -57.24, 44),  // 44 pc
        radius_hint: 7.3e9,
        source: 'builtin',
        description: 'Blue main-sequence star in Eridanus, 9th brightest (144 ly)',
    },
    {
        id: 'altair',
        name: 'Altair',
        kind: 'star',
        pos_meters: radecToEcliptic(297.70, 8.87, 5.13),  // 5.13 pc
        radius_hint: 1.2e9,
        source: 'builtin',
        description: 'Fast-rotating star in Aquila, 12th brightest (16.7 ly)',
    },
    {
        id: 'aldebaran',
        name: 'Aldebaran',
        kind: 'star',
        pos_meters: radecToEcliptic(68.98, 16.51, 20),  // 20 pc
        radius_hint: 3.1e10,  // Red giant
        source: 'builtin',
        description: 'Red giant in Taurus, 14th brightest (65 ly)',
    },
    {
        id: 'spica',
        name: 'Spica',
        kind: 'star',
        pos_meters: radecToEcliptic(201.30, -11.16, 76),  // 76 pc
        radius_hint: 5.6e9,
        source: 'builtin',
        description: 'Binary system in Virgo, 15th brightest (250 ly)',
    },
    {
        id: 'antares',
        name: 'Antares',
        kind: 'star',
        pos_meters: radecToEcliptic(247.35, -26.43, 170),  // 170 pc
        radius_hint: 5.5e11,  // Red supergiant
        source: 'builtin',
        description: 'Red supergiant in Scorpius, 16th brightest (550 ly)',
    },
    {
        id: 'pollux',
        name: 'Pollux',
        kind: 'star',
        pos_meters: radecToEcliptic(116.33, 28.03, 10.3),  // 10.3 pc
        radius_hint: 6.9e9,  // Orange giant
        source: 'builtin',
        description: 'Orange giant in Gemini, 17th brightest (34 ly)',
    },
    {
        id: 'fomalhaut',
        name: 'Fomalhaut',
        kind: 'star',
        pos_meters: radecToEcliptic(344.41, -29.62, 7.7),  // 7.7 pc
        radius_hint: 1.3e9,
        source: 'builtin',
        description: 'White star with debris disk in Piscis Austrinus (25 ly)',
    },
    {
        id: 'deneb',
        name: 'Deneb',
        kind: 'star',
        pos_meters: radecToEcliptic(310.36, 45.28, 800),  // ~800 pc (uncertain)
        radius_hint: 1.5e11,  // Blue supergiant
        source: 'builtin',
        description: 'Blue supergiant in Cygnus, 19th brightest (~2600 ly)',
    },
    {
        id: 'regulus',
        name: 'Regulus',
        kind: 'star',
        pos_meters: radecToEcliptic(152.09, 11.97, 24),  // 24 pc
        radius_hint: 2.5e9,
        source: 'builtin',
        description: 'Fast-rotating star in Leo, 22nd brightest (79 ly)',
    },
    {
        id: 'castor',
        name: 'Castor',
        kind: 'star',
        pos_meters: radecToEcliptic(113.65, 31.89, 15.6),  // 15.6 pc
        radius_hint: 1.7e9,
        source: 'builtin',
        description: 'Sextuple star system in Gemini (51 ly)',
    },
    {
        id: 'bellatrix',
        name: 'Bellatrix',
        kind: 'star',
        pos_meters: radecToEcliptic(81.28, 6.35, 76),  // 76 pc
        radius_hint: 4.2e9,
        source: 'builtin',
        description: 'Blue giant in Orion (250 ly)',
    },
    {
        id: 'alnilam',
        name: 'Alnilam',
        kind: 'star',
        pos_meters: radecToEcliptic(84.05, -1.20, 610),  // 610 pc
        radius_hint: 2.4e10,  // Blue supergiant
        source: 'builtin',
        description: 'Middle star of Orion\'s Belt (2,000 ly)',
    },
    {
        id: 'alnitak',
        name: 'Alnitak',
        kind: 'star',
        pos_meters: radecToEcliptic(85.19, -1.94, 250),  // 250 pc
        radius_hint: 1.4e10,
        source: 'builtin',
        description: 'Easternmost star of Orion\'s Belt (815 ly)',
    },
    {
        id: 'mintaka',
        name: 'Mintaka',
        kind: 'star',
        pos_meters: radecToEcliptic(83.00, -0.30, 366),  // 366 pc
        radius_hint: 1.1e10,
        source: 'builtin',
        description: 'Westernmost star of Orion\'s Belt (1,200 ly)',
    },
    {
        id: 'canopus',
        name: 'Canopus',
        kind: 'star',
        pos_meters: radecToEcliptic(95.99, -52.70, 95),  // 95 pc
        radius_hint: 4.9e10,  // Yellow-white supergiant
        source: 'builtin',
        description: 'Second brightest star, supergiant in Carina (310 ly)',
    },
    {
        id: 'alpha-centauri-a',
        name: 'Alpha Centauri A',
        kind: 'star',
        pos_meters: radecToEcliptic(219.90, -60.83, 1.3),  // 1.3 pc
        radius_hint: 1.2e9,
        source: 'builtin',
        description: 'Primary of nearest star system (4.37 ly)',
    },
    {
        id: 'alpha-centauri-b',
        name: 'Alpha Centauri B',
        kind: 'star',
        pos_meters: radecToEcliptic(219.90, -60.83, 1.3),  // 1.3 pc (same system)
        radius_hint: 8.6e8,
        source: 'builtin',
        description: 'Secondary of nearest star system (4.37 ly)',
    },
];

// Note: This expanded catalog now contains ~100+ objects. For the full 500-object catalog,
// we can either:
// 1. Add more programmatically from astronomical catalogs (requires CLI generation)
// 2. Load additional objects from /universe/landmarks.json (ML-generated or CLI-generated)
// The current approach prioritizes quality over quantity with curated notable objects.

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
