#!/usr/bin/env python3
"""
Convert deep sky objects from RA/Dec to ecliptic Cartesian coordinates.
Generates landmarks.json for the universe viewer.
"""

import json
import math
import sys
from pathlib import Path

# Constants
LIGHT_YEAR_METERS = 9.461e15  # 1 light year in meters
OBLIQUITY_DEG = 23.4392911  # Earth's axial tilt (J2000)

def ra_dec_to_cartesian(ra_deg: float, dec_deg: float, distance_ly: float) -> tuple:
    """Convert RA/Dec/distance to heliocentric Cartesian coordinates (ecliptic frame)."""

    ra_rad = math.radians(ra_deg)
    dec_rad = math.radians(dec_deg)
    obliquity_rad = math.radians(OBLIQUITY_DEG)

    distance_m = distance_ly * LIGHT_YEAR_METERS

    # Equatorial (J2000) to Cartesian
    cos_dec = math.cos(dec_rad)
    x_eq = distance_m * cos_dec * math.cos(ra_rad)
    y_eq = distance_m * cos_dec * math.sin(ra_rad)
    z_eq = distance_m * math.sin(dec_rad)

    # Rotate from equatorial to ecliptic (J2000)
    cos_e = math.cos(obliquity_rad)
    sin_e = math.sin(obliquity_rad)

    x = x_eq
    y = y_eq * cos_e + z_eq * sin_e
    z = -y_eq * sin_e + z_eq * cos_e

    return x, y, z


def convert_deep_sky_objects(input_file: Path, output_file: Path, include_existing: Path = None):
    """Convert deep sky objects and optionally merge with existing landmarks."""

    # Load deep sky objects
    with open(input_file) as f:
        objects = json.load(f)

    landmarks = []

    # Load existing landmarks if provided
    if include_existing and include_existing.exists():
        try:
            with open(include_existing) as f:
                existing = json.load(f)
                # Filter to keep only stars (remove old deep sky if any)
                landmarks.extend([lm for lm in existing if lm.get('kind') == 'star'])
                print(f"Loaded {len(landmarks)} existing star landmarks")
        except Exception as e:
            print(f"Warning: Could not load existing landmarks: {e}")

    # Convert deep sky objects
    for obj in objects:
        x, y, z = ra_dec_to_cartesian(obj['ra'], obj['dec'], obj['distance_ly'])

        landmark = {
            'id': obj['id'],
            'name': obj['name'],
            'kind': obj['kind'],
            'pos_meters': {'x': x, 'y': y, 'z': z},
            'description': obj.get('description', '')
        }

        # Add radius hint for better visualization
        if obj['kind'] == 'galaxy':
            # Galaxies: radius proportional to distance (typical angular size ~1 degree)
            landmark['radius_hint'] = obj['distance_ly'] * LIGHT_YEAR_METERS * 0.01
        elif obj['kind'] == 'nebula':
            # Nebulae: smaller, about 0.1 degree typical
            landmark['radius_hint'] = obj['distance_ly'] * LIGHT_YEAR_METERS * 0.001
        elif obj['kind'] == 'cluster':
            # Clusters: small, about 0.2 degree typical
            landmark['radius_hint'] = obj['distance_ly'] * LIGHT_YEAR_METERS * 0.002

        landmarks.append(landmark)

    print(f"Added {len(objects)} deep sky objects")
    print(f"Total landmarks: {len(landmarks)}")

    # Write output
    with open(output_file, 'w') as f:
        json.dump(landmarks, f, indent=2)

    print(f"Written to {output_file}")

    # Print summary by kind
    by_kind = {}
    for lm in landmarks:
        k = lm.get('kind', 'unknown')
        by_kind[k] = by_kind.get(k, 0) + 1

    print("\nLandmarks by type:")
    for k, c in sorted(by_kind.items()):
        print(f"  {k}: {c}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build deep sky landmarks')
    parser.add_argument('--universe', default='universe_gaia_50000',
                        help='Universe directory containing landmarks.json')
    parser.add_argument('--output', default=None,
                        help='Output file (default: same as input)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent

    input_file = script_dir / 'data' / 'deep_sky_objects.json'
    existing_file = script_dir / args.universe / 'landmarks.json'
    output_file = Path(args.output) if args.output else existing_file

    # Check if we should include existing landmarks
    include_existing = existing_file if existing_file.exists() else None

    convert_deep_sky_objects(input_file, output_file, include_existing)
