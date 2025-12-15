import struct
import math
import os

# CONFIGURATION
INPUT_FILE = "assets/MPCORB.DAT"
OUTPUT_FILE = "assets/real_belt.bin"
MAX_ASTEROIDS = 100_000  # keep in sync with Rust if you clamp there


def parse_mpcorb_line(line: str):
    """
    Parses a single line of MPCORB.DAT format.
    See: https://minorplanetcenter.net/iau/info/MPOrbitFormat.html
    """
    try:
        # Skip header/footer
        if not line or len(line) < 103:
            return None

        # Read fields (Fixed width)
        # Mean anomaly (M) at epoch
        M_deg = float(line[26:35])
        # Argument of perihelion (w)
        w_deg = float(line[37:46])
        # Longitude of ascending node (node)
        node_deg = float(line[48:57])
        # Inclination (i)
        i_deg = float(line[59:68])
        # Eccentricity (e)
        e = float(line[70:79])
        # Semi-major axis (a) - AU
        a = float(line[92:103])

        # Convert degrees to radians
        deg2rad = math.pi / 180.0

        return {
            "semi_major_axis": a,
            "eccentricity": e,
            "inclination": i_deg * deg2rad,
            "arg_periapsis": w_deg * deg2rad,
            "long_asc_node": node_deg * deg2rad,
            "mean_anomaly_0": M_deg * deg2rad,
            "residual_scale": 0.0,
            "count": 0,
        }
    except ValueError:
        return None


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please download it first.")
        return

    print(f"Processing {INPUT_FILE}...")

    asteroids = []
    with open(INPUT_FILE, "r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            # Skip obvious header lines
            if "MPCORB" in line or "Displayable" in line:
                continue

            data = parse_mpcorb_line(line)
            if data:
                asteroids.append(data)
                if len(asteroids) >= MAX_ASTEROIDS:
                    break

    print(f"Parsed {len(asteroids)} orbits. Saving to {OUTPUT_FILE}...")

    # Pack into binary struct (matches Rust `KeplerParams`)
    # Rust layout (repr(C, align(16))): 7 f32 + 1 u32 == 32 bytes
    # semi_major_axis, eccentricity, inclination, arg_periapsis,
    # long_asc_node, mean_anomaly_0, residual_scale, count
    with open(OUTPUT_FILE, "wb") as f:
        for ast in asteroids:
            packed = struct.pack(
                "<7fI",
                ast["semi_major_axis"],
                ast["eccentricity"],
                ast["inclination"],
                ast["arg_periapsis"],
                ast["long_asc_node"],
                ast["mean_anomaly_0"],
                ast["residual_scale"],
                ast["count"],
            )
            f.write(packed)

    print(f"Done. Wrote {len(asteroids)} asteroids ({len(asteroids) * 32} bytes).")


if __name__ == "__main__":
    main()

