#!/usr/bin/env python3
"""Real ephemeris data compiler (NASA JPL Horizons CSV).

This script upgrades the demo orbit pipeline to real ephemeris samples while
preserving the runtime asset contract:

- assets/<name>_orbit.json  (Kepler params + residual_scale + count)
- assets/<name>_residuals.bin (little-endian u32 stream; low16=radial, high16=transverse)

The WGSL shader uses Gaussian gravitational constant units (AU, days):
  n = k / a^(3/2), where k ~= 0.01720209895 (rad/day)

Input format (between $$SOE and $$EOE):
  JDTDB, Calendar Date, X, Y, Z
where X,Y,Z are in AU.

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

TAU = 2.0 * math.pi
K_GAUSS = 0.01720209895  # rad/day, Gaussian gravitational constant
MU_SUN_AU3_PER_DAY2 = K_GAUSS * K_GAUSS


def _normalize_angle_rad(x: float) -> float:
    x = math.fmod(x, TAU)
    if x < 0.0:
        x += TAU
    return x


def _solve_kepler_E(M: np.ndarray, e: float, max_iter: int = 30, tol: float = 1e-13) -> np.ndarray:
    """Solve Kepler's equation M = E - e sin(E) for eccentric anomaly E."""
    M = np.asarray(M, dtype=np.float64)
    M = np.remainder(M, TAU)

    if e < 0.8:
        E = M.copy()
    else:
        E = np.full_like(M, math.pi)

    for _ in range(int(max_iter)):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = f / fp
        E = E - dE
        if float(np.max(np.abs(dE))) < tol:
            break

    return E


@dataclass(frozen=True)
class KeplerFit:
    a: float  # AU
    e: float
    i: float  # rad
    O: float  # rad (longitude of ascending node)
    w: float  # rad (argument of periapsis)
    M0: float  # rad at t=0 days


def _kepler_position_3d_au_days(t_days: np.ndarray, el: KeplerFit) -> np.ndarray:
    """Return Nx3 position in AU at times t_days (days)."""
    t_days = np.asarray(t_days, dtype=np.float64).reshape(-1)

    a = float(el.a)
    e = float(el.e)
    i = float(el.i)
    O = float(el.O)
    w = float(el.w)
    M0 = float(el.M0)

    n = K_GAUSS / (a ** 1.5)  # rad/day
    M = M0 + n * t_days

    E = _solve_kepler_E(M, e)

    x_orb = a * (np.cos(E) - e)
    y_orb = a * (math.sqrt(1.0 - e * e) * np.sin(E))

    cw = math.cos(w)
    sw = math.sin(w)
    ci = math.cos(i)
    si = math.sin(i)
    cO = math.cos(O)
    sO = math.sin(O)

    # z-rot by w
    x1 = x_orb * cw - y_orb * sw
    y1 = x_orb * sw + y_orb * cw

    # x-rot by i
    x2 = x1
    y2 = y1 * ci
    z2 = y1 * si

    # z-rot by O
    x3 = x2 * cO - y2 * sO
    y3 = x2 * sO + y2 * cO
    z3 = z2

    return np.stack([x3, y3, z3], axis=1).astype(np.float64, copy=False)


def _safe_t_dir_from_r(r_dir: np.ndarray) -> np.ndarray:
    """Mimic shader's transverse basis: normalize(cross(up, r_dir)) with fallback."""
    r_dir = np.asarray(r_dir, dtype=np.float64).reshape(3)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    t = np.cross(up, r_dir)
    if float(np.dot(t, t)) < 1e-12:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        t = np.cross(up, r_dir)

    n = float(np.linalg.norm(t))
    if n <= 0.0:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return t / n


def _state_to_elements_au_days(r: np.ndarray, v: np.ndarray, mu: float) -> KeplerFit:
    """Derive classical orbital elements from a single state in AU and AU/day."""
    r = np.asarray(r, dtype=np.float64).reshape(3)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    mu = float(mu)

    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))
    if r_norm <= 0.0:
        raise ValueError("Invalid state: |r| must be > 0")

    h_vec = np.cross(r, v)
    h = float(np.linalg.norm(h_vec))
    if h <= 0.0:
        raise ValueError("Invalid state: |h| must be > 0")

    k = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_vec = np.cross(k, h_vec)
    n = float(np.linalg.norm(n_vec))

    e_vec = (np.cross(v, h_vec) / mu) - (r / r_norm)
    e = float(np.linalg.norm(e_vec))
    e = min(max(e, 0.0), 0.999999)

    energy = 0.5 * v_norm * v_norm - mu / r_norm
    if abs(energy) < 1e-18:
        a = 1e30
    else:
        a = -mu / (2.0 * energy)

    inc = math.acos(max(-1.0, min(1.0, h_vec[2] / h)))

    if n < 1e-16:
        O = 0.0
    else:
        O = _normalize_angle_rad(math.atan2(n_vec[1], n_vec[0]))

    if n < 1e-16 or e < 1e-12:
        w = 0.0
    else:
        w = _normalize_angle_rad(
            math.atan2(float(np.dot(np.cross(n_vec, e_vec), h_vec)) / h, float(np.dot(n_vec, e_vec)))
        )

    # True anomaly
    if e < 1e-12:
        if n < 1e-16:
            nu = _normalize_angle_rad(math.atan2(r[1], r[0]))
        else:
            nu = _normalize_angle_rad(
                math.atan2(float(np.dot(np.cross(n_vec, r), h_vec)) / h, float(np.dot(n_vec, r)))
            )
    else:
        nu = _normalize_angle_rad(math.atan2(float(np.dot(np.cross(e_vec, r), h_vec)) / h, float(np.dot(e_vec, r))))

    # Eccentric anomaly & mean anomaly
    if e < 1e-12:
        E = nu
    else:
        E = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(nu / 2.0), math.sqrt(1.0 + e) * math.cos(nu / 2.0))
    M0 = _normalize_angle_rad(E - e * math.sin(E))

    return KeplerFit(a=float(a), e=float(e), i=float(inc), O=float(O), w=float(w), M0=float(M0))


class PlanetariumCompiler:
    def __init__(self, out_dir: str | Path = "assets") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def parse_jpl_horizons(self, filepath: str | Path) -> Tuple[np.ndarray, np.ndarray]:
        """Parse a NASA JPL Horizons CSV segment between $$SOE and $$EOE."""
        path = Path(filepath)
        lines = path.read_text(encoding="utf-8").splitlines()

        in_block = False
        jd: list[float] = []
        xyz: list[tuple[float, float, float]] = []

        for raw in lines:
            s = raw.strip()
            if not in_block:
                if s == "$$SOE":
                    in_block = True
                continue

            if s == "$$EOE":
                break

            if not s or s.startswith("#"):
                continue

            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 5:
                continue

            try:
                jd_i = float(parts[0])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
            except Exception:
                continue

            jd.append(jd_i)
            xyz.append((x, y, z))

        if len(jd) < 3:
            raise ValueError(f"Not enough samples parsed from {path} (need >= 3)")

        jd_arr = np.asarray(jd, dtype=np.float64)
        t_days = jd_arr - float(jd_arr[0])
        xyz_arr = np.asarray(xyz, dtype=np.float64)

        if not np.all(np.diff(t_days) > 0.0):
            raise ValueError("Parsed times must be strictly increasing")

        return t_days, xyz_arr

    def _fit_kepler_3d(self, t_days: np.ndarray, xyz_au: np.ndarray) -> KeplerFit:
        t_days = np.asarray(t_days, dtype=np.float64).reshape(-1)
        xyz_au = np.asarray(xyz_au, dtype=np.float64)
        if xyz_au.ndim != 2 or xyz_au.shape[1] != 3:
            raise ValueError("xyz_au must have shape (N,3)")
        if t_days.shape[0] != xyz_au.shape[0]:
            raise ValueError("t_days and xyz_au length mismatch")

        v_est = np.gradient(xyz_au, t_days, axis=0)
        guess = _state_to_elements_au_days(xyz_au[0], v_est[0], MU_SUN_AU3_PER_DAY2)

        # Try SciPy refinement if available.
        try:
            from scipy.optimize import least_squares  # type: ignore
        except Exception:
            return guess

        p0 = np.array([guess.a, guess.e, guess.i, guess.O, guess.w, guess.M0], dtype=np.float64)
        lower = np.array([1e-6, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        upper = np.array([np.inf, 0.999999, math.pi, TAU, TAU, TAU], dtype=np.float64)

        def residuals(p: np.ndarray) -> np.ndarray:
            a, e, i, O, w, M0 = [float(x) for x in p]
            el = KeplerFit(a=a, e=e, i=i, O=O, w=w, M0=M0)
            pred = _kepler_position_3d_au_days(t_days, el)
            return (pred - xyz_au).reshape(-1)

        fit = least_squares(residuals, p0, bounds=(lower, upper), max_nfev=4000)
        a, e, i, O, w, M0 = [float(x) for x in fit.x]
        return KeplerFit(a=a, e=e, i=i, O=O, w=w, M0=M0)

    def compile_orbit_assets(
        self,
        *,
        t_days: np.ndarray,
        xyz_au: np.ndarray,
        name: str,
        clip_percentile: float = 99.9,
    ) -> dict:
        """Fit Kepler, compute residuals, and emit `<name>_orbit.json` + `<name>_residuals.bin`."""
        el = self._fit_kepler_3d(t_days, xyz_au)

        base = _kepler_position_3d_au_days(t_days, el)
        dr = (xyz_au - base).astype(np.float64, copy=False)

        eps = 1e-30
        r_norm = np.linalg.norm(base, axis=1)
        r_dir = base / (r_norm[:, None] + eps)

        t_dir = np.stack([_safe_t_dir_from_r(r_dir[k]) for k in range(r_dir.shape[0])], axis=0)

        radial = np.sum(dr * r_dir, axis=1)
        transverse = np.sum(dr * t_dir, axis=1)

        both = np.concatenate([np.abs(radial), np.abs(transverse)], axis=0)
        vmax = float(np.percentile(both, float(clip_percentile))) if both.size else 0.0
        if not math.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0

        residual_scale = vmax / 32767.0
        if not math.isfinite(residual_scale) or residual_scale <= 0.0:
            residual_scale = 1.0

        q_r = np.rint(radial / residual_scale).clip(-32768, 32767).astype(np.int16)
        q_t = np.rint(transverse / residual_scale).clip(-32768, 32767).astype(np.int16)

        packed = q_r.astype(np.uint16).astype(np.uint32) | (q_t.astype(np.uint16).astype(np.uint32) << 16)
        packed = packed.astype("<u4", copy=False)

        orbit_json = {
            "a": float(el.a),
            "e": float(el.e),
            "i": float(el.i),
            "w": float(el.w),
            "O": float(el.O),
            "M0": float(el.M0),
            "residual_scale": float(residual_scale),
            "count": int(packed.size),
        }

        orbit_path = self.out_dir / f"{name}_orbit.json"
        residuals_path = self.out_dir / f"{name}_residuals.bin"

        orbit_path.write_text(json.dumps(orbit_json, indent=2) + "\n", encoding="utf-8")
        residuals_path.write_bytes(packed.tobytes(order="C"))

        rms = float(np.sqrt(np.mean(np.sum(dr * dr, axis=1))))

        return {
            "name": str(name),
            "input_samples": int(t_days.size),
            "outputs": {
                "orbit_json": str(orbit_path),
                "residuals_bin": str(residuals_path),
            },
            "fit": {
                "a": float(el.a),
                "e": float(el.e),
                "i": float(el.i),
                "O": float(el.O),
                "w": float(el.w),
                "M0": float(el.M0),
            },
            "residuals": {
                "residual_scale": float(residual_scale),
                "count": int(packed.size),
                "clip_percentile": float(clip_percentile),
                "rms_au": float(rms),
            },
        }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compile real NASA JPL Horizons ephemeris into orbit assets")
    ap.add_argument("--input", default="assets/eros_data.csv", help="Horizons CSV file (default: assets/eros_data.csv)")
    ap.add_argument("--out", default="assets", help="Output directory (default: assets/)")
    ap.add_argument("--name", default="eros", help="Asset name prefix (default: eros)")
    ap.add_argument("--clip-percentile", type=float, default=99.9, help="Percentile for residual scale estimation")

    args = ap.parse_args(list(argv) if argv is not None else None)

    compiler = PlanetariumCompiler(args.out)
    t_days, xyz = compiler.parse_jpl_horizons(args.input)
    info = compiler.compile_orbit_assets(t_days=t_days, xyz_au=xyz, name=str(args.name), clip_percentile=float(args.clip_percentile))

    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


