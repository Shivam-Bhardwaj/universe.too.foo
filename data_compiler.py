#!/usr/bin/env python3
"""Neural Planetarium data compiler.

Implements three pipeline modules:
1) Physics Compression (Kepler + Shannon): fit Keplerian elements to (x,y,z,t),
   compute residuals, decompose into radial/transverse components, quantize to
   int16, and pack into u32.
2) Registry builder: Object_ID(u32) -> CelestialProperty JSON mapping.
3) Neural weight exporter: flatten a (mock) PyTorch MLP into raw float32.

Notes
-----
- Binary outputs are written little-endian and are 4-byte aligned.
- By default, `residuals.bin` is **only** the packed `u32` stream
  (`radial:int16` | `transverse:int16`). A small fixed-size header can be
  included optionally for standalone decoding/debugging.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


TAU = 2.0 * math.pi
AU_METERS = 1.495978707e11


def _normalize_angle_rad(x: float) -> float:
    x = math.fmod(x, TAU)
    if x < 0.0:
        x += TAU
    return x


def _normalize_angle_rad_np(x: np.ndarray) -> np.ndarray:
    return np.remainder(x, TAU)


def _rotation_matrix_perifocal_to_inertial(raan: float, inc: float, argp: float) -> np.ndarray:
    """Matches `OrbitalElements::perifocal_to_ecliptic()` in Rust."""
    cos_o = math.cos(raan)
    sin_o = math.sin(raan)
    cos_i = math.cos(inc)
    sin_i = math.sin(inc)
    cos_w = math.cos(argp)
    sin_w = math.sin(argp)

    return np.array(
        [
            [cos_o * cos_w - sin_o * sin_w * cos_i, -cos_o * sin_w - sin_o * cos_w * cos_i, sin_o * sin_i],
            [sin_o * cos_w + cos_o * sin_w * cos_i, -sin_o * sin_w + cos_o * cos_w * cos_i, -cos_o * sin_i],
            [sin_w * sin_i, cos_w * sin_i, cos_i],
        ],
        dtype=np.float64,
    )


def _solve_kepler_eccentric_anomaly(M: np.ndarray, e: float, max_iter: int = 50, tol: float = 1e-12) -> np.ndarray:
    """Solve Kepler's equation M = E - e sin(E) for E (vectorized)."""
    M = _normalize_angle_rad_np(M)
    if e < 0.8:
        E = M.copy()
    else:
        E = np.full_like(M, math.pi)

    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = f / fp
        E = E - dE
        if float(np.max(np.abs(dE))) < tol:
            break

    return E


def _true_anomaly_from_E(E: np.ndarray, e: float) -> np.ndarray:
    # tan(nu/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    s = math.sqrt((1.0 + e) / (1.0 - e))
    return 2.0 * np.arctan(s * np.tan(E / 2.0))


@dataclass(frozen=True)
class KeplerElements:
    a: float
    e: float
    i: float
    raan: float
    argp: float
    m0: float
    t0: float
    mu: float


def _kepler_propagate(elements: KeplerElements, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate positions and velocities at times `t` (seconds)."""
    a = float(elements.a)
    e = float(elements.e)
    inc = float(elements.i)
    raan = float(elements.raan)
    argp = float(elements.argp)
    m0 = float(elements.m0)
    mu = float(elements.mu)
    t0 = float(elements.t0)

    n = math.sqrt(mu / (a**3))  # mean motion [rad/s]
    M = m0 + n * (t - t0)
    M = _normalize_angle_rad_np(M)

    E = _solve_kepler_eccentric_anomaly(M, e)
    nu = _true_anomaly_from_E(E, e)

    # Position in perifocal frame using eccentric anomaly (stable)
    x_pf = a * (np.cos(E) - e)
    y_pf = a * (math.sqrt(1.0 - e * e) * np.sin(E))

    # Velocity in perifocal frame using true anomaly (matches Rust)
    p = a * (1.0 - e * e)
    h = math.sqrt(mu * p)
    vx_pf = -(mu / h) * np.sin(nu)
    vy_pf = (mu / h) * (e + np.cos(nu))

    pf_pos = np.stack([x_pf, y_pf, np.zeros_like(x_pf)], axis=1)
    pf_vel = np.stack([vx_pf, vy_pf, np.zeros_like(vx_pf)], axis=1)

    R = _rotation_matrix_perifocal_to_inertial(raan, inc, argp)  # 3x3
    pos = pf_pos @ R.T
    vel = pf_vel @ R.T
    return pos, vel


def _state_to_elements(r: np.ndarray, v: np.ndarray, mu: float, t0: float) -> KeplerElements:
    """Derive classical orbital elements from a single state (r,v)."""
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
    if abs(energy) < 1e-24:
        # Parabolic-ish; fall back to a large a.
        a = 1e30
    else:
        a = -mu / (2.0 * energy)

    # Inclination
    inc = math.acos(max(-1.0, min(1.0, h_vec[2] / h)))

    # RAAN
    if n < 1e-16:
        raan = 0.0
    else:
        raan = math.atan2(n_vec[1], n_vec[0])
        raan = _normalize_angle_rad(raan)

    # Argument of periapsis
    if n < 1e-16 or e < 1e-12:
        argp = 0.0
    else:
        # ω = atan2( (n x e)·h, n·e * |h| )
        argp = math.atan2(float(np.dot(np.cross(n_vec, e_vec), h_vec)) / h, float(np.dot(n_vec, e_vec)))
        argp = _normalize_angle_rad(argp)

    # True anomaly
    if e < 1e-12:
        # Circular: use angle from node line to position
        if n < 1e-16:
            nu = math.atan2(r[1], r[0])
        else:
            # ν = atan2( (n x r)·h, n·r * |h| )
            nu = math.atan2(float(np.dot(np.cross(n_vec, r), h_vec)) / h, float(np.dot(n_vec, r)))
        nu = _normalize_angle_rad(nu)
    else:
        nu = math.atan2(float(np.dot(np.cross(e_vec, r), h_vec)) / h, float(np.dot(e_vec, r)))
        nu = _normalize_angle_rad(nu)

    # Eccentric anomaly and mean anomaly at epoch
    if e < 1e-12:
        E = nu
    else:
        E = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(nu / 2.0), math.sqrt(1.0 + e) * math.cos(nu / 2.0))
    M0 = E - e * math.sin(E)
    M0 = _normalize_angle_rad(M0)

    return KeplerElements(a=a, e=e, i=inc, raan=raan, argp=argp, m0=M0, t0=float(t0), mu=mu)


def _estimate_dt(t: np.ndarray, rel_tol: float = 1e-4, abs_tol: float = 0.0) -> float:
    if t.size < 2:
        return 0.0
    dt = np.diff(t)
    dt_med = float(np.median(dt))
    if dt_med <= 0.0:
        return 0.0
    if float(np.max(np.abs(dt - dt_med))) <= max(abs_tol, rel_tol * dt_med):
        return dt_med
    return 0.0


def _decompose_residuals(dr: np.ndarray, r_pred: np.ndarray, v_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (radial, transverse, normal) components of residual vector."""
    eps = 1e-30

    r_norm = np.linalg.norm(r_pred, axis=1)
    r_hat = r_pred / (r_norm[:, None] + eps)

    # Transverse direction: velocity projected onto plane perpendicular to r_hat
    v_proj = v_pred - (np.sum(v_pred * r_hat, axis=1)[:, None] * r_hat)
    v_norm = np.linalg.norm(v_proj, axis=1)
    t_hat = v_proj / (v_norm[:, None] + eps)

    # Normal direction
    n_hat = np.cross(r_hat, t_hat)
    n_norm = np.linalg.norm(n_hat, axis=1)
    n_hat = n_hat / (n_norm[:, None] + eps)

    radial = np.sum(dr * r_hat, axis=1)
    transverse = np.sum(dr * t_hat, axis=1)
    normal = np.sum(dr * n_hat, axis=1)
    return radial, transverse, normal


def _quantize_int16(values: np.ndarray, step: Optional[float], clip_percentile: float = 99.9) -> Tuple[np.ndarray, float]:
    """Quantize float values to int16 with symmetric uniform quantization."""
    values = np.asarray(values, dtype=np.float64)

    if step is None or step <= 0.0:
        if values.size == 0:
            step = 1.0
        else:
            vmax = float(np.percentile(np.abs(values), clip_percentile))
            if not math.isfinite(vmax) or vmax <= 0.0:
                step = 1.0
            else:
                step = vmax / 32767.0

    q = np.rint(values / step)
    q = np.clip(q, -32768, 32767).astype(np.int16)
    return q, float(step)


def _pack_int16_pair_to_u32(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Pack two int16 arrays into a u32 array: bits[0..15]=low, bits[16..31]=high."""
    low_u = np.asarray(low, dtype=np.int16).astype(np.uint16)
    high_u = np.asarray(high, dtype=np.int16).astype(np.uint16)
    packed = low_u.astype(np.uint32) | (high_u.astype(np.uint32) << 16)
    return packed.astype(np.uint32)


@dataclass
class CelestialProperty:
    radius_wrt_sun: float
    neural_model_id: int
    albedo_tint: Tuple[float, float, float]

    def to_json_obj(self) -> Dict[str, Any]:
        return {
            "radius_wrt_sun": float(self.radius_wrt_sun),
            "neural_model_id": int(self.neural_model_id),
            "albedo_tint": [float(self.albedo_tint[0]), float(self.albedo_tint[1]), float(self.albedo_tint[2])],
        }


class PlanetariumCompiler:
    """Python data compiler for the Neural Planetarium engine."""

    # 4s + 3*u32 + 9*f64 + 2*f64 = 104 bytes (4-byte aligned)
    _RESIDUALS_HEADER = struct.Struct("<4sIII9d2d")
    _RESIDUALS_MAGIC = b"KSHN"  # Kepler + SHaNnon (radial/transverse)

    def __init__(
        self,
        out_dir: str | Path = ".",
        *,
        mu: float = 1.32712440018e20,  # Sun μ [m^3/s^2]
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.mu = float(mu)

    def compile_physics(
        self,
        t: np.ndarray,
        xyz: np.ndarray,
        *,
        output_name: str = "residuals.bin",
        meta_output_name: Optional[str] = None,
        include_header: bool = False,
        time_scale: float = 1.0,
        pos_scale: float = 1.0,
        radial_step: Optional[float] = None,
        transverse_step: Optional[float] = None,
        clip_percentile: float = 99.9,
        max_nfev: int = 4000,
    ) -> Dict[str, Any]:
        """Fit Kepler elements and emit packed residual bitstream.

        Parameters
        ----------
        t: (N,) array
            Sample times (monotonic increasing). Units are arbitrary, but must
            be consistent with `time_scale` (seconds per input unit).
        xyz: (N,3) array
            Sample positions. Units are arbitrary, but must be consistent with
            `pos_scale` (meters per input unit).

        Returns
        -------
        dict with fitted elements and quantization info.
        """
        try:
            from scipy.optimize import least_squares
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "SciPy is required for Kepler fitting. Install with `pip install scipy`."
            ) from e

        t = np.asarray(t, dtype=np.float64).reshape(-1)
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N,3)")
        if t.shape[0] != xyz.shape[0]:
            raise ValueError("t and xyz must have the same length")
        if t.size < 3:
            raise ValueError("Need at least 3 samples to fit Kepler elements")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(xyz)):
            raise ValueError("Input contains NaN/Inf")
        if not np.all(np.diff(t) > 0.0):
            raise ValueError("t must be strictly increasing")

        time_scale = float(time_scale)
        pos_scale = float(pos_scale)
        if not math.isfinite(time_scale) or time_scale <= 0.0:
            raise ValueError("time_scale must be a positive finite number")
        if not math.isfinite(pos_scale) or pos_scale <= 0.0:
            raise ValueError("pos_scale must be a positive finite number")

        # Convert input units -> SI units (seconds, meters) for consistency with μ.
        t_si = t * time_scale
        xyz_si = xyz * pos_scale

        t0 = float(t_si[0])
        dt_est = _estimate_dt(t_si)

        # Initial guess from a state vector
        v_est = np.gradient(xyz_si, t_si, axis=0)
        try:
            guess = _state_to_elements(xyz_si[0], v_est[0], self.mu, t0)
            a0, e0, i0, raan0, argp0, m00 = (
                guess.a,
                guess.e,
                guess.i,
                guess.raan,
                guess.argp,
                guess.m0,
            )
        except Exception:
            r = np.linalg.norm(xyz_si, axis=1)
            rmin = float(np.min(r))
            rmax = float(np.max(r))
            a0 = 0.5 * (rmin + rmax)
            e0 = (rmax - rmin) / (rmax + rmin + 1e-30)
            e0 = float(np.clip(e0, 0.0, 0.9))
            i0, raan0, argp0, m00 = 0.0, 0.0, 0.0, 0.0

        # Fit parameters p = [a, e, i, raan, argp, m0]
        p0 = np.array([a0, e0, i0, raan0, argp0, m00], dtype=np.float64)
        lower = np.array([1e-6, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        upper = np.array([np.inf, 0.999999, math.pi, TAU, TAU, TAU], dtype=np.float64)

        def residuals(p: np.ndarray) -> np.ndarray:
            a, e, inc, raan, argp, m0 = [float(x) for x in p]
            el = KeplerElements(a=a, e=e, i=inc, raan=raan, argp=argp, m0=m0, t0=t0, mu=self.mu)
            pred, _vel = _kepler_propagate(el, t_si)
            return (pred - xyz_si).reshape(-1)

        fit = least_squares(residuals, p0, bounds=(lower, upper), max_nfev=int(max_nfev))
        a, e, inc, raan, argp, m0 = [float(x) for x in fit.x]
        elements = KeplerElements(a=a, e=e, i=inc, raan=raan, argp=argp, m0=m0, t0=t0, mu=self.mu)

        pred, vel = _kepler_propagate(elements, t_si)
        dr = xyz_si - pred
        radial, transverse, normal = _decompose_residuals(dr, pred, vel)

        q_rad, step_rad = _quantize_int16(radial, radial_step, clip_percentile=clip_percentile)
        q_trn, step_trn = _quantize_int16(transverse, transverse_step, clip_percentile=clip_percentile)
        packed = _pack_int16_pair_to_u32(q_rad, q_trn)

        # Payload: packed u32 stream
        out_path = self.out_dir / output_name
        with open(out_path, "wb") as f:
            if include_header:
                header = self._RESIDUALS_HEADER.pack(
                    self._RESIDUALS_MAGIC,
                    1,  # version
                    int(t_si.size),
                    0,  # flags (reserved)
                    float(elements.a),
                    float(elements.e),
                    float(elements.i),
                    float(elements.raan),
                    float(elements.argp),
                    float(elements.m0),
                    float(elements.mu),
                    float(elements.t0),
                    float(dt_est),
                    float(step_rad),
                    float(step_trn),
                )
                assert (len(header) % 4) == 0
                f.write(header)
            f.write(packed.astype("<u4", copy=False).tobytes(order="C"))

        # Alignment check
        size = out_path.stat().st_size
        if (size % 4) != 0:
            raise RuntimeError(f"Output is not 4-byte aligned: {out_path} size={size}")

        # Stats
        rms_fit = float(np.sqrt(np.mean(np.sum(dr * dr, axis=1))))
        dropped_rms = float(np.sqrt(np.mean(normal * normal)))

        info = {
            "output": str(out_path),
            "samples": int(t_si.size),
            "dt_est": float(dt_est),
            "units": {
                "time_scale_seconds_per_input_unit": float(time_scale),
                "pos_scale_meters_per_input_unit": float(pos_scale),
            },
            "kepler": {
                "a": float(a),
                "e": float(e),
                "i": float(inc),
                "raan": float(raan),
                "argp": float(argp),
                "m0": float(m0),
                "t0": float(t0),
                "mu": float(self.mu),
            },
            "quantization": {
                "radial_step": float(step_rad),
                "transverse_step": float(step_trn),
                "clip_percentile": float(clip_percentile),
            },
            "fit_rms_position": rms_fit,
            "dropped_normal_rms": dropped_rms,
            "optimizer": {
                "success": bool(fit.success),
                "cost": float(fit.cost),
                "nfev": int(getattr(fit, "nfev", -1)),
                "message": str(getattr(fit, "message", "")),
            },
        }

        if meta_output_name is not None:
            meta_path = self.out_dir / str(meta_output_name)
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(info, mf, indent=2, sort_keys=True)
                mf.write("\n")
            info["meta_output"] = str(meta_path)

        return info

    def build_registry(
        self,
        *,
        output_name: str = "registry.json",
        registry: Optional[Mapping[int, CelestialProperty]] = None,
    ) -> str:
        """Generate `registry.json` mapping Object_ID(u32) -> properties."""
        if registry is None:
            registry = self._default_registry()

        out: Dict[str, Any] = {}
        for obj_id, props in registry.items():
            oid = int(obj_id)
            if oid < 0 or oid > 0xFFFF_FFFF:
                raise ValueError(f"Object_ID out of u32 range: {oid}")
            out[str(oid)] = props.to_json_obj()

        out_path = self.out_dir / output_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
            f.write("\n")

        return str(out_path)

    def export_neural_brain(
        self,
        *,
        output_name: str = "neural_decoder.bin",
        seed: int = 0,
        state_dict_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Export raw float32 weights for a simple MLP.

        Model: 35 -> 64 -> 64 -> 4 (Linear + ReLU + Linear + ReLU + Linear).
        Layout (row-major): W0, b0, W1, b1, W2, b2 as a flat float32 array.
        """
        # Prefer PyTorch if available (weights can be loaded from a state_dict),
        # but fall back to deterministic NumPy weights so the Rust/WGPU pipeline
        # can still be exercised without torch installed.
        try:  # pragma: no cover
            import torch
            import torch.nn as nn
        except Exception:
            torch = None  # type: ignore[assignment]
            nn = None  # type: ignore[assignment]

        if torch is None or nn is None:
            rng = np.random.default_rng(int(seed))
            # Shapes must match WGSL offsets: 35->64->64->4
            fc1_w = rng.standard_normal((64, 35)).astype(np.float32) * 0.02
            fc1_b = rng.standard_normal((64,)).astype(np.float32) * 0.02
            fc2_w = rng.standard_normal((64, 64)).astype(np.float32) * 0.02
            fc2_b = rng.standard_normal((64,)).astype(np.float32) * 0.02
            fc3_w = rng.standard_normal((4, 64)).astype(np.float32) * 0.02
            fc3_b = rng.standard_normal((4,)).astype(np.float32) * 0.02

            flat = np.concatenate(
                [
                    fc1_w.reshape(-1),
                    fc1_b.reshape(-1),
                    fc2_w.reshape(-1),
                    fc2_b.reshape(-1),
                    fc3_w.reshape(-1),
                    fc3_b.reshape(-1),
                ]
            ).astype("<f4", copy=False)
            torch_used = False
        else:
            class SimpleMLP(nn.Module):  # type: ignore[misc]
                def __init__(self) -> None:
                    super().__init__()
                    self.fc1 = nn.Linear(35, 64)
                    self.fc2 = nn.Linear(64, 64)
                    self.fc3 = nn.Linear(64, 4)

                def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                    x = torch.relu(self.fc1(x))
                    x = torch.relu(self.fc2(x))
                    return self.fc3(x)

            torch.manual_seed(int(seed))
            model = SimpleMLP().cpu().eval()

            if state_dict_path is not None:
                sd = torch.load(str(state_dict_path), map_location="cpu")
                model.load_state_dict(sd)

            parts: list[np.ndarray] = []
            for layer in (model.fc1, model.fc2, model.fc3):
                w = layer.weight.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
                b = layer.bias.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
                parts.append(w.reshape(-1))  # row-major
                parts.append(b.reshape(-1))

            flat = np.concatenate(parts).astype("<f4", copy=False)
            torch_used = True

        out_path = self.out_dir / output_name
        with open(out_path, "wb") as f:
            f.write(flat.tobytes(order="C"))

        size = out_path.stat().st_size
        if (size % 4) != 0:
            raise RuntimeError(f"Output is not 4-byte aligned: {out_path} size={size}")

        return {
            "output": str(out_path),
            "floats": int(flat.size),
            "bytes": int(size),
            "torch_used": bool(torch_used),
            "layout": ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"],
            "shapes": {
                "fc1.weight": [64, 35],
                "fc1.bias": [64],
                "fc2.weight": [64, 64],
                "fc2.bias": [64],
                "fc3.weight": [4, 64],
                "fc3.bias": [4],
            },
        }

    def _default_registry(self) -> Dict[int, CelestialProperty]:
        # Approximate radii normalized by Sun radius.
        # Values are not authoritative; they're placeholders for the palette system.
        sun_r_km = 696_340.0

        def r_norm(km: float) -> float:
            return km / sun_r_km

        return {
            0: CelestialProperty(radius_wrt_sun=1.0, neural_model_id=0, albedo_tint=(1.0, 1.0, 1.0)),  # Sun
            1: CelestialProperty(radius_wrt_sun=r_norm(2_439.7), neural_model_id=1, albedo_tint=(0.9, 0.85, 0.8)),  # Mercury
            2: CelestialProperty(radius_wrt_sun=r_norm(6_051.8), neural_model_id=1, albedo_tint=(0.95, 0.9, 0.75)),  # Venus
            3: CelestialProperty(radius_wrt_sun=r_norm(6_371.0), neural_model_id=2, albedo_tint=(0.85, 0.9, 1.0)),  # Earth
            4: CelestialProperty(radius_wrt_sun=r_norm(3_389.5), neural_model_id=2, albedo_tint=(1.0, 0.7, 0.6)),  # Mars
            5: CelestialProperty(radius_wrt_sun=r_norm(69_911.0), neural_model_id=3, albedo_tint=(0.95, 0.9, 0.85)),  # Jupiter
            6: CelestialProperty(radius_wrt_sun=r_norm(58_232.0), neural_model_id=3, albedo_tint=(0.95, 0.9, 0.8)),  # Saturn
            7: CelestialProperty(radius_wrt_sun=r_norm(25_362.0), neural_model_id=3, albedo_tint=(0.8, 0.9, 0.95)),  # Uranus
            8: CelestialProperty(radius_wrt_sun=r_norm(24_622.0), neural_model_id=3, albedo_tint=(0.75, 0.85, 0.95)),  # Neptune
        }


def _load_timeseries(path: Path, *, order: str = "xyzt") -> Tuple[np.ndarray, np.ndarray]:
    """Load a (t, xyz) time-series.

    Supported:
    - .npz with arrays `t` and `xyz`.
    - .npy with shape (N,4).
    - .csv with header containing columns t,x,y,z (order irrelevant).
      If no header, uses `order` (default 'xyzt').
    """
    suffix = path.suffix.lower()

    if suffix == ".npz":
        data = np.load(path)
        if "t" in data and "xyz" in data:
            t = np.asarray(data["t"], dtype=np.float64).reshape(-1)
            xyz = np.asarray(data["xyz"], dtype=np.float64)
            return t, xyz
        # Fall back: accept x,y,z,t
        keys = set(data.files)
        if {"x", "y", "z", "t"}.issubset(keys):
            t = np.asarray(data["t"], dtype=np.float64).reshape(-1)
            xyz = np.stack(
                [np.asarray(data["x"], dtype=np.float64), np.asarray(data["y"], dtype=np.float64), np.asarray(data["z"], dtype=np.float64)],
                axis=1,
            )
            return t, xyz
        raise ValueError(f"Unsupported npz layout: keys={data.files}")

    if suffix == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError(".npy input must have shape (N,4)")
        cols = {c: i for i, c in enumerate(order)}
        t = arr[:, cols["t"]]
        xyz = np.stack([arr[:, cols["x"]], arr[:, cols["y"]], arr[:, cols["z"]]], axis=1)
        return t, xyz

    if suffix == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
        has_header = any(ch.isalpha() for ch in first)

        if has_header:
            rec = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
            names = [n.lower() for n in rec.dtype.names or ()]
            name_to_col = {n: i for i, n in enumerate(rec.dtype.names or ())}

            def col(*cands: str) -> str:
                for c in cands:
                    if c in names:
                        return rec.dtype.names[names.index(c)]
                raise ValueError(f"Missing required column. Have: {rec.dtype.names}")

            t_name = col("t", "time")
            x_name = col("x")
            y_name = col("y")
            z_name = col("z")

            t = np.asarray(rec[t_name], dtype=np.float64)
            xyz = np.stack(
                [
                    np.asarray(rec[x_name], dtype=np.float64),
                    np.asarray(rec[y_name], dtype=np.float64),
                    np.asarray(rec[z_name], dtype=np.float64),
                ],
                axis=1,
            )
            return t, xyz

        arr = np.genfromtxt(path, delimiter=",", dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != 4:
            raise ValueError("CSV without header must have exactly 4 columns")
        cols = {c: i for i, c in enumerate(order)}
        t = arr[:, cols["t"]]
        xyz = np.stack([arr[:, cols["x"]], arr[:, cols["y"]], arr[:, cols["z"]]], axis=1)
        return t, xyz

    raise ValueError(f"Unsupported input format: {path}")


def _demo_orbit(mu: float, n: int = 2048, dt: float = 3600.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a slightly perturbed Earth-like orbit time-series (demo)."""
    # Earth-like nominal elements
    a = 1.496e11
    e = 0.0167
    inc = math.radians(0.5)
    raan = math.radians(15.0)
    argp = math.radians(45.0)
    m0 = 0.0
    t0 = 0.0
    el = KeplerElements(a=a, e=e, i=inc, raan=raan, argp=argp, m0=m0, t0=t0, mu=mu)

    t = np.arange(n, dtype=np.float64) * float(dt)
    pos, _vel = _kepler_propagate(el, t)

    # Inject synthetic perturbation to make residuals non-zero
    wobble = 2.0e6  # 2000 km
    noise = np.stack(
        [
            wobble * np.sin(0.00002 * t),
            wobble * np.cos(0.000013 * t),
            0.25 * wobble * np.sin(0.000017 * t),
        ],
        axis=1,
    )
    xyz = pos + noise
    return t, xyz


def _demo_planar_orbit_au_days(days: int = 365, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Generate a planar (XY) Earth-like orbit in AU sampled daily.

    This matches the simplified WGSL prototype that operates in AU/day units.
    """
    days = int(days)
    if days <= 2:
        raise ValueError("days must be >= 3")

    rng = np.random.default_rng(int(seed))

    # Nominal Earth-ish parameters (AU, radians)
    a = 1.0
    e = 0.0167
    inc = 0.0
    O = 0.0
    w = math.radians(30.0)
    M0 = 0.0

    # Gaussian gravitational constant (AU^(3/2) / day)
    k_gauss = 0.01720209895
    n = k_gauss / (a ** 1.5)  # rad/day

    t_days = np.arange(days, dtype=np.float64)
    M = (M0 + n * t_days).astype(np.float64)
    E = _solve_kepler_eccentric_anomaly(M, e)

    X = a * (np.cos(E) - e)
    Y = a * (math.sqrt(1.0 - e * e) * np.sin(E))

    cw = math.cos(w)
    sw = math.sin(w)
    xw = X * cw - Y * sw
    yw = X * sw + Y * cw

    base = np.stack([xw, yw, np.zeros_like(xw)], axis=1)

    # Inject small, smooth perturbations in the same radial/transverse basis the shader uses.
    r_norm = np.linalg.norm(base, axis=1, keepdims=True)
    r_hat = base / (r_norm + 1e-12)
    t_hat = np.stack([-r_hat[:, 1], r_hat[:, 0], np.zeros_like(r_hat[:, 0])], axis=1)

    # ~3000 km in AU
    amp = 3000.0e3 / AU_METERS
    radial = amp * np.sin(2.0 * math.pi * t_days / 27.0) + 0.15 * amp * rng.standard_normal(t_days.shape)
    transverse = amp * np.cos(2.0 * math.pi * t_days / 31.0) + 0.15 * amp * rng.standard_normal(t_days.shape)

    xyz = base + r_hat * radial[:, None] + t_hat * transverse[:, None]

    meta = {"a": float(a), "e": float(e), "i": float(inc), "O": float(O), "w": float(w), "M0": float(M0)}
    return t_days, xyz, meta


def _pack_shannon_residuals_planar(
    t_days: np.ndarray,
    xyz_au: np.ndarray,
    *,
    a: float,
    e: float,
    w: float,
    M0: float,
    clip_percentile: float = 99.9,
) -> Tuple[np.ndarray, float]:
    """Compute planar Kepler prediction and pack radial/transverse residuals into u32 stream."""
    t_days = np.asarray(t_days, dtype=np.float64).reshape(-1)
    xyz_au = np.asarray(xyz_au, dtype=np.float64)
    if xyz_au.ndim != 2 or xyz_au.shape[1] != 3:
        raise ValueError("xyz_au must have shape (N,3)")
    if t_days.shape[0] != xyz_au.shape[0]:
        raise ValueError("t_days and xyz_au must have same length")

    k_gauss = 0.01720209895
    n = k_gauss / (float(a) ** 1.5)

    M = (float(M0) + n * t_days).astype(np.float64)
    E = _solve_kepler_eccentric_anomaly(M, float(e))

    X = float(a) * (np.cos(E) - float(e))
    Y = float(a) * (math.sqrt(1.0 - float(e) * float(e)) * np.sin(E))

    cw = math.cos(float(w))
    sw = math.sin(float(w))
    xw = X * cw - Y * sw
    yw = X * sw + Y * cw
    base = np.stack([xw, yw, np.zeros_like(xw)], axis=1)

    dr = xyz_au - base
    r_norm = np.linalg.norm(base, axis=1, keepdims=True)
    r_hat = base / (r_norm + 1e-12)
    t_hat = np.stack([-r_hat[:, 1], r_hat[:, 0], np.zeros_like(r_hat[:, 0])], axis=1)

    err_r = np.sum(dr * r_hat, axis=1)
    err_t = np.sum(dr * t_hat, axis=1)

    vmax = float(np.percentile(np.abs(np.concatenate([err_r, err_t])), float(clip_percentile)))
    if not math.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    # Single shared scale (units per LSB), matching shader's `* orbit.residual_scale`.
    residual_scale = vmax / 32767.0

    q_r = np.rint(err_r / residual_scale).clip(-32768, 32767).astype(np.int16)
    q_t = np.rint(err_t / residual_scale).clip(-32768, 32767).astype(np.int16)

    packed = _pack_int16_pair_to_u32(q_r, q_t)  # low=radial, high=transverse
    return packed, float(residual_scale)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Neural Planetarium data compiler")
    ap.add_argument("--out", default="assets", help="Output directory (default: assets/)")
    ap.add_argument("--mu", type=float, default=1.32712440018e20, help="Gravitational parameter μ (m^3/s^2)")

    sub = ap.add_subparsers(dest="cmd", required=False)

    ap_all = sub.add_parser("build-assets", help="Generate a runnable demo asset set in assets/")
    ap_all.add_argument("--name", type=str, default="earth", help="Object name prefix (default: earth)")
    ap_all.add_argument("--days", type=int, default=365, help="Number of daily samples (default: 365)")
    ap_all.add_argument("--seed", type=int, default=0, help="Seed for demo noise/weights")
    ap_all.add_argument("--clip-percentile", type=float, default=99.9, help="Percentile for residual scale estimation")

    ap_phys = sub.add_parser("compile-physics", help="Fit Kepler elements and emit residuals.bin")
    ap_phys.add_argument("--input", type=str, default=None, help="Input time-series (.npz/.npy/.csv). If omitted, runs a demo orbit.")
    ap_phys.add_argument("--order", type=str, default="xyzt", help="Column order for 4-column .npy/CSV without header (e.g. 'txyz' or 'xyzt')")
    ap_phys.add_argument("--output", type=str, default="residuals.bin", help="Output filename")
    ap_phys.add_argument("--meta-output", type=str, default=None, help="Optional JSON sidecar with fit + quantization info")
    ap_phys.add_argument("--include-header", action="store_true", help="Include a fixed-size header before the u32 stream")
    ap_phys.add_argument("--time-unit", type=str, default="seconds", choices=["seconds", "days"], help="Input time unit")
    ap_phys.add_argument("--pos-unit", type=str, default="meters", choices=["meters", "km", "au"], help="Input position unit")
    ap_phys.add_argument("--time-scale", type=float, default=None, help="Override seconds per input time unit")
    ap_phys.add_argument("--pos-scale", type=float, default=None, help="Override meters per input position unit")
    ap_phys.add_argument("--radial-step", type=float, default=None, help="Quantization step for radial residual (units per LSB)")
    ap_phys.add_argument("--transverse-step", type=float, default=None, help="Quantization step for transverse residual (units per LSB)")
    ap_phys.add_argument("--clip-percentile", type=float, default=99.9, help="Percentile for auto-quant range")
    ap_phys.add_argument("--max-nfev", type=int, default=4000, help="Optimizer eval budget")

    ap_reg = sub.add_parser("build-registry", help="Generate registry.json palette")
    ap_reg.add_argument("--output", type=str, default="registry.json", help="Output filename")

    ap_nn = sub.add_parser("export-neural-brain", help="Export raw float32 weights")
    ap_nn.add_argument("--output", type=str, default="neural_decoder.bin", help="Output filename")
    ap_nn.add_argument("--seed", type=int, default=0, help="Random seed for mock weights")
    ap_nn.add_argument("--load-pth", type=str, default=None, help="Optional .pth state_dict to load before exporting")

    import sys
    if argv is None:
        argv = sys.argv[1:]
    # Convenience: `python3 data_compiler.py` generates demo assets to ./assets/
    if len(argv) == 0:
        argv = ["build-assets"]
    args = ap.parse_args(argv)

    compiler = PlanetariumCompiler(args.out, mu=args.mu)

    if args.cmd == "build-assets":
        t_days, xyz_au, meta = _demo_planar_orbit_au_days(days=args.days, seed=args.seed)
        packed, residual_scale = _pack_shannon_residuals_planar(
            t_days,
            xyz_au,
            a=meta["a"],
            e=meta["e"],
            w=meta["w"],
            M0=meta["M0"],
            clip_percentile=float(args.clip_percentile),
        )

        orbit_json_path = Path(args.out) / f"{args.name}_orbit.json"
        residuals_path = Path(args.out) / f"{args.name}_residuals.bin"

        orbit_json = {
            "a": float(meta["a"]),
            "e": float(meta["e"]),
            "i": float(meta["i"]),
            "w": float(meta["w"]),
            "O": float(meta["O"]),
            "M0": float(meta["M0"]),
            "residual_scale": float(residual_scale),
            "count": int(packed.size),
        }

        with open(orbit_json_path, "w", encoding="utf-8") as f:
            json.dump(orbit_json, f, indent=2)
            f.write("\n")

        with open(residuals_path, "wb") as f:
            f.write(packed.astype("<u4", copy=False).tobytes(order="C"))

        brain_info = compiler.export_neural_brain(output_name="neural_decoder.bin", seed=int(args.seed))

        print(json.dumps(
            {
                "orbit_json": str(orbit_json_path),
                "residuals_bin": str(residuals_path),
                "neural_decoder_bin": brain_info["output"],
                "counts": {
                    "residuals": int(packed.size),
                    "weights_floats": int(brain_info["floats"]),
                },
                "notes": {
                    "time_unit": "days",
                    "pos_unit": "AU",
                },
            },
            indent=2,
        ))
        return 0

    if args.cmd == "compile-physics":
        if args.input is None:
            t, xyz = _demo_orbit(args.mu)
        else:
            t, xyz = _load_timeseries(Path(args.input), order=args.order)

        unit_time_scale = 1.0 if args.time_unit == "seconds" else 86400.0
        if args.pos_unit == "meters":
            unit_pos_scale = 1.0
        elif args.pos_unit == "km":
            unit_pos_scale = 1000.0
        else:
            unit_pos_scale = AU_METERS

        time_scale = float(args.time_scale) if args.time_scale is not None else unit_time_scale
        pos_scale = float(args.pos_scale) if args.pos_scale is not None else unit_pos_scale

        info = compiler.compile_physics(
            t,
            xyz,
            output_name=args.output,
            meta_output_name=args.meta_output,
            include_header=bool(args.include_header),
            time_scale=time_scale,
            pos_scale=pos_scale,
            radial_step=args.radial_step,
            transverse_step=args.transverse_step,
            clip_percentile=args.clip_percentile,
            max_nfev=args.max_nfev,
        )
        print(json.dumps(info, indent=2))
        return 0

    if args.cmd == "build-registry":
        out = compiler.build_registry(output_name=args.output)
        print(out)
        return 0

    if args.cmd == "export-neural-brain":
        info = compiler.export_neural_brain(output_name=args.output, seed=args.seed, state_dict_path=args.load_pth)
        print(json.dumps(info, indent=2))
        return 0

    raise RuntimeError("Unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())
