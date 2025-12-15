#!/usr/bin/env python3
"""Train the 35→64→64→4 asteroid MLP from a real 3D mesh.

Replaces the procedural `get_ground_truth()` function with a mesh-based Signed
Distance Field (SDF) query using `trimesh`.

Important contracts (must match WGSL shader + asset loader):
- Network architecture: 35 -> 64 -> 64 -> 4
- Export format: little-endian float32, flattened in this order:
  layer1.weight, layer1.bias, layer2.weight, layer2.bias, output.weight, output.bias
- Output meaning: [R,G,B, displacement]
  - RGB: sigmoid(0..1)
  - displacement: raw float (unbounded)

We train displacement as: displacement = -sdf * displacement_scale
so that positive SDF outside the shape becomes negative displacement (push inward)
when evaluated on a base unit-sphere mesh.

Example:
  python3 train_real_shape.py --mesh assets/eros.obj --output assets/neural_decoder.bin

Dependencies:
  pip install trimesh scipy rtree

Notes:
- `trimesh` signed distance requires a reasonably well-formed mesh. If true SDF
  fails, we fall back to unsigned distance + inside test.
- This script streams batches; over many epochs it will evaluate millions of SDF points.
"""

from __future__ import annotations

import argparse
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as e:
    print("Error: Required packages not found.", file=sys.stderr)
    print(f"Missing: {e}", file=sys.stderr)
    print("\nInstall (example):", file=sys.stderr)
    print("  pip install numpy torch", file=sys.stderr)
    print("  pip install trimesh scipy rtree", file=sys.stderr)
    sys.exit(1)

try:
    import trimesh
except ImportError as e:
    print("Error: trimesh is required for mesh SDF training.", file=sys.stderr)
    print(f"Missing: {e}", file=sys.stderr)
    print("\nInstall:")
    print("  pip install trimesh scipy rtree")
    sys.exit(1)

VERBOSE: int = 0


def log(level: int, msg: str) -> None:
    """Print `msg` when VERBOSE >= level."""
    if VERBOSE >= level:
        print(msg, flush=True)


def _format_bytes(n: float) -> str:
    if not np.isfinite(n) or n < 0:
        return "n/a"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while n >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f}{units[i]}"


def _proc_rss_bytes() -> Optional[int]:
    """Current process RSS from /proc (Linux)."""
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            parts = f.read().strip().split()
        if len(parts) < 2:
            return None
        rss_pages = int(parts[1])
        page = int(getattr(os, "sysconf")("SC_PAGE_SIZE"))
        return rss_pages * page
    except Exception:
        return None


def resource_line(device: torch.device) -> str:
    """Return a compact resource usage string."""
    pieces: list[str] = []

    # CPU / process memory
    rss = _proc_rss_bytes()
    if rss is not None:
        pieces.append(f"proc_rss={_format_bytes(float(rss))}")

    # Peak RSS (ru_maxrss is KiB on Linux, bytes on macOS)
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        maxrss = float(ru.ru_maxrss)
        # Linux: KiB, macOS: bytes
        if sys.platform.startswith("linux"):
            maxrss *= 1024.0
        pieces.append(f"proc_rss_peak={_format_bytes(maxrss)}")
    except Exception:
        pass

    # Optional psutil for CPU% and system RAM
    try:
        import psutil  # type: ignore

        p = psutil.Process()
        # CPU percent requires a prior call; still useful-ish if user runs longer.
        cpu_pct = p.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        pieces.append(f"cpu%={cpu_pct:.1f}")
        pieces.append(f"ram={_format_bytes(float(vm.used))}/{_format_bytes(float(vm.total))} ({vm.percent:.0f}%)")
    except Exception:
        pass

    # GPU memory (works without extra deps)
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            i = int(torch.cuda.current_device())
            alloc = float(torch.cuda.memory_allocated(i))
            reserv = float(torch.cuda.memory_reserved(i))
            total = None
            try:
                props = torch.cuda.get_device_properties(i)
                total = float(getattr(props, "total_memory", 0.0))
            except Exception:
                total = None

            if total and total > 0:
                pieces.append(f"vram_alloc={_format_bytes(alloc)}/{_format_bytes(total)}")
                pieces.append(f"vram_resv={_format_bytes(reserv)}/{_format_bytes(total)}")
            else:
                pieces.append(f"vram_alloc={_format_bytes(alloc)}")
                pieces.append(f"vram_resv={_format_bytes(reserv)}")
        except Exception:
            pass

    if not pieces:
        return "resources=n/a"
    return " ".join(pieces)


def _rgb_from_xyz_torch(xyz: torch.Tensor) -> torch.Tensor:
    """Cheap, smooth color function (deterministic) used as a stand-in texture."""
    base = 0.5 + 0.5 * torch.sin(xyz * 10.0)
    return torch.sigmoid(base)


def _make_targets_from_xyz_sdf(
    xyz: torch.Tensor,
    sdf: torch.Tensor,
    *,
    displacement_scale: float,
) -> torch.Tensor:
    """Targets: [rgb, displacement], where displacement = -sdf * scale."""
    rgb = _rgb_from_xyz_torch(xyz)
    displacement = (-sdf * float(displacement_scale)).unsqueeze(1)
    return torch.cat([rgb, displacement], dim=1)


def _device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


class AsteroidDecoder(nn.Module):
    """MLP matching the WGSL shader architecture exactly."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(35, 64)
        self.layer2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 4)

        # Stable init (similar spirit to train_asteroid.py)
        nn.init.xavier_uniform_(self.layer1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.layer2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.output.weight, gain=0.5)
        nn.init.constant_(self.layer1.bias, 0.0)
        nn.init.constant_(self.layer2.bias, 0.0)
        nn.init.constant_(self.output.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad xyz -> 35 dims (shader currently uses only xyz; code dims are 0)
        if x.shape[1] == 3:
            padding = torch.zeros(x.shape[0], 32, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)

        h1 = torch.relu(self.layer1(x))
        h2 = torch.relu(self.layer2(h1))
        out = self.output(h2)

        rgb = torch.sigmoid(out[:, :3])
        displacement = out[:, 3:4]
        return torch.cat([rgb, displacement], dim=1)


def load_and_normalize_mesh(path: Path, *, radius: float = 0.95, process: bool = True) -> trimesh.Trimesh:
    print(f"[Mesh] Loading {path}...")
    mesh = trimesh.load(str(path), force=None, process=process)

    # If it's a Scene, grab the first geometry
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError(f"Mesh scene has no geometry: {path}")
        mesh = list(mesh.geometry.values())[0]

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(mesh)}")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {path}")

    log(1, f"[Mesh] verts={int(mesh.vertices.shape[0])} faces={int(mesh.faces.shape[0])} watertight={bool(mesh.is_watertight)}")
    try:
        b = mesh.bounds
        log(2, f"[Mesh] bounds(min={b[0].tolist()}, max={b[1].tolist()})")
    except Exception:
        pass
    try:
        log(2, f"[Mesh] euler_number={getattr(mesh, 'euler_number', None)} is_winding_consistent={bool(getattr(mesh, 'is_winding_consistent', False))}")
    except Exception:
        pass

    # Center around origin (use center of mass if available)
    try:
        center = mesh.center_mass
    except Exception:
        center = mesh.centroid

    mesh.vertices = mesh.vertices - center
    try:
        log(2, f"[Mesh] centered by {np.asarray(center, dtype=np.float64).tolist()}")
    except Exception:
        pass

    # Scale so max vertex distance is `radius`
    max_dist = float(np.max(np.linalg.norm(mesh.vertices, axis=1)))
    if not np.isfinite(max_dist) or max_dist <= 0.0:
        raise ValueError(f"Invalid mesh scale (max_dist={max_dist})")

    scale = float(radius) / max_dist
    mesh.vertices = mesh.vertices * scale

    print(f"[Mesh] Normalized: scale={scale:.6f}, watertight={bool(mesh.is_watertight)}")
    log(2, f"[Mesh] target_radius={float(radius):.6f}, max_dist_pre={max_dist:.6f}, max_dist_post={float(np.max(np.linalg.norm(mesh.vertices, axis=1))):.6f}")
    try:
        log(2, f"[Mesh] volume={float(mesh.volume):.6f} (units^3), area={float(mesh.area):.6f} (units^2)")
    except Exception:
        pass
    return mesh


class SDFQuery:
    """Cache-heavy SDF helper to avoid rebuilding acceleration structures per batch."""

    def __init__(self, mesh: trimesh.Trimesh, *, prebuild: bool = True) -> None:
        self.mesh = mesh
        self.pq = trimesh.proximity.ProximityQuery(mesh)
        self._backend_reported = False
        self._sign_flip = False
        self._inside_engine: str = "unknown"
        if prebuild:
            self.prebuild()
        self._calibrate_sign_once()

    def prebuild(self) -> None:
        """Force-build trimesh caches so epoch 0 doesn't pay everything."""
        t0 = time.perf_counter()
        # Spatial index for proximity queries (rtree)
        try:
            _ = self.mesh.triangles_tree
            log(2, "[SDF] prebuild triangles_tree: ok")
        except ModuleNotFoundError as e:
            if getattr(e, "name", "") == "rtree":
                raise RuntimeError(
                    "trimesh requires the optional dependency 'rtree' for proximity / distance queries.\n"
                    "Install it in your environment:\n"
                    "  pip install rtree\n"
                ) from e
        except Exception as e:
            log(2, f"[SDF] prebuild triangles_tree: skipped ({e})")

        # Ray engine init for contains() (may use embree/embreex if installed)
        try:
            _ = self.mesh.contains(np.zeros((1, 3), dtype=np.float64))
            self._inside_engine = "mesh.contains"
            log(2, "[SDF] prebuild contains(): ok")
        except Exception as e:
            log(2, f"[SDF] prebuild contains(): skipped ({e})")

        log(1, f"[SDF] prebuild_time={(time.perf_counter() - t0):.2f}s")

    def _try_signed(self, points: np.ndarray) -> np.ndarray:
        if hasattr(self.pq, "signed_distance"):
            return np.asarray(self.pq.signed_distance(points), dtype=np.float64)
        return np.asarray(trimesh.proximity.signed_distance(self.mesh, points), dtype=np.float64)

    def _unsigned(self, points: np.ndarray) -> np.ndarray:
        if hasattr(self.pq, "distance"):
            return np.asarray(self.pq.distance(points), dtype=np.float64)
        closest, _dist, _tri = trimesh.proximity.closest_point(self.mesh, points)
        return np.linalg.norm(points - closest, axis=1).astype(np.float64, copy=False)

    def _inside(self, points: np.ndarray) -> Optional[np.ndarray]:
        try:
            if self.mesh.is_watertight:
                return np.asarray(self.mesh.contains(points), dtype=bool)
        except Exception:
            pass
        try:
            hull = self.mesh.convex_hull
            return np.asarray(hull.contains(points), dtype=bool)
        except Exception:
            return None

    def _calibrate_sign_once(self) -> None:
        """Try to ensure negative inside at the origin (only once)."""
        try:
            origin = np.zeros((1, 3), dtype=np.float64)
            inside0 = self._inside(origin)
            if inside0 is None:
                return
            inside_origin = bool(inside0[0])
            if not inside_origin:
                return
            sdf0 = float(self(origin, calibrate=False)[0])
            # If origin is inside and SDF positive, flip globally.
            if sdf0 > 0.0:
                self._sign_flip = True
        except Exception:
            return

    def __call__(self, points: np.ndarray, *, calibrate: bool = False) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        t0 = time.perf_counter()
        backend = "unknown"

        sdf: Optional[np.ndarray] = None
        try:
            sdf = self._try_signed(points)
            backend = "trimesh_signed_distance"
        except ModuleNotFoundError as e:
            if getattr(e, "name", "") == "rtree":
                raise RuntimeError(
                    "trimesh requires the optional dependency 'rtree' for proximity / distance queries.\n"
                    "Install it in your environment:\n"
                    "  pip install rtree\n"
                ) from e
        except Exception:
            sdf = None

        if sdf is None or not np.all(np.isfinite(sdf)):
            unsigned = self._unsigned(points)
            inside = self._inside(points)
            if inside is None:
                raise RuntimeError(
                    "Failed to compute signed distance: trimesh signed_distance failed and no reliable inside-test is available. "
                    "Try providing a watertight mesh (or .stl/.ply), or simplify the mesh."
                )
            sdf = unsigned
            sdf[inside] *= -1.0
            backend = "unsigned_distance+inside_test"

        if self._sign_flip:
            sdf = -sdf
            backend = backend + " (flipped_sign)"

        if not self._backend_reported:
            self._backend_reported = True
            log(1, f"[SDF] backend={backend} inside_engine={self._inside_engine}")

        log(3, f"[SDF] queried {int(points.shape[0])} pts in {(time.perf_counter() - t0)*1000.0:.2f} ms")
        return sdf.astype(np.float64, copy=False)


def signed_distance(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Back-compat wrapper (uncached). Prefer using `SDFQuery(mesh)` for speed."""
    return SDFQuery(mesh, prebuild=False)(points)


def _require_scipy() -> None:
    try:
        import scipy  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "SciPy is required for voxel SDF grid mode.\n"
            "Install:\n"
            "  pip install scipy\n"
        ) from e


def precompute_sdf_grid(
    *,
    mesh: trimesh.Trimesh,
    grid_res: int,
    grid_padding: int,
    out_path: Path,
) -> Path:
    """Bake a mesh into a signed distance volume using voxelization + distance transform.

    Output .npz contains:
    - sdf: float32 (nx, ny, nz) in voxel index order (x,y,z)
    - index_to_world: float64 (4,4) maps voxel index -> world (normalized mesh space)
    - world_to_index: float64 (4,4) inverse transform
    - pitch: float32 voxel size
    - dims: int32 (3,) [nx, ny, nz]
    """
    _require_scipy()
    from scipy.ndimage import binary_fill_holes, distance_transform_edt  # type: ignore

    grid_res = int(grid_res)
    grid_padding = int(grid_padding)
    if grid_res < 16:
        raise ValueError("--grid-res must be >= 16")
    if grid_padding < 0:
        raise ValueError("--grid-padding must be >= 0")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a fixed cube grid that covers at least [-1, 1]^3.
    # We interpret `grid_padding` as extra margin (in voxels) beyond [-1,1].
    base_bounds = 1.0
    denom = 1.0 - (2.0 * float(grid_padding) / float(max(1, grid_res - 1)))
    if denom <= 0.0:
        bounds = 1.1
    else:
        bounds = base_bounds / denom
    pitch = (2.0 * bounds) / float(max(1, grid_res - 1))

    print(f"[Grid] Voxelizing into fixed grid: res={grid_res} bounds={bounds:.6f} pitch={pitch:.6f}")
    t0 = time.perf_counter()
    vg = mesh.voxelized(pitch=float(pitch))
    vg_solid = vg.fill()

    # Rasterize filled voxels into our fixed dense grid (O(n_voxels), no res^3 contains()).
    origin = np.array([-bounds, -bounds, -bounds], dtype=np.float64)
    solid = np.zeros((grid_res, grid_res, grid_res), dtype=bool)  # x,y,z order
    pts = np.asarray(vg_solid.points, dtype=np.float64)  # voxel centers in world space
    if pts.size == 0:
        raise RuntimeError("Voxelization produced no filled voxels. Check mesh scale/normalization.")

    ijk = np.round((pts - origin[None, :]) / float(pitch)).astype(np.int64)
    keep = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < grid_res)
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < grid_res)
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < grid_res)
    )
    ijk = ijk[keep]
    solid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    # Fill any remaining small voids.
    solid = binary_fill_holes(solid)

    # Signed distance via two distance transforms:
    # outside: distance to nearest solid voxel
    # inside:  distance to nearest empty voxel
    dt_out = distance_transform_edt(~solid).astype(np.float32, copy=False)
    dt_in = distance_transform_edt(solid).astype(np.float32, copy=False)
    sdf = (dt_out - dt_in) * np.float32(pitch)

    dims = np.array(sdf.shape, dtype=np.int32)  # (res,res,res) x,y,z order
    # Fixed grid transform: world = origin + pitch * idx
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = float(pitch)
    T[1, 1] = float(pitch)
    T[2, 2] = float(pitch)
    T[:3, 3] = origin
    world_to_index = np.linalg.inv(T).astype(np.float64, copy=False)

    np.savez_compressed(
        out_path,
        sdf=sdf.astype(np.float32, copy=False),
        index_to_world=T,
        world_to_index=world_to_index,
        pitch=np.float32(pitch),
        dims=dims,
        axis_order=np.array(["x", "y", "z"]),
    )

    dt = time.perf_counter() - t0
    print(f"[Grid] Built sdf grid dims={tuple(int(x) for x in dims)} in {dt:.2f}s -> {out_path}")

    # Sanity check at origin (nearest-voxel sample)
    try:
        idx0 = (world_to_index @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))[:3]
        idxn = np.round(idx0).astype(np.int64)
        idxn = np.clip(idxn, 0, dims - 1)
        s0 = float(sdf[int(idxn[0]), int(idxn[1]), int(idxn[2])])
        print(f"[Grid] SDF(origin)~{s0:.6f} (negative means inside)")
    except Exception:
        pass

    return out_path


class VoxelSDFQueryNP:
    """Fast trilinear interpolation on CPU using SciPy map_coordinates."""

    def __init__(self, npz_path: Path) -> None:
        _require_scipy()
        from scipy.ndimage import map_coordinates  # type: ignore

        self._map_coordinates = map_coordinates
        data = np.load(str(npz_path))
        self.sdf = np.asarray(data["sdf"], dtype=np.float32)  # (x,y,z)
        self.world_to_index = np.asarray(data["world_to_index"], dtype=np.float64)
        self.dims = tuple(int(x) for x in np.asarray(data["dims"], dtype=np.int64))

    def __call__(self, points_xyz: np.ndarray) -> np.ndarray:
        points_xyz = np.asarray(points_xyz, dtype=np.float64)
        ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
        p4 = np.concatenate([points_xyz, ones], axis=1)
        ijk = (p4 @ self.world_to_index.T)[:, :3].T  # shape (3, N) in x,y,z index order
        # Trilinear interpolation
        out = self._map_coordinates(self.sdf, ijk, order=1, mode="nearest").astype(np.float32, copy=False)
        return out


class VoxelSDFQueryTorch:
    """Fast trilinear interpolation on GPU using torch.grid_sample."""

    def __init__(self, npz_path: Path, *, device: torch.device) -> None:
        data = np.load(str(npz_path))
        sdf = np.asarray(data["sdf"], dtype=np.float32)  # (x,y,z)
        w2i = np.asarray(data["world_to_index"], dtype=np.float64)
        dims = np.asarray(data["dims"], dtype=np.int64)

        self.nx, self.ny, self.nz = int(dims[0]), int(dims[1]), int(dims[2])
        self.device = device

        # grid_sample expects (N,C,D,H,W) with coords order (x,y,z) mapping to (W,H,D).
        # Our sdf is stored (x,y,z), so permute to (z,y,x) for (D,H,W).
        sdf_t = torch.from_numpy(sdf).to(device=device, dtype=torch.float32)
        self.grid_tex = sdf_t.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)  # (1,1,nz,ny,nx)

        w2i_t = torch.from_numpy(w2i).to(device=device, dtype=torch.float32)
        self.world_to_index = w2i_t  # (4,4)

    def __call__(self, points_xyz: torch.Tensor) -> torch.Tensor:
        # points_xyz: (N,3) in world space.
        n = points_xyz.shape[0]
        ones = torch.ones((n, 1), device=points_xyz.device, dtype=points_xyz.dtype)
        p4 = torch.cat([points_xyz, ones], dim=1)  # (N,4)
        ijk = (p4 @ self.world_to_index.t())[:, :3]  # (N,3) in x,y,z index coords

        # Normalize to [-1,1] for grid_sample with align_corners=True.
        # x -> W (nx), y -> H (ny), z -> D (nz)
        nxm1 = float(max(1, self.nx - 1))
        nym1 = float(max(1, self.ny - 1))
        nzm1 = float(max(1, self.nz - 1))
        gx = (ijk[:, 0] / nxm1) * 2.0 - 1.0
        gy = (ijk[:, 1] / nym1) * 2.0 - 1.0
        gz = (ijk[:, 2] / nzm1) * 2.0 - 1.0
        grid = torch.stack([gx, gy, gz], dim=1).view(1, n, 1, 1, 3)

        sdf = torch.nn.functional.grid_sample(
            self.grid_tex,
            grid,
            mode="bilinear",  # trilinear for 5D
            padding_mode="border",
            align_corners=True,
        )
        return sdf.view(-1)  # (N,)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _dataset_paths(dataset_dir: Path) -> Dict[str, Path]:
    return {
        "xyz": dataset_dir / "xyz.npy",
        "sdf": dataset_dir / "sdf.npy",
        "meta": dataset_dir / "meta.json",
    }


def precompute_sdf_dataset(
    *,
    mesh: trimesh.Trimesh,
    sdfq,
    out_dir: Path,
    total_points: int,
    chunk_points: int,
    sdf_batch_points: int,
    seed: int,
    near_sigma: float,
    frac_mesh_near: float,
    frac_unit_sphere: float,
    frac_unit_ball: float,
) -> None:
    """Precompute (xyz, sdf) dataset to out_dir/xyz.npy + out_dir/sdf.npy + meta.json.

    Storage format:
    - xyz.npy: float32, shape (N, 3)
    - sdf.npy: float32, shape (N,)
    """
    import json
    from numpy.lib.format import open_memmap

    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    paths = _dataset_paths(out_dir)

    n = int(total_points)
    if n <= 0:
        raise ValueError("total_points must be > 0")
    chunk = int(chunk_points)
    if chunk <= 0:
        raise ValueError("chunk_points must be > 0")
    sdf_bs = int(sdf_batch_points)
    if sdf_bs <= 0:
        raise ValueError("sdf_batch_points must be > 0")

    # Deterministic precompute
    rng = np.random.default_rng(int(seed))

    log(1, f"[Dataset] writing to {out_dir}")
    log(1, f"[Dataset] points={n:,} chunk={chunk:,} bytes≈{_format_bytes(float(n) * 16.0)}")
    log(1, f"[Dataset] sdf_batch={sdf_bs:,} (limits peak memory during trimesh queries)")
    if chunk > 100_000:
        log(1, "[Dataset] warning: large --dataset-chunk can use lots of RAM during SDF; consider 50k")

    xyz_mm = open_memmap(str(paths["xyz"]), mode="w+", dtype=np.float32, shape=(n, 3))
    sdf_mm = open_memmap(str(paths["sdf"]), mode="w+", dtype=np.float32, shape=(n,))

    t0 = time.perf_counter()
    written = 0
    while written < n:
        m = min(chunk, n - written)

        # Match the same mixture strategy as online batching.
        n_mesh_near = int(round(m * float(frac_mesh_near)))
        n_sphere = int(round(m * float(frac_unit_sphere)))
        n_ball = m - n_mesh_near - n_sphere
        if n_ball < 0:
            n_ball = 0
            n_sphere = m - n_mesh_near

        pts = []
        if n_mesh_near > 0:
            # Prefer true mesh surface sampling when available; otherwise fall back
            # to unit-sphere surface fuzz (works for voxel-grid-only mode).
            if getattr(mesh, "faces", None) is not None and int(getattr(mesh, "faces").shape[0]) > 0:
                surf, _ = trimesh.sample.sample_surface(mesh, n_mesh_near)
                noise = rng.normal(0.0, float(near_sigma), size=surf.shape).astype(np.float64)
                pts.append(surf.astype(np.float64, copy=False) + noise)
            else:
                x = rng.normal(size=(n_mesh_near, 3)).astype(np.float64)
                x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
                x = x + rng.normal(0.0, float(near_sigma), size=x.shape).astype(np.float64)
                pts.append(x)
        if n_sphere > 0:
            x = rng.normal(size=(n_sphere, 3)).astype(np.float64)
            x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
            pts.append(x)
        if n_ball > 0 and float(frac_unit_ball) > 0.0:
            # rejection sample in cube
            out = []
            need = int(n_ball)
            while need > 0:
                cand = rng.uniform(-1.0, 1.0, size=(need * 2, 3)).astype(np.float64)
                mask = (np.sum(cand * cand, axis=1) <= 1.0)
                sel = cand[mask]
                if sel.shape[0] > need:
                    sel = sel[:need]
                out.append(sel)
                need -= sel.shape[0]
            pts.append(np.concatenate(out, axis=0))
        elif n_ball > 0:
            pts.append(rng.uniform(-1.0, 1.0, size=(n_ball, 3)).astype(np.float64))

        query_points = np.concatenate(pts, axis=0).astype(np.float64, copy=False)

        # Shuffle within chunk so distribution is mixed
        rng.shuffle(query_points, axis=0)

        # Compute SDF in smaller batches to avoid huge peak allocations inside trimesh.
        sdf = np.empty((m,), dtype=np.float32)
        for j0 in range(0, m, sdf_bs):
            j1 = min(m, j0 + sdf_bs)
            sdf[j0:j1] = np.asarray(sdfq(query_points[j0:j1]), dtype=np.float32)
        xyz_mm[written : written + m] = query_points.astype(np.float32, copy=False)
        sdf_mm[written : written + m] = sdf
        written += m

        if VERBOSE >= 1:
            elapsed = time.perf_counter() - t0
            rate = written / max(elapsed, 1e-9)
            log(1, f"[Dataset] {written:,}/{n:,} ({written/n*100.0:.1f}%) @ {rate:,.0f} pts/s")

    # Flush memmaps
    del xyz_mm
    del sdf_mm

    meta = {
        "version": 1,
        "points": int(n),
        "seed": int(seed),
        "mix": {
            "frac_mesh_near": float(frac_mesh_near),
            "frac_unit_sphere": float(frac_unit_sphere),
            "frac_unit_ball": float(frac_unit_ball),
            "near_sigma": float(near_sigma),
        },
        "sdf_batch_points": int(sdf_bs),
        "files": {k: str(v.name) for k, v in paths.items()},
        "notes": "xyz float32 (N,3), sdf float32 (N,), units in normalized mesh space",
    }
    paths["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log(1, f"[Dataset] done in {(time.perf_counter() - t0):.2f}s")


class MemmapSDFDataset(torch.utils.data.Dataset):  # type: ignore[attr-defined]
    """Dataset backed by numpy .npy memmaps: xyz.npy + sdf.npy."""

    def __init__(self, dataset_dir: Path) -> None:
        dataset_dir = Path(dataset_dir)
        paths = _dataset_paths(dataset_dir)
        if not paths["xyz"].exists() or not paths["sdf"].exists():
            raise FileNotFoundError(f"Dataset missing xyz.npy/sdf.npy in {dataset_dir}")
        self.xyz = np.load(paths["xyz"], mmap_mode="r")
        self.sdf = np.load(paths["sdf"], mmap_mode="r")
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError("xyz.npy must have shape (N,3)")
        if self.sdf.ndim != 1 or self.sdf.shape[0] != self.xyz.shape[0]:
            raise ValueError("sdf.npy must have shape (N,) and match xyz length")

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # Return numpy arrays; DataLoader will stack/convert efficiently.
        return self.xyz[idx], self.sdf[idx]


class RandomBatchSDFIterable(torch.utils.data.IterableDataset):  # type: ignore[attr-defined]
    """High-throughput batch sampler for memmap datasets.

    Avoids per-item __getitem__ + collate overhead by sampling an entire batch
    of indices at once and returning already-batched arrays.
    """

    def __init__(
        self,
        dataset_dir: Path,
        *,
        batch_size: int,
        steps: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = int(batch_size)
        self.steps = int(steps)
        self.seed = int(seed)

    def __iter__(self):
        # Worker-aware RNG so workers don't produce identical batches.
        worker = torch.utils.data.get_worker_info()  # type: ignore[attr-defined]
        wid = int(worker.id) if worker is not None else 0
        rng = np.random.default_rng(self.seed + 1009 * wid)

        paths = _dataset_paths(self.dataset_dir)
        xyz = np.load(paths["xyz"], mmap_mode="r")
        sdf = np.load(paths["sdf"], mmap_mode="r")
        n = int(xyz.shape[0])

        for _ in range(self.steps):
            idx = rng.integers(0, n, size=(self.batch_size,), endpoint=False)
            # Make arrays writable to avoid torch warning.
            xb = np.asarray(xyz[idx], dtype=np.float32).copy()
            sb = np.asarray(sdf[idx], dtype=np.float32).copy()
            yield xb, sb


def _sample_points_on_unit_sphere(n: int) -> np.ndarray:
    x = np.random.normal(size=(n, 3)).astype(np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x


def _sample_points_in_unit_ball(n: int) -> np.ndarray:
    # Rejection sampling in cube [-1,1]^3
    out = []
    need = int(n)
    while need > 0:
        cand = np.random.uniform(-1.0, 1.0, size=(need * 2, 3)).astype(np.float64)
        mask = (np.sum(cand * cand, axis=1) <= 1.0)
        sel = cand[mask]
        if sel.shape[0] > need:
            sel = sel[:need]
        out.append(sel)
        need -= sel.shape[0]
    return np.concatenate(out, axis=0)


def get_training_batch(
    mesh: trimesh.Trimesh,
    batch_size: int,
    *,
    device: torch.device,
    sdf_query: Optional[object] = None,
    near_sigma: float,
    frac_mesh_near: float,
    frac_unit_sphere: float,
    frac_unit_ball: float,
    displacement_scale: float,
    sdf_batch: int = 262144,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate one training batch.

    Mix:
    - Points near the mesh surface (sample on mesh + Gaussian noise)
    - Points on the unit sphere surface (matches shader query distribution)
    - Random points inside unit ball (global SDF supervision)

    Targets: [rgb, displacement], where displacement = -sdf * displacement_scale.
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    # Compute counts (ensure sum == batch_size)
    n_mesh_near = int(round(batch_size * float(frac_mesh_near)))
    n_sphere = int(round(batch_size * float(frac_unit_sphere)))
    n_ball = batch_size - n_mesh_near - n_sphere
    if n_ball < 0:
        # Rebalance by stealing from sphere
        n_ball = 0
        n_sphere = batch_size - n_mesh_near

    pts = []

    if n_mesh_near > 0:
        surf, _ = trimesh.sample.sample_surface(mesh, n_mesh_near)
        noise = np.random.normal(0.0, float(near_sigma), size=surf.shape).astype(np.float64)
        pts.append(surf.astype(np.float64, copy=False) + noise)

    if n_sphere > 0:
        pts.append(_sample_points_on_unit_sphere(n_sphere))

    if n_ball > 0 and float(frac_unit_ball) > 0.0:
        pts.append(_sample_points_in_unit_ball(n_ball))
    elif n_ball > 0:
        # Fallback: uniform cube if user disables ball.
        pts.append(np.random.uniform(-1.0, 1.0, size=(n_ball, 3)).astype(np.float64))

    query_points = np.concatenate(pts, axis=0).astype(np.float64, copy=False)

    # Signed distance (batched to avoid huge allocations if batch_size is large)
    t_sdf = time.perf_counter()
    sdf = np.empty((query_points.shape[0],), dtype=np.float64)
    for start in range(0, query_points.shape[0], int(sdf_batch)):
        end = min(query_points.shape[0], start + int(sdf_batch))
        if sdf_query is None:
            sdf[start:end] = signed_distance(mesh, query_points[start:end])
        else:
            # `sdf_query` is expected to be a callable like `SDFQuery`
            sdf[start:end] = sdf_query(query_points[start:end])  # type: ignore[misc]
    sdf_ms = (time.perf_counter() - t_sdf) * 1000.0

    # Target displacement matches shader usage
    disp = (-sdf * float(displacement_scale)).astype(np.float32)

    # Simple deterministic-ish procedural color (not physically based)
    # Keep it bounded and smooth so the model can learn it cheaply.
    qp32 = query_points.astype(np.float32, copy=False)
    base = 0.5 + 0.5 * np.sin(qp32 * 10.0)
    rgb = 1.0 / (1.0 + np.exp(-base))  # sigmoid

    inputs_t = torch.from_numpy(qp32).to(device=device, dtype=torch.float32)
    targets_t = torch.cat(
        [
            torch.from_numpy(rgb).to(device=device, dtype=torch.float32),
            torch.from_numpy(disp).to(device=device, dtype=torch.float32).unsqueeze(1),
        ],
        dim=1,
    )

    # Lightweight stats for debugging
    if VERBOSE >= 2:
        inside_frac = float(np.mean(sdf < 0.0)) if sdf.size else 0.0
        log(
            2,
            "[Batch] "
            f"counts(mesh_near={n_mesh_near}, unit_sphere={n_sphere}, other={n_ball}), "
            f"sdf(min={float(np.min(sdf)):.4f}, mean={float(np.mean(sdf)):.4f}, max={float(np.max(sdf)):.4f}, inside={inside_frac*100.0:.1f}%), "
            f"disp(min={float(np.min(disp)):.4f}, mean={float(np.mean(disp)):.4f}, max={float(np.max(disp)):.4f}), "
            f"sdf_time={sdf_ms:.1f}ms"
        )

    return inputs_t, targets_t


def export_weights(model: nn.Module, output_path: Path) -> None:
    """Export model weights to binary file matching data_compiler.py format."""
    model.eval()

    # Collect parameters in registration order (matches data_compiler.py exactly)
    flat_buffer = []

    print("[Export] Packing weights:")
    for name, param in model.named_parameters():
        data = param.detach().cpu().contiguous().numpy().astype(np.float32, copy=False).reshape(-1)
        flat_buffer.append(data)
        print(f"  -> {name}: {data.size} elements (shape {list(param.shape)})")

    flat = np.concatenate(flat_buffer).astype("<f4", copy=False)

    expected_total = 64 * 35 + 64 + 64 * 64 + 64 + 4 * 64 + 4  # 6724
    if flat.size != expected_total:
        raise ValueError(f"Weight count mismatch: got {flat.size}, expected {expected_total}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(flat.tobytes(order="C"))

    size = output_path.stat().st_size
    if (size % 4) != 0:
        raise RuntimeError(f"Output is not 4-byte aligned: {output_path} size={size}")

    print(f"[Export] Saved to: {output_path} ({flat.size} floats, {size} bytes)\n")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train asteroid decoder from a real mesh SDF")
    ap.add_argument("--mesh", type=str, default="assets/eros.obj", help="Path to mesh file (.obj/.ply/.stl)")
    ap.add_argument("--output", type=str, default="assets/neural_decoder.bin", help="Output weights path")
    ap.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    ap.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    ap.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v, -vv, -vvv).",
    )

    ap.add_argument("--mesh-radius", type=float, default=0.95, help="Normalize mesh max radius to this")
    ap.add_argument("--near-sigma", type=float, default=0.05, help="Std-dev for near-surface noise")
    ap.add_argument("--frac-mesh-near", type=float, default=0.5, help="Fraction of batch near mesh surface")
    ap.add_argument("--frac-unit-sphere", type=float, default=0.25, help="Fraction of batch on unit sphere surface")
    ap.add_argument("--frac-unit-ball", type=float, default=1.0, help="If >0, sample remaining points in unit ball")

    ap.add_argument("--disp-scale", type=float, default=0.5, help="displacement = -sdf * disp_scale")
    ap.add_argument("--disp-weight", type=float, default=10.0, help="Loss weight for displacement channel")
    ap.add_argument("--rgb-weight", type=float, default=1.0, help="Loss weight for RGB channels")
    ap.add_argument("--print-interval", type=int, default=100, help="Print loss every N epochs")
    ap.add_argument(
        "--stats-interval",
        type=int,
        default=250,
        help="When verbose, print extra batch stats every N epochs (default: 250).",
    )
    ap.add_argument(
        "--resource-interval",
        type=int,
        default=100,
        help="Print resource usage every N epochs (default: 100).",
    )
    ap.add_argument(
        "--prebuild-sdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prebuild trimesh acceleration structures before training (default: true).",
    )

    # Voxel-grid SDF mode (fast approximate supervision)
    ap.add_argument("--precompute-sdf-grid", action="store_true", help="Bake a voxel SDF grid (.npz) and exit.")
    ap.add_argument("--sdf-grid-out", type=str, default="assets/eros_sdf_grid_256.npz", help="Output path for baked SDF grid (.npz).")
    ap.add_argument("--grid-res", type=int, default=256, help="Target voxel resolution (pitch derived from this).")
    ap.add_argument("--grid-padding", type=int, default=4, help="Pad the voxel grid by this many voxels on each side.")
    ap.add_argument("--sdf-grid", type=str, default=None, help="Path to baked SDF grid (.npz) to use for training.")
    ap.add_argument("--verify-sdf-grid", action="store_true", help="Compare voxel-grid SDF vs mesh SDF on random points (needs --mesh and --sdf-grid).")

    # Fast path: precompute SDF dataset once, then train from disk.
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset dir (contains xyz.npy + sdf.npy). If set, training loads from disk instead of querying trimesh every epoch.",
    )
    ap.add_argument(
        "--precompute-dataset",
        action="store_true",
        help="Precompute the dataset specified by --dataset and exit (or train if --train-after-precompute is set).",
    )
    ap.add_argument(
        "--train-after-precompute",
        action="store_true",
        help="If --precompute-dataset is set, continue into training after writing dataset.",
    )
    ap.add_argument("--dataset-size", type=int, default=5_000_000, help="Points to precompute (default: 5,000,000)")
    ap.add_argument("--dataset-chunk", type=int, default=200_000, help="Points per write chunk (default: 200,000)")
    ap.add_argument(
        "--dataset-sdf-batch",
        type=int,
        default=50_000,
        help="Max points per trimesh SDF query during precompute (default: 50,000). Lower if you see RAM spikes/crashes.",
    )

    # DataLoader knobs (disk training)
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers for dataset training (default: 8)")
    ap.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True, help="Pin CPU memory for faster H2D copies (default: true)")
    ap.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor (default: 4)")
    ap.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True, help="Keep workers alive between epochs (default: true)")
    ap.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="When training from --dataset, number of random batches per epoch (default: dataset_len//batch).",
    )

    # GPU math knobs
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA AMP mixed precision (default: true)")
    ap.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="Enable TF32 matmul on Ampere+ GPUs (default: true)")

    args = ap.parse_args(argv)
    global VERBOSE
    VERBOSE = int(args.verbose or 0)

    mesh_path = Path(args.mesh)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = _device_from_arg(args.device)

    print("=" * 70)
    print("Mesh Neural SDF Trainer")
    print("=" * 70)
    print(f"Mesh: {mesh_path}")
    print(f"Output: {args.output}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Near sigma: {args.near_sigma}")
    print(f"Mix: mesh_near={args.frac_mesh_near}, unit_sphere={args.frac_unit_sphere}, unit_ball={'on' if args.frac_unit_ball>0 else 'off'}")
    print(f"Target: displacement = -sdf * {args.disp_scale}")
    print("=" * 70)

    log(1, f"[Env] python={sys.version.split()[0]} platform={platform.platform()}")
    log(1, f"[Env] numpy={np.__version__} torch={torch.__version__} trimesh={trimesh.__version__}")
    if device.type == "cuda":
        try:
            log(1, f"[Env] cuda_available={torch.cuda.is_available()} cuda_device={torch.cuda.get_device_name(0)}")
        except Exception:
            log(1, f"[Env] cuda_available={torch.cuda.is_available()}")

    if device.type == "cuda":
        # Make matmul faster on NVIDIA (4070 supports TF32). Safe for training.
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
            torch.backends.cudnn.allow_tf32 = bool(args.tf32)
            torch.set_float32_matmul_precision("high" if bool(args.tf32) else "highest")
            log(1, f"[Perf] tf32={'on' if bool(args.tf32) else 'off'}")
        except Exception:
            pass

    # Decide whether we actually need the mesh file.
    needs_mesh = bool(args.precompute_sdf_grid) or (args.sdf_grid is None and args.dataset is None) or (args.dataset is not None and bool(args.precompute_dataset) and args.sdf_grid is None) or bool(args.verify_sdf_grid)
    if needs_mesh and not mesh_path.exists():
        print(f"Error: mesh file not found: {mesh_path}", file=sys.stderr)
        return 2

    mesh: Optional[trimesh.Trimesh] = None
    sdfq: Optional[SDFQuery] = None
    if needs_mesh:
        mesh = load_and_normalize_mesh(mesh_path, radius=float(args.mesh_radius), process=True)
        sdfq = SDFQuery(mesh, prebuild=bool(args.prebuild_sdf))

    # Mode: bake voxel SDF grid and exit.
    if bool(args.precompute_sdf_grid):
        assert mesh is not None
        out = precompute_sdf_grid(
            mesh=mesh,
            grid_res=int(args.grid_res),
            grid_padding=int(args.grid_padding),
            out_path=Path(str(args.sdf_grid_out)),
        )
        print(f"[Grid] wrote {out}")
        return 0

    # Load voxel grid if requested.
    voxel_np: Optional[VoxelSDFQueryNP] = None
    voxel_torch: Optional[VoxelSDFQueryTorch] = None
    if args.sdf_grid is not None:
        voxel_np = VoxelSDFQueryNP(Path(str(args.sdf_grid)))
        voxel_torch = VoxelSDFQueryTorch(Path(str(args.sdf_grid)), device=device)
        log(1, f"[Grid] loaded {args.sdf_grid}")

    # Quick SDF sanity check
    try:
        if voxel_np is not None:
            s0 = float(voxel_np(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))[0])
            print(f"[Grid] SDF(origin) ~ {s0:.6f} (negative means inside)")
        elif sdfq is not None:
            s0 = float(sdfq(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))[0])
            print(f"[Mesh] SDF(origin) = {s0:.6f} (negative means inside)")
    except Exception as e:
        print(f"[SDF] Warning: SDF sanity check failed: {e}")

    # Optional verification: compare grid SDF vs mesh SDF on random points.
    if bool(args.verify_sdf_grid):
        if voxel_np is None or sdfq is None:
            print("Error: --verify-sdf-grid requires both --sdf-grid and a valid --mesh.", file=sys.stderr)
            return 2
        n = 2000
        pts = np.random.uniform(-1.1, 1.1, size=(n, 3)).astype(np.float64)
        t0 = time.perf_counter()
        s_mesh = sdfq(pts)
        t1 = time.perf_counter()
        s_grid = voxel_np(pts)
        t2 = time.perf_counter()
        err = s_grid - s_mesh.astype(np.float32)
        print(f"[Verify] mesh_time={(t1-t0)*1000.0:.1f}ms grid_time={(t2-t1)*1000.0:.1f}ms n={n}")
        print(f"[Verify] abs_err: mean={float(np.mean(np.abs(err))):.6f} p95={float(np.percentile(np.abs(err),95)):.6f} max={float(np.max(np.abs(err))):.6f}")
        # If the user asked for verification and set --epochs 0, treat this as verify-only.
        if int(args.epochs) == 0:
            return 0

    model = AsteroidDecoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))
    mse = nn.MSELoss()

    # Optional: precompute dataset.
    if args.dataset is not None and bool(args.precompute_dataset):
        ds_dir = Path(str(args.dataset))
        if voxel_np is not None:
            # Fast dataset precompute from voxel grid (no trimesh calls).
            precompute_sdf_dataset(
                mesh=mesh if mesh is not None else trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64)),
                sdfq=lambda pts: voxel_np(pts).astype(np.float64),  # type: ignore[arg-type]
                out_dir=ds_dir,
                total_points=int(args.dataset_size),
                chunk_points=int(args.dataset_chunk),
                sdf_batch_points=int(args.dataset_sdf_batch),
                seed=int(args.seed),
                near_sigma=float(args.near_sigma),
                frac_mesh_near=float(args.frac_mesh_near),
                frac_unit_sphere=float(args.frac_unit_sphere),
                frac_unit_ball=float(args.frac_unit_ball),
            )
        else:
            assert mesh is not None and sdfq is not None
            precompute_sdf_dataset(
                mesh=mesh,
                sdfq=sdfq,
                out_dir=ds_dir,
                total_points=int(args.dataset_size),
                chunk_points=int(args.dataset_chunk),
                sdf_batch_points=int(args.dataset_sdf_batch),
                seed=int(args.seed),
                near_sigma=float(args.near_sigma),
                frac_mesh_near=float(args.frac_mesh_near),
                frac_unit_sphere=float(args.frac_unit_sphere),
                frac_unit_ball=float(args.frac_unit_ball),
            )
        if not bool(args.train_after_precompute):
            return 0

    # Training loop
    use_dataset = args.dataset is not None
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and bool(args.amp)))  # type: ignore[attr-defined]

    model.train()
    t_train = time.perf_counter()
    ema_loss: Optional[float] = None

    if voxel_torch is not None and not use_dataset:
        # Train from voxel grid directly (no disk dataset, no trimesh).
        steps_per_epoch = int(args.steps_per_epoch) if args.steps_per_epoch is not None else 200
        log(1, f"[GridTrain] steps_per_epoch={steps_per_epoch}")

        for epoch in range(int(args.epochs)):
            t_epoch = time.perf_counter()
            loss_epoch = 0.0
            steps = 0

            for _ in range(steps_per_epoch):
                # Mix: half near surface (unit sphere fuzz), half volume (box)
                b = int(args.batch_size)
                p1 = torch.randn((b // 2, 3), device=device, dtype=torch.float32)
                p1 = torch.nn.functional.normalize(p1, dim=1)
                p1 = p1 + torch.randn_like(p1) * float(args.near_sigma)
                p2 = (torch.rand((b - b // 2, 3), device=device, dtype=torch.float32) * 2.0) - 1.0
                xyz = torch.cat([p1, p2], dim=0)

                sdf = voxel_torch(xyz)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and bool(args.amp))):  # type: ignore[attr-defined]
                    pred = model(xyz)
                    tgt = _make_targets_from_xyz_sdf(xyz, sdf, displacement_scale=float(args.disp_scale))
                    loss_disp = mse(pred[:, 3], tgt[:, 3]) * float(args.disp_weight)
                    loss_rgb = mse(pred[:, :3], tgt[:, :3]) * float(args.rgb_weight)
                    loss = loss_disp + loss_rgb

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_epoch += float(loss.item())
                steps += 1

            loss_val = loss_epoch / max(1, steps)
            if ema_loss is None:
                ema_loss = loss_val
            else:
                ema_loss = 0.97 * ema_loss + 0.03 * loss_val

            if epoch % int(args.print_interval) == 0:
                elapsed = time.perf_counter() - t_epoch
                ms = elapsed * 1000.0
                pts_per_s = (float(steps) * float(int(args.batch_size))) / max(elapsed, 1e-9)
                res = ""
                if int(args.resource_interval) > 0 and (epoch % int(args.resource_interval) == 0):
                    res = " | " + resource_line(device)
                print(
                    f"Epoch {epoch:5d}: loss={loss_val:.6f} ema={float(ema_loss):.6f} time={ms:.1f}ms steps={steps} speed={pts_per_s/1e6:.2f} Mpts/s{res}"
                )

    elif use_dataset:
        ds_dir = Path(str(args.dataset))
        dataset = MemmapSDFDataset(ds_dir)
        log(1, f"[Dataset] loaded {len(dataset):,} points from {ds_dir}")

        default_steps = max(1, len(dataset) // int(args.batch_size))
        steps_per_epoch = int(args.steps_per_epoch) if args.steps_per_epoch is not None else default_steps
        log(1, f"[Dataset] steps_per_epoch={steps_per_epoch} (default would be {default_steps})")

        # High-throughput iterable that returns already-batched arrays.
        iterable = RandomBatchSDFIterable(
            ds_dir,
            batch_size=int(args.batch_size),
            steps=int(steps_per_epoch),
            seed=int(args.seed),
        )
        loader = torch.utils.data.DataLoader(  # type: ignore[attr-defined]
            iterable,
            batch_size=None,
            num_workers=int(args.workers),
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers) and int(args.workers) > 0,
            prefetch_factor=int(args.prefetch_factor) if int(args.workers) > 0 else None,
        )

        for epoch in range(int(args.epochs)):
            t_epoch = time.perf_counter()
            loss_epoch = 0.0
            steps = 0
            for xyz_np, sdf_np in loader:
                # xyz_np: (B,3) float32, sdf_np: (B,) float32
                xyz = xyz_np.to(device=device, non_blocking=True)  # type: ignore[union-attr]
                sdf = sdf_np.to(device=device, non_blocking=True)  # type: ignore[union-attr]

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and bool(args.amp))):  # type: ignore[attr-defined]
                    pred = model(xyz)
                    tgt = _make_targets_from_xyz_sdf(xyz, sdf, displacement_scale=float(args.disp_scale))
                    loss_disp = mse(pred[:, 3], tgt[:, 3]) * float(args.disp_weight)
                    loss_rgb = mse(pred[:, :3], tgt[:, :3]) * float(args.rgb_weight)
                    loss = loss_disp + loss_rgb

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_epoch += float(loss.item())
                steps += 1

            loss_val = loss_epoch / max(1, steps)
            if ema_loss is None:
                ema_loss = loss_val
            else:
                ema_loss = 0.97 * ema_loss + 0.03 * loss_val

            if epoch % int(args.print_interval) == 0:
                ms = (time.perf_counter() - t_epoch) * 1000.0
                res = ""
                if int(args.resource_interval) > 0 and (epoch % int(args.resource_interval) == 0):
                    res = " | " + resource_line(device)
                print(f"Epoch {epoch:5d}: loss={loss_val:.6f} ema={float(ema_loss):.6f} time={ms:.1f}ms steps={steps}{res}")
    else:
        for epoch in range(int(args.epochs)):
            t_epoch = time.perf_counter()
            show_stats = (VERBOSE >= 2) and (epoch % int(max(1, args.stats_interval)) == 0)
            t_data = time.perf_counter()
            x, y = get_training_batch(
                mesh,
                int(args.batch_size),
                device=device,
                sdf_query=sdfq,
                near_sigma=float(args.near_sigma),
                frac_mesh_near=float(args.frac_mesh_near),
                frac_unit_sphere=float(args.frac_unit_sphere),
                frac_unit_ball=float(args.frac_unit_ball),
                displacement_scale=float(args.disp_scale),
            )
            data_ms = (time.perf_counter() - t_data) * 1000.0

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and bool(args.amp))):  # type: ignore[attr-defined]
                pred = model(x)
                loss_disp = mse(pred[:, 3], y[:, 3]) * float(args.disp_weight)
                loss_rgb = mse(pred[:, :3], y[:, :3]) * float(args.rgb_weight)
                loss = loss_disp + loss_rgb

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_val = float(loss.item())
            if ema_loss is None:
                ema_loss = loss_val
            else:
                ema_loss = 0.97 * ema_loss + 0.03 * loss_val

            if epoch % int(args.print_interval) == 0:
                ms = (time.perf_counter() - t_epoch) * 1000.0
                res = ""
                if int(args.resource_interval) > 0 and (epoch % int(args.resource_interval) == 0):
                    res = " | " + resource_line(device)
                print(
                    f"Epoch {epoch:5d}: loss={loss_val:.6f} (disp={float(loss_disp.item()):.6f}, rgb={float(loss_rgb.item()):.6f}) "
                    f"ema={float(ema_loss):.6f} time={ms:.1f}ms data={data_ms:.1f}ms{res}"
                )
            elif show_stats:
                ms = (time.perf_counter() - t_epoch) * 1000.0
                log(2, f"[Epoch {epoch}] loss={loss_val:.6f} ema={float(ema_loss):.6f} time={ms:.1f}ms data={data_ms:.1f}ms")

    log(1, f"[Training] total_time={(time.perf_counter() - t_train):.2f}s")
    export_weights(model, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
