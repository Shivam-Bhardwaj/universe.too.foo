#!/usr/bin/env python3
"""Train MLP to learn a procedural Cratered Sphere asteroid shape.

This script trains a 35→64→64→4 MLP to approximate a procedural function that
generates a cratered sphere with surface texture. The trained weights are
exported to `assets/neural_decoder.bin` in the exact format expected by the
Rust/WGPU shader.

Model Architecture (must match shader.wgsl exactly):
- Input: 35 dims (only first 3 used: xyz position)
- Layer 1: 35 → 64, ReLU
- Layer 2: 64 → 64, ReLU
- Output: 64 → 4 (RGB + Displacement)
  - RGB: sigmoid activation (0-1 range)
  - Displacement: raw output (unbounded)
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as e:
    print("Error: Required packages not found.", file=sys.stderr)
    print(f"Missing: {e}", file=sys.stderr)
    print("\nIf you have a virtual environment (.venv), try:", file=sys.stderr)
    print("  source .venv/bin/activate  # or: .venv/bin/python train_asteroid.py", file=sys.stderr)
    print("\nOr install dependencies:", file=sys.stderr)
    print("  python3 -m venv venv", file=sys.stderr)
    print("  source venv/bin/activate", file=sys.stderr)
    print("  pip install torch numpy", file=sys.stderr)
    sys.exit(1)


def get_ground_truth(xyz: torch.Tensor) -> torch.Tensor:
    """Generate ground truth (RGB, displacement) for cratered sphere.
    
    Parameters
    ----------
    xyz : torch.Tensor, shape (N, 3)
        Input positions (should be on unit sphere surface for training).
    
    Returns
    -------
    torch.Tensor, shape (N, 4)
        Ground truth [R, G, B, displacement] for each point.
    """
    # Base sphere SDF: distance from origin - 1.0
    r = torch.norm(xyz, dim=1, keepdim=True)
    sdf_base = r - 1.0
    
    # Sinusoidal noise for surface texture
    # Multi-frequency noise for realistic rock texture
    noise_high = torch.sin(10.0 * xyz[:, 0:1]) * torch.sin(10.0 * xyz[:, 1:2]) * torch.sin(10.0 * xyz[:, 2:3])
    noise_mid = torch.sin(5.0 * xyz[:, 0:1]) * torch.sin(5.0 * xyz[:, 1:2]) * torch.sin(5.0 * xyz[:, 2:3])
    noise_low = torch.sin(2.0 * xyz[:, 0:1]) * torch.sin(2.0 * xyz[:, 1:2]) * torch.sin(2.0 * xyz[:, 2:3])
    
    noise = 0.05 * noise_high + 0.03 * noise_mid + 0.02 * noise_low
    sdf_noise = sdf_base + noise
    
    # Define craters as spherical indentations
    # Each crater is: sdf_crater = ||xyz - center|| - radius
    # We subtract craters using smooth min (soft union): sdf = min(sdf, crater_sdf)
    craters = [
        {"center": torch.tensor([0.7, 0.3, 0.5], device=xyz.device), "radius": 0.25, "depth": 0.15},
        {"center": torch.tensor([-0.5, 0.6, -0.3], device=xyz.device), "radius": 0.2, "depth": 0.12},
        {"center": torch.tensor([0.2, -0.8, 0.4], device=xyz.device), "radius": 0.3, "depth": 0.18},
        {"center": torch.tensor([-0.3, -0.4, -0.7], device=xyz.device), "radius": 0.22, "depth": 0.14},
    ]
    
    sdf = sdf_noise
    crater_mask = torch.zeros_like(sdf)
    
    for crater in craters:
        center = crater["center"]
        radius = crater["radius"]
        depth = crater["depth"]
        
        # Distance to crater center
        dist_to_center = torch.norm(xyz - center.unsqueeze(0), dim=1, keepdim=True)
        # Crater SDF: inside crater sphere, sdf becomes negative
        crater_sdf = dist_to_center - radius
        
        # Smooth minimum (soft union) to blend craters
        # Using smooth_min(sdf, crater_sdf, k=10.0) ≈ min(sdf, crater_sdf) but differentiable
        k = 10.0
        h = torch.clamp((crater_sdf - sdf) / k + 0.5, 0.0, 1.0)
        sdf = sdf * h + crater_sdf * (1.0 - h) - k * h * (1.0 - h)
        
        # Track crater regions for color darkening
        crater_mask = torch.maximum(crater_mask, torch.sigmoid(-crater_sdf * 20.0))
    
    # Generate RGB color based on height/noise and crater regions
    # Base color: gray-brown asteroid color
    base_r = 0.6 + 0.2 * torch.sigmoid(noise * 5.0)
    base_g = 0.5 + 0.15 * torch.sigmoid(noise * 5.0)
    base_b = 0.4 + 0.1 * torch.sigmoid(noise * 5.0)
    
    # Darken in craters
    crater_darken = 0.3 * crater_mask
    r = torch.clamp(base_r - crater_darken, 0.0, 1.0)
    g = torch.clamp(base_g - crater_darken, 0.0, 1.0)
    b = torch.clamp(base_b - crater_darken, 0.0, 1.0)
    
    # Displacement: negative SDF scaled appropriately
    # When evaluated on unit sphere (r≈1), negative SDF means "push inward"
    # Scale factor chosen so displacement is in reasonable range [-1, 1]
    displacement_scale = 0.5
    displacement = -sdf * displacement_scale
    
    return torch.cat([r, g, b, displacement], dim=1)


class AsteroidDecoder(nn.Module):
    """MLP matching the WGSL shader architecture exactly."""
    
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(35, 64)
        self.layer2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 4)
        
        # Initialize with small weights for stable training
        nn.init.xavier_uniform_(self.layer1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.layer2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.output.weight, gain=0.5)
        nn.init.constant_(self.layer1.bias, 0.0)
        nn.init.constant_(self.layer2.bias, 0.0)
        nn.init.constant_(self.output.bias, 0.0)
    
    def forward(self, x):
        """Forward pass matching shader logic.
        
        Input x should be shape (N, 35) but only first 3 dims are used.
        Output is (N, 4): [R, G, B, displacement]
        """
        # Pad input to 35 dims (first 3 are xyz, rest are zeros)
        if x.shape[1] == 3:
            padding = torch.zeros(x.shape[0], 32, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        
        # Layer 1: 35 → 64, ReLU
        h1 = torch.relu(self.layer1(x))
        
        # Layer 2: 64 → 64, ReLU
        h2 = torch.relu(self.layer2(h1))
        
        # Output: 64 → 4
        out = self.output(h2)
        
        # Apply sigmoid to RGB (first 3), keep displacement raw (4th)
        rgb = torch.sigmoid(out[:, :3])
        displacement = out[:, 3:4]
        
        return torch.cat([rgb, displacement], dim=1)


def sample_unit_sphere_surface(n: int, device: torch.device) -> torch.Tensor:
    """Sample n random points uniformly on the surface of a unit sphere."""
    # Generate random points in 3D space
    xyz = torch.randn(n, 3, device=device)
    # Normalize to unit sphere surface
    xyz = xyz / torch.norm(xyz, dim=1, keepdim=True)
    return xyz


def train_model(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 4096,
    epochs: int = 1000,
    lr: float = 1e-3,
    print_interval: int = 50,
) -> None:
    """Train the model to learn the cratered sphere function."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"\n[Training] Starting training for {epochs} epochs")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Device: {device}\n")
    
    best_loss = float('inf')
    patience = 0
    max_patience = 100
    
    for epoch in range(epochs):
        # Sample random points on unit sphere surface
        xyz = sample_unit_sphere_surface(batch_size, device)
        
        # Get ground truth
        with torch.no_grad():
            target = get_ground_truth(xyz)
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(xyz)
        loss = criterion(pred, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track best loss
        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            patience = 0
        else:
            patience += 1
        
        # Print progress
        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {loss_val:.6f} | Best: {best_loss:.6f}")
        
        # Early stopping if loss is very low
        if loss_val < 1e-5:
            print(f"\n[Training] Converged at epoch {epoch+1} with loss {loss_val:.6e}")
            break
        
        # Early stopping if no improvement
        if patience >= max_patience:
            print(f"\n[Training] Early stopping at epoch {epoch+1} (no improvement for {max_patience} epochs)")
            break
    
    print(f"\n[Training] Finished. Final loss: {best_loss:.6f}\n")


def export_weights(model: nn.Module, output_path: Path) -> None:
    """Export model weights to binary file matching data_compiler.py format.
    
    Format: layer1.weight, layer1.bias, layer2.weight, layer2.bias,
            output.weight, output.bias
    All flattened row-major, little-endian float32.
    
    This matches the exact export logic from data_compiler.py:
    iterating named_parameters() in registration order.
    """
    model.eval()
    
    # Collect parameters in registration order (matches data_compiler.py exactly)
    flat_buffer = []
    
    print("[Export] Packing weights:")
    for name, param in model.named_parameters():
        # Detach, move to CPU, convert to numpy, flatten (matches data_compiler.py line 593)
        data = param.detach().cpu().contiguous().numpy().astype(np.float32, copy=False).reshape(-1)
        flat_buffer.append(data)
        print(f"  -> {name}: {data.size} elements (shape {list(param.shape)})")
    
    # Concatenate all parameters
    flat = np.concatenate(flat_buffer).astype("<f4", copy=False)  # Little-endian float32
    
    # Verify expected total count
    expected_total = 64 * 35 + 64 + 64 * 64 + 64 + 4 * 64 + 4  # 6724
    if flat.size != expected_total:
        raise ValueError(
            f"Weight count mismatch: got {flat.size}, expected {expected_total}"
        )
    
    print(f"\n[Export] Total: {flat.size} floats ({flat.size * 4} bytes)")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write binary file
    with open(output_path, "wb") as f:
        f.write(flat.tobytes(order="C"))
    
    # Verify 4-byte alignment
    size = output_path.stat().st_size
    if size % 4 != 0:
        raise RuntimeError(f"Output is not 4-byte aligned: {output_path} size={size}")
    
    print(f"[Export] Saved to: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP to learn cratered sphere asteroid shape"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets/neural_decoder.bin",
        help="Output path for trained weights (default: assets/neural_decoder.bin)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for training (default: 4096)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default: 1000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto)",
    )
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("=" * 70)
    print("Asteroid MLP Trainer")
    print("=" * 70)
    print(f"Output: {args.output}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed: {args.seed}")
    print("=" * 70)
    
    # Create model
    model = AsteroidDecoder().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] Total parameters: {total_params:,}")
    print(f"[Model] Trainable parameters: {trainable_params:,}")
    
    # Train
    train_model(
        model,
        device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )
    
    # Export weights
    output_path = Path(args.output)
    export_weights(model, output_path)
    
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()



