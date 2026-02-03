#!/usr/bin/env python3
"""
Generate training report with convergence plots and metrics.
Parses training.log and generates visualization.
"""

import json
import re
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Run 'pip install matplotlib' for plots.")


def parse_training_log(log_file):
    """Extract loss values from training log."""
    losses = []
    pattern = r'loss=([0-9.]+)'

    with open(log_file) as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                try:
                    loss = float(match.group(1))
                    losses.append(loss)
                except ValueError:
                    pass

    return losses


def generate_report(dataset_dir="universe_train_test"):
    """Generate comprehensive training report."""
    trained_dir = Path(f"{dataset_dir}_trained")

    print("=== HELIOS Training Report ===\n")

    # Check if training completed
    if not trained_dir.exists():
        print(f"ERROR: Trained dataset not found at {trained_dir}")
        print("Please run 'bash scripts/train_validation.sh' first.")
        return

    # Load manifest
    manifest_path = trained_dir / "index.json"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Dataset stats
    print(f"Dataset: {dataset_dir}")
    print(f"Cells trained: {len(manifest['cells'])}")
    total_splats = sum(cell['splat_count'] for cell in manifest['cells'])
    print(f"Total splats: {total_splats}\n")

    # Parse training logs
    if Path("training.log").exists():
        losses = parse_training_log("training.log")

        if losses:
            print(f"Training iterations logged: {len(losses)}")
            print(f"Initial loss: {losses[0]:.6f}")
            print(f"Final loss: {losses[-1]:.6f}")
            reduction = (1 - losses[-1] / losses[0]) * 100
            print(f"Loss reduction: {reduction:.1f}%\n")

            # Save metrics to JSON for website
            metrics = {
                "losses": losses,
                "iterations": len(losses),
                "cells_trained": len(manifest['cells']),
                "total_splats": total_splats,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "reduction_percent": reduction
            }

            output_dir = Path("client/public/papers/data")
            output_dir.mkdir(parents=True, exist_ok=True)

            metrics_path = output_dir / "training_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"✅ Training metrics saved: {metrics_path}\n")

            # Generate convergence plot
            if HAS_MATPLOTLIB:
                plt.figure(figsize=(10, 6))
                plt.plot(losses, linewidth=2, color='#667eea')
                plt.xlabel('Iteration', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.title('Training Convergence', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.tight_layout()

                # Save for website
                plot_path = Path("client/public/papers/figures/convergence.png")
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"✅ Convergence plot saved: {plot_path}")

                # Save for repo root
                plt.savefig('training_convergence.png', dpi=150, bbox_inches='tight')
                print(f"✅ Convergence plot saved: training_convergence.png")
            else:
                print("⚠️  Matplotlib not available - skipping plot generation")
                print("   Install with: pip install matplotlib")
        else:
            print("⚠️  No loss values found in training.log")
    else:
        print("⚠️  No training.log found - skipping loss analysis")
        print("   Make sure training was run with output redirection:")
        print("   cargo run ... 2>&1 | tee training.log")

    print("\n" + "=" * 50)
    print("Report generation complete!")
    print("=" * 50)


if __name__ == "__main__":
    generate_report()
