"""
Regenerate training-curve PNGs for async runs from saved metrics.

Useful when plot styling (e.g., deep-round marker linewidth) has changed
and existing PNGs need to be redrawn without re-running training.

Usage:
    python utils/regenerate_async_plots.py [RUN_ID ...]

Examples:
    python utils/regenerate_async_plots.py
        # Default: async_IID_4R_3C_1L_K1, async_IID_4R_3C_1L_K2

    python utils/regenerate_async_plots.py async_IID_4R_3C_1L_K1 async_IID_4R_3C_1L_K2
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
ASYNC_RESULTS_ROOT = _project_root / "results" / "async-federated"

DEFAULT_RUN_IDS = [
    "async_IID_4R_3C_1L_K1",
    "async_IID_4R_3C_1L_K2",
]

# Styling: bolder green dashed line for deep rounds (matches flower_server.py)
DEEP_LINE_KW = {"color": "darkgreen", "alpha": 0.85, "linestyle": "--", "linewidth": 2.5}


def plot_training_curves(results_dir: Path, plot_save_path: str, round_log: list) -> bool:
    """Plot test loss & accuracy with deep/shallow round markers. Returns True if OK."""
    metrics_path = results_dir / "checkpoints" / "all_metrics.json"
    if not metrics_path.exists():
        print(f"No metrics at {metrics_path}; skipping.")
        return False

    with open(metrics_path, "r") as f:
        all_metrics = json.load(f)
    if not all_metrics:
        print("No round metrics to plot; skipping.")
        return False

    rounds = [m["round"] for m in all_metrics]
    test_losses = [m["test_loss"] for m in all_metrics]
    test_accs = [m["test_accuracy"] for m in all_metrics]

    full_rounds = set()
    for entry in round_log:
        if entry.get("round_type") == "full":
            full_rounds.add(entry["round"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(rounds, test_losses, "b-o", label="Test Loss", markersize=4)
    for r in rounds:
        if r in full_rounds:
            ax1.axvline(x=r, **DEEP_LINE_KW)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss")
    ax1.set_title("Test Loss per Round")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(rounds, test_accs, "r-o", label="Test Accuracy (%)", markersize=4)
    for r in rounds:
        if r in full_rounds:
            ax2.axvline(x=r, **DEEP_LINE_KW)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Test Accuracy per Round")
    ax2.legend()
    ax2.grid(True)

    custom_legend = Line2D([0], [0], label="Full (deep) round", **DEEP_LINE_KW)
    for ax in (ax1, ax2):
        handles, labels = ax.get_legend_handles_labels()
        handles.append(custom_legend)
        ax.legend(handles=handles)

    plt.tight_layout()
    Path(plot_save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_save_path)
    plt.close()
    print(f"Training curves saved: {plot_save_path}")
    return True


def regenerate_plot(run_id: str) -> bool:
    """Regenerate PNG for one async run. Returns True if successful."""
    results_dir = ASYNC_RESULTS_ROOT / run_id
    if not results_dir.exists():
        print(f"ERROR: Run directory not found: {results_dir}")
        return False

    round_metrics_path = results_dir / "round_metrics.json"
    if not round_metrics_path.exists():
        print(f"ERROR: round_metrics.json not found: {round_metrics_path}")
        return False

    with open(round_metrics_path, "r") as f:
        round_log = json.load(f)

    plot_save_path = str(results_dir / f"{run_id}.png")
    return plot_training_curves(results_dir, plot_save_path, round_log)


def main():
    run_ids = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_RUN_IDS

    print(f"Regenerating async training curves for: {run_ids}")
    print()

    for run_id in run_ids:
        print(f"  {run_id}...")
        ok = regenerate_plot(run_id)
        if not ok:
            sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
