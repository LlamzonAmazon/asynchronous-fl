"""
Measure per-group parameter drift across FL rounds from existing checkpoints.

Checks whether the deep group actually changes more slowly than the shallow
group in this ECGCNN.

Produces the number in Section III of the GLOBECOM 2026 paper reporting that the
deep group drifts at a median of 1.16x the shallow rate and exceeds it in 8 of 12
post-initialization transitions. Regenerate that number with:

    python utils/layer_drift.py

Read-only. No training. Operates on checkpoints already on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent

# From federated/asynchronous/config.py: AsyncFLConfig.SHALLOW_PREFIXES
SHALLOW_PREFIXES = ("conv1.", "bn1.", "conv2.", "bn2.")


def load_state(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    # Checkpoints may be a bare state_dict or a wrapper dict.
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    return obj


def group_of(key: str) -> str:
    return "shallow" if key.startswith(SHALLOW_PREFIXES) else "deep"


def is_float(t: torch.Tensor) -> bool:
    # num_batches_tracked is int64 and carries no gradient signal.
    return t.is_floating_point()


def drift(prev: dict, curr: dict) -> dict[str, tuple[float, float, float]]:
    """Return {group: (||delta||, ||prev||, relative)} over float tensors only."""
    acc: dict[str, list[float]] = {"shallow": [0.0, 0.0], "deep": [0.0, 0.0]}
    for key, p in prev.items():
        if key not in curr or not is_float(p):
            continue
        g = group_of(key)
        d = (curr[key].double() - p.double())
        acc[g][0] += float(d.pow(2).sum())
        acc[g][1] += float(p.double().pow(2).sum())
    out = {}
    for g, (dsq, psq) in acc.items():
        dn, pn = dsq ** 0.5, psq ** 0.5
        out[g] = (dn, pn, dn / pn if pn > 0 else float("nan"))
    return out


def report(run_dir: Path, label: str) -> None:
    ckpt = run_dir / "checkpoints"
    paths = sorted(ckpt.glob("model_round_*.pth"),
                   key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if len(paths) < 2:
        print(f"  [skip] {label}: found {len(paths)} checkpoint(s)")
        return

    print(f"\n{label}")
    print(f"  {'transition':<14} {'shallow rel':>12} {'deep rel':>12} {'deep/shallow':>13}")
    print(f"  {'-' * 14} {'-' * 12} {'-' * 12} {'-' * 13}")

    prev_state = load_state(paths[0])
    for path in paths[1:]:
        curr_state = load_state(path)
        d = drift(prev_state, curr_state)
        a = int(path.stem.rsplit("_", 1)[1])
        s_rel, d_rel = d["shallow"][2], d["deep"][2]
        ratio = d_rel / s_rel if s_rel > 0 else float("nan")
        print(f"  {a - 1} -> {a:<9} {s_rel:>12.6f} {d_rel:>12.6f} {ratio:>13.3f}")
        prev_state = curr_state

    # Group sizes, for the record.
    sizes = {"shallow": 0, "deep": 0}
    for key, t in prev_state.items():
        if is_float(t):
            sizes[group_of(key)] += t.numel()
    total = sizes["shallow"] + sizes["deep"]
    print(f"  params: shallow {sizes['shallow']:,} ({100 * sizes['shallow'] / total:.2f}%), "
          f"deep {sizes['deep']:,} ({100 * sizes['deep'] / total:.2f}%)")


def main() -> int:
    runs = [
        (REPO / "results/sync-federated/sync_IID_4R_3C_1L", "Synchronous, IID, seed 42"),
        (REPO / "results/sync-federated/sync_nonIID_4R_3C_1L", "Synchronous, non-IID, seed 42"),
        (REPO / "results/sync-federated_s7/sync_IID_4R_3C_1L_s7", "Synchronous, IID, seed 7"),
        (REPO / "results/sync-federated_s13/sync_IID_4R_3C_1L_s13", "Synchronous, IID, seed 13"),
    ]
    print("Per-group relative drift  ||theta_g^{t+1} - theta_g^t|| / ||theta_g^t||")
    print("Shallow prefixes:", SHALLOW_PREFIXES)
    print("Integer buffers (num_batches_tracked) excluded.")
    for run_dir, label in runs:
        if run_dir.exists():
            report(run_dir, label)
        else:
            print(f"\n[missing] {label}: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
