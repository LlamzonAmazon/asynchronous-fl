#!/usr/bin/env python3
"""
Emit the camera-ready results table and every figure quoted in prose,
computed directly from validated run artifacts.

    python jobs/make_results_table.py

Exists because the submitted Table I contains three transcription errors in
the Upload column (1699.99, 1700.02 and 854.98 MiB against true values of
1699.75, 1699.75 and 855.01) and quotes accuracies that are single runs
without saying so. Numbers that appear in the paper should be generated, not
retyped.

Two conventions are enforced here that the submitted table does not:

  1. ONE BYTE CONVENTION FOR BOTH ARMS. The synchronous counter sums
     `p.numel() for p in model.parameters()`, which omits BatchNorm running
     statistics; the asynchronous counter measures the full state_dict. That
     is a 3,872 B per-model difference, and over 12 client uploads exactly the
     46,464 B by which the two arms appear to differ. The bytes on the wire
     are identical. This script reports the state_dict convention throughout,
     which makes synchronous and K=1 agree exactly, as they must.

  2. REPLICATED ROWS ARE MARKED. Only sync IID and async K=2 IID have multiple
     seeds. Every other row is a single run and is labelled as such rather
     than being silently averaged or silently presented as if replicated.

Runs failing jobs/validate_run.py are excluded and reported, so a silently
degraded federation cannot reach the table.
"""

import json
import math
import statistics as st
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MIB = 1024 * 1024
CLIENTS, ROUNDS = 3, 4

# (label, results dir, seeds present). Seed None means the unsuffixed run.
ARMS = {
    "sync_IID":       [(42, "results/sync-federated/sync_IID_4R_3C_1L"),
                       (7,  "results/sync-federated_s7/sync_IID_4R_3C_1L_s7"),
                       (13, "results/sync-federated_s13/sync_IID_4R_3C_1L_s13")],
    "sync_nonIID":    [(42, "results/sync-federated/sync_nonIID_4R_3C_1L")],
    "async_K1_IID":   [(42, "results/async-federated/async_IID_4R_3C_1L_K1")],
    "async_K2_IID":   [(42, "results/async-federated/async_IID_4R_3C_1L_K2"),
                       (7,  "results/async-federated/async_IID_4R_3C_1L_K2_s7"),
                       (13, "results/async-federated/async_IID_4R_3C_1L_K2_s13")],
    "async_K4_IID":   [(42, "results/async-federated/async_IID_4R_3C_1L_K4")],
    "async_K2_nonIID":[(42, "results/async-federated/async_nonIID_4R_3C_1L_K2")],
    "async_K4_nonIID":[(42, "results/async-federated/async_nonIID_4R_3C_1L_K4")],
}


def is_valid(run_dir: Path) -> bool:
    r = subprocess.run(
        [sys.executable, str(REPO / "jobs" / "validate_run.py"), str(run_dir)],
        capture_output=True, cwd=str(REPO))
    return r.returncode == 0


def collect():
    acc, excluded = {}, []
    for arm, entries in ARMS.items():
        vals = {}
        for seed, rel in entries:
            d = REPO / rel
            if not (d / "checkpoints" / f"metrics_round_{ROUNDS}.json").exists():
                continue
            if not is_valid(d):
                excluded.append(f"{arm} seed {seed} ({rel})")
                continue
            vals[seed] = json.load(
                open(d / "checkpoints" / f"metrics_round_{ROUNDS}.json"))["test_accuracy"]
        acc[arm] = vals
    return acc, excluded


def fmt(vals):
    """mean +- sd for replicated arms, bare value for single runs."""
    v = list(vals.values())
    if not v:
        return "pending", 0
    if len(v) == 1:
        return f"{v[0]:.2f}", 1
    return f"${st.mean(v):.2f} \\pm {st.stdev(v):.2f}$", len(v)


def main():
    acc, excluded = collect()

    net = json.load(open(REPO / "results/async-federated/async_IID_4R_3C_1L_K1/network_metrics.json"))
    full, shallow = net["full_model_bytes_state_dict"], net["shallow_bytes_state_dict"]

    def upload(K):
        deep = len([r for r in range(1, ROUNDS + 1) if r % K == 0])
        return CLIENTS * (deep * full + (ROUNDS - deep) * shallow)

    base = CLIENTS * ROUNDS * full
    up = {K: upload(K) for K in (1, 2, 4)}
    red = {K: 100 * (1 - up[K] / base) for K in (1, 2, 4)}

    if excluded:
        print("%% EXCLUDED (failed jobs/validate_run.py):")
        for e in excluded:
            print(f"%%   {e}")
        print()

    rows = [
        ("Centralized", "84.08", None, None),
        None,
        ("Sync IID",          fmt(acc["sync_IID"]),        base,  0.0),
        ("Sync non-IID",      fmt(acc["sync_nonIID"]),     base,  0.0),
        None,
        ("Async $K=1$ IID",     fmt(acc["async_K1_IID"]),    up[1], red[1]),
        ("Async $K=2$ IID",     fmt(acc["async_K2_IID"]),    up[2], red[2]),
        ("Async $K=4$ IID",     fmt(acc["async_K4_IID"]),    up[4], red[4]),
        ("Async $K=2$ non-IID", fmt(acc["async_K2_nonIID"]), up[2], red[2]),
        ("Async $K=4$ non-IID", fmt(acc["async_K4_nonIID"]), up[4], red[4]),
    ]

    print(r"\begin{table}[!t]")
    print(r"\centering")
    print(r"\caption{Summary of experimental results. $^{\dagger}$~mean $\pm$ standard deviation")
    print(r"over three random seeds; all other rows are single runs. Upload is measured on the")
    print(r"full \texttt{state\_dict} for both arms, and the centralized row is a non-federated")
    print(r"reference rather than an upper bound.}")
    print(r"\label{tab:summary}")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{@{}lccc@{}}")
    print(r"\toprule")
    print(r"\textbf{Experiment} & \textbf{Acc. (\%)} & \textbf{Upload (MiB)} & \textbf{Red.} \\")
    print(r"\midrule")
    for row in rows:
        if row is None:
            print(r"\midrule")
            continue
        label, a, b, r = row
        if isinstance(a, tuple):
            a, n = a
            if n >= 2:
                label += r"$^{\dagger}$"
        bs = "--" if b is None else f"{b / MIB:.2f}"
        rs = "--" if r is None else f"{r:.2f}\\%"
        print(f"{label} & {a} & {bs} & {rs} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Figures quoted in prose.
    print()
    print("%" * 70)
    print("%% FIGURES QUOTED IN PROSE")
    print("%" * 70)
    sy, asy = acc["sync_IID"], acc["async_K2_IID"]
    shared = sorted(set(sy) & set(asy))
    print(f"%% sync IID     n={len(sy)}: " + ", ".join(f"s{k}={v:.2f}" for k, v in sorted(sy.items())))
    print(f"%% async K2 IID n={len(asy)}: " + ", ".join(f"s{k}={v:.2f}" for k, v in sorted(asy.items())))
    if len(sy) >= 2:
        print(f"%% sync  mean +- sd : {st.mean(sy.values()):.2f} +- {st.stdev(sy.values()):.2f}")
    if len(asy) >= 2:
        print(f"%% async mean +- sd : {st.mean(asy.values()):.2f} +- {st.stdev(asy.values()):.2f}")
    if len(shared) >= 2:
        diffs = [asy[s] - sy[s] for s in shared]
        md, sd = st.mean(diffs), st.stdev(diffs)
        se = sd / math.sqrt(len(diffs))
        print(f"%% paired seeds     : {shared}")
        print(f"%% paired diffs     : " + ", ".join(f"{d:+.2f}" for d in diffs))
        print(f"%% mean paired diff : {md:+.2f} pp (sd {sd:.2f}, se {se:.2f}, t={md/se:.2f}, df={len(diffs)-1})")
        try:
            from scipy import stats
            print(f"%% two-tailed p     : {2*(1-stats.t.cdf(abs(md/se), len(diffs)-1)):.3f}")
        except ImportError:
            print("%% two-tailed p     : (scipy unavailable)")
        print(f"%% sign test        : {sum(1 for d in diffs if d>0)}/{len(diffs)} favour async")
    tot_sync = 2 * base
    tot_k2 = up[2] + base
    print(f"%% total comm reduction at K=2 : {100*(1-tot_k2/tot_sync):.2f}%")
    print(f"%% shallow fraction of model   : {100*shallow/full:.2f}%")
    print(f"%% BN-buffer accounting artefact: {(2*480*4 + 4*8)*CLIENTS*ROUNDS:,} B")


if __name__ == "__main__":
    main()
