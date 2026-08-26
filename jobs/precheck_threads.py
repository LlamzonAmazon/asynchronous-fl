#!/usr/bin/env python3
"""
Thread-count pre-check for the FL runs.

Answers two questions before committing an overnight batch:

  1. CORRECTNESS. Does capping PyTorch's intra-op thread count change the
     numerics? Thread count controls how parallel reductions are partitioned
     and float addition is not associative, so in principle it can. Counts
     1, 4, 8 and 10 were already verified bitwise identical on 2026-08-19;
     this re-checks those and adds the untested low counts we actually want
     to use. Any mismatch is a hard stop: it would break comparability with
     the seed 42 results from March.

  2. COST. The runs launch four processes (1 server + 3 clients) and each one
     independently defaults to torch.get_num_threads() == 10 on this machine.
     That is 40 threads on 10 cores. This measures wall-clock for three
     concurrent trainers at several thread caps, so the slowdown from capping
     is a measured number rather than a guess, and it measures scheduling
     latency on a separate probe process as a proxy for how usable the
     machine feels while a run is going.

Nothing here touches results/. It trains on synthetic tensors of the exact
production shape, so no partition pickle is loaded and no artifact is written.

Usage:
    python jobs/precheck_threads.py                 # full check, roughly 7 min
    python jobs/precheck_threads.py --batches 3     # shorter
    python jobs/precheck_threads.py --skip-bench    # correctness only, under 1 min
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Production shapes and hyperparameters, from federated/synchronous/config.py
# and models/ecg_cnn.py. Kept in sync deliberately: the benchmark is only
# meaningful if it exercises the same convolutions as a real round.
BATCH_SIZE = 32
TIME_STEPS = 5000
NUM_LEADS = 12
DROPOUT_RATE = 0.4
LEARNING_RATE = 0.001
SEED = 42

# Thread caps to test. 10 is the current (unset) default.
CORRECTNESS_THREADS = [10, 8, 4, 3, 2, 1]
BENCH_THREADS = [10, 4, 3, 2]

# A real round is 205 batches per client.
BATCHES_PER_ROUND = 205
ROUNDS_PER_RUN = 4
RUNS_REMAINING = 3  # seed 7 async, seed 13 sync, seed 13 async


# ----------------------------------------------------------------------------
# Worker roles. Each runs in its own process so that the thread cap is applied
# before any parallel tensor work, which is the only point at which PyTorch's
# native backend will honour it.
# ----------------------------------------------------------------------------

def _build():
    """Construct model, optimizer, loss and a fixed input batch, all seeded."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from models.ecg_cnn import ECGCNN
    from utils.seed import set_seed

    # set_seed pins threads from FL_NUM_THREADS before touching any tensor.
    set_seed(SEED)

    model = ECGCNN(num_leads=NUM_LEADS, dropout_rate=DROPOUT_RATE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # Fixed synthetic batch of the production shape (batch, time, leads).
    gen = torch.Generator().manual_seed(SEED)
    data = torch.randn(BATCH_SIZE, TIME_STEPS, NUM_LEADS, generator=gen)
    target = torch.randint(0, 2, (BATCH_SIZE, 1), generator=gen).float()

    return torch, model, optimizer, criterion, data, target


def _train_step(torch, model, optimizer, criterion, data, target):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()


def role_hash():
    """One deterministic training step. Emit a hash of the resulting state."""
    torch, model, optimizer, criterion, data, target = _build()

    # Re-seed immediately before the step so dropout masks are fixed too.
    torch.manual_seed(SEED)
    loss = _train_step(torch, model, optimizer, criterion, data, target)

    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())

    print(json.dumps({
        "threads": torch.get_num_threads(),
        "loss_hex": loss.hex(),   # exact, no decimal rounding
        "loss": loss,
        "state_sha256": h.hexdigest(),
    }), flush=True)


def role_bench(batches):
    """Time `batches` training steps after a warmup step."""
    torch, model, optimizer, criterion, data, target = _build()

    _train_step(torch, model, optimizer, criterion, data, target)  # warmup

    t0 = time.perf_counter()
    for _ in range(batches):
        _train_step(torch, model, optimizer, criterion, data, target)
    elapsed = time.perf_counter() - t0

    print(json.dumps({
        "threads": torch.get_num_threads(),
        "batches": batches,
        "elapsed_s": elapsed,
        "per_batch_s": elapsed / batches,
    }), flush=True)


def role_probe(duration):
    """Scheduling-latency probe: a proxy for how responsive the machine feels.

    Each iteration sleeps 5 ms then does a small fixed amount of work. On an
    idle machine an iteration lands near its nominal cost. Under CPU
    starvation the wakeup is delayed and the tail blows out, which is exactly
    what makes a desktop feel unusable.
    """
    samples = []
    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        time.sleep(0.005)
        acc = 0.0
        for i in range(20000):
            acc += i * 0.5
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    print(json.dumps({
        "n": len(samples),
        "p50_ms": samples[len(samples) // 2],
        "p95_ms": samples[int(len(samples) * 0.95)],
        "max_ms": samples[-1],
    }), flush=True)


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def _spawn(role, threads, extra=None):
    env = dict(os.environ)
    env["FL_NUM_THREADS"] = str(threads)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--role", role]
    if extra:
        cmd += extra
    return subprocess.Popen(
        cmd, cwd=str(REPO), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _collect(proc):
    out, err = proc.communicate()
    line = next((l for l in reversed(out.strip().splitlines()) if l.startswith("{")), None)
    if line is None:
        raise RuntimeError(f"worker produced no result.\nstdout:\n{out}\nstderr:\n{err}")
    return json.loads(line)


def check_correctness():
    print("=" * 72)
    print("1. CORRECTNESS: is the training step invariant to thread count?")
    print("=" * 72)
    print(f"{'threads':>8}  {'loss':<24} {'state sha256':<20} verdict")
    print("-" * 72)

    reference = None
    all_match = True

    for t in CORRECTNESS_THREADS:
        result = _collect(_spawn("hash", t))
        if reference is None:
            reference = result
            verdict = "reference"
        elif (result["state_sha256"] == reference["state_sha256"]
              and result["loss_hex"] == reference["loss_hex"]):
            verdict = "identical"
        else:
            verdict = "*** DIFFERS ***"
            all_match = False
        print(f"{result['threads']:>8}  {result['loss']!r:<24} "
              f"{result['state_sha256'][:16]}...  {verdict}")

    print()
    if all_match:
        print("PASS. Every thread count produced a bitwise identical model state")
        print("      and an exactly equal loss. Capping threads is numerically free,")
        print("      so new runs stay comparable to the seed 42 results from March.")
    else:
        print("FAIL. Thread count changes the numerics on this machine.")
        print("      Do NOT cap threads. Results would not be comparable across runs.")
    return all_match


def bench_config(threads, batches, probe_seconds):
    """Three concurrent trainers plus a latency probe, matching a real round."""
    probe = _spawn("probe", 1, ["--duration", str(probe_seconds)])
    workers = [_spawn("bench", threads, ["--batches", str(batches)]) for _ in range(3)]

    t0 = time.perf_counter()
    results = [_collect(w) for w in workers]
    wall = time.perf_counter() - t0
    probe_result = _collect(probe)

    per_batch = wall / batches
    return {
        "threads": threads,
        "wall_s": wall,
        "per_batch_s": per_batch,
        "slowest_worker_per_batch_s": max(r["per_batch_s"] for r in results),
        "probe_p50_ms": probe_result["p50_ms"],
        "probe_p95_ms": probe_result["p95_ms"],
    }


def check_cost(batches):
    print()
    print("=" * 72)
    print("2. COST: 3 concurrent trainers (a real round's client load)")
    print("=" * 72)
    print("Probe latency is a separate low-load process. Higher means the")
    print("machine is starved and feels unresponsive.")
    print()
    print(f"{'threads':>8} {'total':>8} {'s/batch':>9} {'round':>8} {'3 runs':>8} "
          f"{'probe p50':>10} {'probe p95':>10}")
    print("-" * 72)

    rows = []
    for t in BENCH_THREADS:
        r = bench_config(t, batches, probe_seconds=max(30, batches * 22))
        rows.append(r)
        round_min = r["per_batch_s"] * BATCHES_PER_ROUND / 60.0
        runs_h = round_min * ROUNDS_PER_RUN * RUNS_REMAINING / 60.0
        print(f"{t:>8} {t * 4:>7}t {r['per_batch_s']:>8.2f}s "
              f"{round_min:>7.0f}m {runs_h:>7.1f}h "
              f"{r['probe_p50_ms']:>9.1f}m {r['probe_p95_ms']:>9.1f}m")

    print()
    print("'threads' is the cap per process; the second column is the total")
    print("thread demand across the 4 processes a run actually launches,")
    print("against 10 physical cores.")

    baseline = next(r for r in rows if r["threads"] == 10)
    print()
    print("Relative to the current uncapped default (10):")
    for r in rows:
        if r["threads"] == 10:
            continue
        speed = r["per_batch_s"] / baseline["per_batch_s"]
        resp = baseline["probe_p95_ms"] / r["probe_p95_ms"] if r["probe_p95_ms"] else float("inf")
        verb = "faster" if speed < 1 else "slower"
        print(f"  FL_NUM_THREADS={r['threads']}: {abs(1 - speed) * 100:>5.1f}% {verb} "
              f"to train, machine {resp:>4.1f}x more responsive at p95")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=["hash", "bench", "probe"],
                        help="internal: worker mode, not for direct use")
    parser.add_argument("--batches", type=int, default=5,
                        help="timed batches per worker (default 5)")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="internal: probe duration in seconds")
    parser.add_argument("--skip-bench", action="store_true",
                        help="run the correctness check only")
    args = parser.parse_args()

    if args.role == "hash":
        return role_hash()
    if args.role == "bench":
        return role_bench(args.batches)
    if args.role == "probe":
        return role_probe(args.duration)

    import torch
    print(f"repo   : {REPO}")
    print(f"python : {sys.executable}")
    print(f"torch  : {torch.__version__}")
    print(f"cores  : {os.cpu_count()} logical")
    print(f"default: torch.get_num_threads() == {torch.get_num_threads()} per process, "
          f"4 processes per run")
    print()

    ok = check_correctness()
    if not ok:
        print("\nStopping. Correctness gate failed; the cost numbers are moot.")
        sys.exit(1)
    if not args.skip_bench:
        check_cost(args.batches)


if __name__ == "__main__":
    main()
