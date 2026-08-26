#!/usr/bin/env python3
"""
Validate that an FL run actually did what its metrics file implies.

    python jobs/validate_run.py <run_dir> [--clients 3] [--rounds 4]

Exit status 0 if the run is sound, 1 if it is not.

Motivation: on 2026-08-21 the seed 7 async K=2 run finished, wrote
metrics_round_4.json, and reported 82.76% accuracy. Two of its three clients
had in fact died of gRPC "Socket closed" errors after round 2, so the last two
rounds were federated learning over a single client. The chain script judged
the stage complete because the metrics file existed, and nothing else looked.
An accuracy number from a silently degraded federation is worse than a missing
one, because it looks usable.

Checks performed:
  1. The final round's metrics file exists.
  2. The log records clients x rounds local training invocations. A client
     that dies mid-run simply stops producing them.
  3. No gRPC socket teardown or MemoryError appears in the log.
  4. For async runs, every entry in round_metrics.json carries the upload
     volume of a full client cohort, not a partial one. This is the check
     that caught the seed 7 failure, because byte totals are recorded from
     observed messages rather than computed analytically.
"""

import argparse
import json
import re
import sys
from pathlib import Path


def _fail(msgs, text):
    msgs.append(text)


def validate(run_dir: Path, clients: int, rounds: int):
    problems = []

    metrics = run_dir / "checkpoints" / f"metrics_round_{rounds}.json"
    if not metrics.exists():
        _fail(problems, f"missing final metrics file: {metrics}")

    log = run_dir / "last_run.log"
    if not log.exists():
        _fail(problems, f"missing log: {log}")
        return problems

    raw = log.read_bytes().decode("utf-8", "replace")

    expected_calls = clients * rounds
    calls = len(re.findall(r"\[Client \d+\] Training on", raw))
    if calls != expected_calls:
        _fail(problems,
              f"local training invocations: {calls}, expected {expected_calls} "
              f"({clients} clients x {rounds} rounds). Clients dropped out.")

    for pattern, label in (("Socket closed", "gRPC socket teardown"),
                           ("MemoryError", "out of memory")):
        hits = len(re.findall(pattern, raw))
        if hits:
            _fail(problems, f"{label} in log ({hits} occurrence(s))")

    # Async runs record observed upload volume per round, so a partial cohort
    # shows up directly as a short byte count. The per-client unit comes from
    # the model sizes in network_metrics.json, never from the observations
    # themselves: deriving it from the data would make a uniformly degraded
    # run look self-consistent.
    rm = run_dir / "round_metrics.json"
    nm = run_dir / "network_metrics.json"
    if rm.exists() and nm.exists():
        try:
            entries = json.loads(rm.read_text())
            net = json.loads(nm.read_text())
        except json.JSONDecodeError as exc:
            _fail(problems, f"metrics JSON unreadable: {exc}")
            return problems
        if isinstance(entries, dict):
            entries = entries.get("round_log", [])

        unit = {
            "shallow_only": net.get("shallow_bytes_state_dict"),
            "full": net.get("full_model_bytes_state_dict"),
        }
        for e in entries:
            kind = e.get("round_type")
            u = unit.get(kind)
            if not u:
                continue
            n = e["bytes_up_total"] / u
            if abs(n - clients) > 1e-6:
                _fail(problems,
                      f"round {e['round']} ({kind}): upload volume equals "
                      f"{n:g} client(s), expected {clients}")

    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"INVALID {run_dir}: not a directory")
        sys.exit(1)

    problems = validate(run_dir, args.clients, args.rounds)
    if problems:
        print(f"INVALID {run_dir.name}")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    print(f"VALID {run_dir.name}: {args.clients} clients x {args.rounds} rounds, no dropouts")
    sys.exit(0)


if __name__ == "__main__":
    main()
