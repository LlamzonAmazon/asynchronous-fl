#!/usr/bin/env python3
"""Detect and recover stalled FL chains.

Last night a chain deadlocked on a gRPC "ping timeout" and sat at ~0% CPU for
15 hours without exiting, so neither the exit code nor the log revealed it. A
healthy chain keeps its Python workers busy; a deadlocked one does not. CPU is
therefore the reliable stall signal, and log freshness is not (rounds take ~75
minutes, so logs are legitimately silent for long stretches).

Conservative by construction: a chain must sit below CPU_FLOOR for
STALL_POLLS consecutive polls before any action, each seed is restarted at most
MAX_RESTARTS times, and a chain with no Python workers is treated as
between-stages rather than stalled.
"""
import os, re, subprocess, sys, time
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLL_SECONDS = 300
CPU_FLOOR = 2.0        # percent; observed deadlock sat at 0.0-0.2%, healthy
                       # disk-bound data prep runs 15-18%, so 2.0 separates them
STALL_POLLS = 4        # consecutive low-CPU polls (20 min) before acting
MAX_RESTARTS = 2
LOG = os.path.join(REPO, "jobs", "logs", "watchdog.log")

PORTS = {"7": "8081", "13": "8082"}
STAGES = {"7": "async", "13": "both"}


def log(msg):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def snapshot():
    """Return {pid: (ppid, cpu, cmdline)} for every process."""
    out = subprocess.run(["ps", "-ax", "-o", "pid=,ppid=,%cpu=,command="],
                         capture_output=True, text=True).stdout
    procs = {}
    for ln in out.splitlines():
        parts = ln.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, cpu, cmd = parts
        try:
            procs[int(pid)] = (int(ppid), float(cpu), cmd)
        except ValueError:
            continue
    return procs


def chains(procs):
    """Map seed -> (root_pid, [descendant pids]).

    Process tree is: bash run_seed_chain.sh -> {caffeinate ... FL_SEED=N ...,
    python run_fl.py -> server + clients}. Only the caffeinate/env process
    carries FL_SEED in its command line, and it has no Python children, so the
    seed is read from it but the tree is rooted at its PARENT.
    """
    kids = defaultdict(list)
    for pid, (ppid, _, _) in procs.items():
        kids[ppid].append(pid)

    def descendants(root):
        out, stack = [], [root]
        while stack:
            p = stack.pop()
            for c in kids.get(p, []):
                out.append(c)
                stack.append(c)
        return out

    found = {}
    for pid, (ppid, _, cmd) in procs.items():
        if "watchdog" in cmd:
            continue
        m = re.search(r"FL_SEED=(\d+)", cmd)
        if not m or "run_seed_chain.sh" not in cmd:
            continue
        seed = m.group(1)
        root = ppid if ppid in procs else pid
        found[seed] = (root, descendants(root))
    return found


def kill_tree(procs, root, sig="-TERM"):
    pids = []
    kids = defaultdict(list)
    for pid, (ppid, _, _) in procs.items():
        kids[ppid].append(pid)
    stack = [root]
    while stack:
        p = stack.pop()
        pids.append(p)
        stack.extend(kids.get(p, []))
    for p in reversed(pids):
        subprocess.run(["kill", sig, str(p)], capture_output=True)
    return pids


def cpu_of(procs, pids):
    return sum(procs[p][1] for p in pids if p in procs and "Python" in procs[p][2])


def workers(procs, pids):
    return [p for p in pids if p in procs and "Python" in procs[p][2]]


def restart(seed, root):
    log(f"seed {seed}: restarting (stages={STAGES[seed]}, port={PORTS[seed]})")
    procs = snapshot()
    killed = kill_tree(procs, root, "-TERM")
    time.sleep(5)
    kill_tree(snapshot(), root, "-KILL")
    time.sleep(2)
    log(f"seed {seed}: killed {len(killed)} processes under root {root}")
    stamp = time.strftime("%m%d_%H%M")
    logf = os.path.join(REPO, "jobs", "logs", f"seed{seed}_restart_{stamp}.log")
    env = dict(os.environ, FL_SEED=seed, FL_K="2",
               FL_PORT=PORTS[seed], FL_STAGES=STAGES[seed])
    with open(logf, "w") as fh:
        subprocess.Popen(["caffeinate", "-ims",
                          os.path.join(REPO, "jobs", "run_seed_chain.sh")],
                         stdout=fh, stderr=subprocess.STDOUT, env=env,
                         start_new_session=True, cwd=REPO)
    log(f"seed {seed}: relaunched, log={os.path.basename(logf)}")


def main():
    low = defaultdict(int)
    restarts = defaultdict(int)
    log(f"watchdog started (poll={POLL_SECONDS}s, floor={CPU_FLOOR}%, "
        f"stall={STALL_POLLS} polls, max_restarts={MAX_RESTARTS})")
    while True:
        procs = snapshot()
        live = chains(procs)
        if not live:
            log("no chains running; watchdog exiting")
            return
        for seed, (cpid, desc) in sorted(live.items()):
            w = workers(procs, desc)
            cpu = cpu_of(procs, desc)
            if not w:
                low[seed] = 0
                log(f"seed {seed}: no python workers (between stages), cpu={cpu:.1f}%")
                continue
            if cpu < CPU_FLOOR:
                low[seed] += 1
                log(f"seed {seed}: LOW cpu={cpu:.1f}% across {len(w)} workers "
                    f"({low[seed]}/{STALL_POLLS})")
                if low[seed] >= STALL_POLLS:
                    if restarts[seed] >= MAX_RESTARTS:
                        log(f"seed {seed}: STALLED but restart limit reached; leaving it")
                        low[seed] = 0
                        continue
                    restarts[seed] += 1
                    log(f"seed {seed}: STALLED (restart {restarts[seed]}/{MAX_RESTARTS})")
                    restart(seed, cpid)
                    low[seed] = 0
            else:
                if low[seed]:
                    log(f"seed {seed}: recovered, cpu={cpu:.1f}%")
                low[seed] = 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("watchdog interrupted")
        sys.exit(0)
