#!/usr/bin/env python3
"""
Pre-flight check on a freshly downloaded Overleaf export, before editing it.

    python jobs/overleaf_preflight.py ~/Downloads/<extracted-folder>

Answers three questions, in this order:

  1. WHAT CHANGED IN OVERLEAF SINCE SUBMISSION. The local
     GLOBECOM/main-conference-v4.tex is the submitted snapshot. Coauthors have
     edited the Overleaf copy since (the JSPS KAKENHI \\thanks{} arrived that
     way and never passed through the local file). Any edit we make has to be
     applied on top of their work, not instead of it, and a download/edit/
     re-upload cycle silently overwrites anything they change in the window.
     This prints their diff so we know what we are building on.

  2. DOES THE +2 LINE OFFSET STILL HOLD. notes/14 quotes line numbers against
     the submitted snapshot. If coauthors added or removed lines in the
     preamble, that offset moves and every quoted number is wrong.

  3. DO THE PASTE ANCHORS STILL EXIST. All 25 steps in notes/16 are anchored on
     search strings verified against the SUBMITTED file. If a coauthor already
     reworded one of those sentences, the anchor is gone and that step needs
     rethinking rather than blind application.

Read-only. Touches nothing.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUBMITTED = REPO / "GLOBECOM" / "main-conference-v4.tex"

# Every anchor notes/16 depends on, in document order.
ANCHORS = [
    ("step 1  eq label",      r"\begin{equation}"),
    ("step 2  cite dupe",     r"\usepackage{cite}"),
    ("step 2  documentclass", r"\documentclass"),
    ("step 2  geometry",      "geometry"),
    ("step 2  hyperref",      "hyperref"),
    ("step 4  abstract",      "while maintaining accuracy within 1 percentage point"),
    ("step 4  total comm",    r"total communication by 24.8\%"),
    ("step 5  energy intro",  "Communication inefficiency also impacts energy consumption"),
    ("step 6  inversion 1c",  "Deep neural networks exhibit layer-wise convergence heterogeneity"),
    ("step 7  contribution",  "we propose a periodic asynchronous layer-wise update framework"),
    ("step 8  related work",  "Despite these advances, existing approaches address"),
    ("step 10 methodology",   r"\section{Proposed Methodology}"),
    ("step 11 eq (3) ref",    "The formulation in (3) implicitly assumes"),
    ("step 12 introduce",     "we introduce a periodic layer-wise schedule"),
    ("step 13 upper bound",   "performance upper bound"),
    ("step 14 congestion",    "congestion"),
    ("step 16 table label",   r"\label{tab:summary}"),
    ("step 17 table intro",   r"Table~\ref{tab:summary} summarizes the results."),
    ("step 18 non-IID claim", "Third, the method remains robust under moderate non-IID"),
    ("step 21 conclusion 2",  "Evaluation on PTB-XL ECG classification demonstrates"),
    ("step 22 conclusion 1",  "we proposed a periodic asynchronous layer-wise update strategy"),
]

# Markers that identify the current Overleaf state rather than the submission.
COAUTHOR_MARKERS = [
    (r"\thanks{This work is supported by JSPS KAKENHI", "KAKENHI funding acknowledgment"),
    (r"\IEEEoverridecommandlockouts", "IEEEoverridecommandlockouts uncommented"),
]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)

    folder = Path(sys.argv[1]).expanduser()
    if not folder.is_dir():
        print(f"not a directory: {folder}")
        sys.exit(1)

    incoming = folder / "main-conference.tex"
    if not incoming.exists():
        cands = sorted(folder.rglob("main-conference.tex"))
        if not cands:
            print(f"no main-conference.tex under {folder}")
            print("files present:")
            for f in sorted(folder.iterdir())[:30]:
                print(f"   {f.name}")
            sys.exit(1)
        incoming = cands[0]

    print("=" * 74)
    print(f"incoming : {incoming}")
    print(f"submitted: {SUBMITTED}")
    print("=" * 74)

    new = incoming.read_text()
    old = SUBMITTED.read_text()
    new_lines, old_lines = new.splitlines(), old.splitlines()

    # ---- 1. what coauthors changed ------------------------------------------
    print("\n1. WHAT CHANGED IN OVERLEAF SINCE SUBMISSION")
    print("-" * 74)
    if new == old:
        print("   identical to the submitted snapshot (no coauthor edits)")
    else:
        r = subprocess.run(["diff", "-u", str(SUBMITTED), str(incoming)],
                           capture_output=True, text=True)
        body = [l for l in r.stdout.splitlines()
                if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
        print(f"   {len(body)} changed line(s), {len(new_lines) - len(old_lines):+d} net")
        for l in r.stdout.splitlines():
            print(f"   {l}")

    print("\n   expected coauthor markers:")
    for marker, label in COAUTHOR_MARKERS:
        present = marker in new
        print(f"     {'OK  ' if present else 'MISS'} {label}")
    if not all(m in new for m, _ in COAUTHOR_MARKERS):
        print("     WARNING: an expected coauthor edit is absent. Confirm this export is")
        print("              really the live Overleaf copy and not an older download.")

    # ---- 2. line offset ------------------------------------------------------
    print("\n2. LINE OFFSET (notes/14 quotes submitted-file line numbers)")
    print("-" * 74)
    probe = "Despite these advances, existing approaches address"
    o = next((i + 1 for i, l in enumerate(old_lines) if probe in l), None)
    n = next((i + 1 for i, l in enumerate(new_lines) if probe in l), None)
    if o and n:
        print(f"   body probe: submitted line {o} -> incoming line {n}   offset {n - o:+d}")
        if n - o == 2:
            print("   offset is +2 as documented in notes/14 and notes/16")
        else:
            print(f"   OFFSET CHANGED: use +{n - o}, not +2. Prefer the search anchors.")
    else:
        print("   probe sentence not found; rely on search anchors only")

    # ---- 3. anchors ----------------------------------------------------------
    print("\n3. PASTE ANCHORS IN THE INCOMING FILE")
    print("-" * 74)
    missing = []
    for label, anchor in ANCHORS:
        hits = [i + 1 for i, l in enumerate(new_lines) if anchor in l]
        if not hits:
            missing.append((label, anchor))
        print(f"   {'OK  ' if hits else 'MISS'} {label:<22} {str(hits[:4]):<24} {anchor[:34]}")

    print()
    if missing:
        print(f"   {len(missing)} ANCHOR(S) MISSING. A coauthor has reworded these since")
        print("   submission. Do NOT apply those steps blindly; re-read the current text first:")
        for label, anchor in missing:
            print(f"     - {label}: {anchor[:60]}")
        sys.exit(1)
    print("   all anchors present; notes/16 can be applied as written")

    # ---- companion files -----------------------------------------------------
    print("\n4. COMPANION FILES")
    print("-" * 74)
    for name in ("ref.bib", "main-thesis-version.tex"):
        inc = incoming.parent / name
        loc = REPO / "GLOBECOM" / name
        if not inc.exists():
            print(f"   {name}: not in export")
            continue
        if not loc.exists():
            print(f"   {name}: present in export, no local copy to compare")
            continue
        same = inc.read_bytes().strip() == loc.read_bytes().strip()
        print(f"   {name}: {'identical to local' if same else 'DIFFERS from local'}")


if __name__ == "__main__":
    main()
