#!/usr/bin/env python
"""Diagnose descend-vs-plateau for a controller-tuning run.

Reads the live `trace.csv` (iter,score,best_so_far) if present, else the
post-hoc `trials.json` (iter,score,...), computes the best-so-far (cumulative
min) curve, and prints a verdict:

  STILL DESCENDING  best found inside the last window  -> budget-limited, raise it
  PLATEAUED         cummin flat (<1%) over the last window -> converged / stuck
  SLOWING           some but shrinking improvement in the last window

Usage:
  python diagnose_convergence.py runs_controller/asmc_clean [runs_controller/pid_clean ...]
  python diagnose_convergence.py --all runs_controller      # every *_clean / *_noisy dir
"""
import sys, json, os, glob, math

SPARK = "▁▂▃▄▅▆▇█"


def load_evals(rundir):
    """Return (source, [(iter, score), ...]) sorted by iter. trace.csv wins."""
    tp = os.path.join(rundir, "trace.csv")
    if os.path.exists(tp):
        rows = []
        with open(tp) as f:
            next(f, None)  # header
            for line in f:
                p = line.strip().split(",")
                if len(p) >= 2:
                    rows.append((int(float(p[0])), float(p[1])))
        if rows:
            return "trace.csv", sorted(rows)
    jp = os.path.join(rundir, "trials.json")
    if os.path.exists(jp):
        d = json.load(open(jp))
        rows = [(int(t["iter"]), float(t["score"])) for t in d if t.get("score") is not None]
        return "trials.json", sorted(rows)
    return None, []


def cummin(scores):
    out, m = [], math.inf
    for s in scores:
        m = min(m, s)
        out.append(m)
    return out


def sparkline(vals):
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return SPARK[0] * len(vals)
    return "".join(SPARK[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


def diagnose(rundir):
    src, evals = load_evals(rundir)
    name = os.path.basename(rundir.rstrip("/\\"))
    if not evals:
        print(f"[{name}] no trace.csv or trials.json found in {rundir}")
        return
    iters = [e[0] for e in evals]
    scores = [e[1] for e in evals]
    cm = cummin(scores)
    n = len(cm)
    best = cm[-1]
    argbest = iters[min(range(n), key=lambda i: (cm[i], i))]  # first eval reaching best
    # window ~ one generation's worth (>=8, or 20% of run)
    W = max(8, round(0.2 * n))
    W = min(W, n - 1) if n > 1 else 1
    ref = cm[max(0, n - 1 - W)]
    rel = (ref - best) / max(abs(ref), 1e-9)
    stall = iters[-1] - argbest

    if argbest >= iters[-1] - W:
        verdict = "STILL DESCENDING — best is inside the last window ⇒ budget-limited, raise budget"
    elif rel < 0.01:
        verdict = "PLATEAUED — best-so-far flat (<1%) over the last window ⇒ converged / stuck"
    else:
        verdict = "SLOWING — improvement continuing but shrinking in the last window"

    print(f"[{name}]  source={src}")
    print(f"   evals={n}   best={best:.4f} (first reached @ eval {argbest})   "
          f"stall={stall} evals since best")
    print(f"   last-{W}-eval cummin improvement = {rel*100:.2f}%")
    print(f"   cummin: {sparkline(cm)}  ({cm[0]:.3g} → {best:.3g})")
    print(f"   VERDICT: {verdict}\n")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); return
    if args[0] == "--all":
        root = args[1] if len(args) > 1 else "runs_controller"
        dirs = sorted(d for d in glob.glob(os.path.join(root, "*_*"))
                      if os.path.isdir(d) and (os.path.exists(os.path.join(d, "trace.csv"))
                                               or os.path.exists(os.path.join(d, "trials.json"))))
    else:
        dirs = args
    for d in dirs:
        diagnose(d)


if __name__ == "__main__":
    main()
