#!/usr/bin/env python
# =============================================================================
# run_cross_study.py — Fan out cross_eval.py over all finished A1 runs.
#
# Discovers run directories under the checkpoint root and runs cross_eval.py for
# each one whose inverse_lbfgs.pth exists. Skips runs that already have
# cross_metrics.json unless --force is set. Finally runs cross_report.py.
#
# Run from code_insights/:
#   python Mecanum_PINN_Mamba_ForceRecon_v1/run_cross_study.py
#   python Mecanum_PINN_Mamba_ForceRecon_v1/run_cross_study.py --force --ckpt-dir ...
# =============================================================================
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PKG_ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT_DIR = PKG_ROOT / "checkpoints_mamba_v1"
CROSS_EVAL = PKG_ROOT / "cross_eval.py"
CROSS_REPORT = PKG_ROOT / "cross_report.py"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fan out A1 cross-subset evaluation over finished runs.")
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR,
                    help="checkpoint root containing run directories")
    ap.add_argument("--python", default=sys.executable,
                    help="Python executable")
    ap.add_argument("--stage", choices=["forward", "inverse", "both"], default="both",
                    help="stage passed to cross_eval.py")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if cross_metrics.json already exists")
    ap.add_argument("--no-report", action="store_true",
                    help="skip the final cross_report.py aggregation")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise SystemExit(f"checkpoint directory not found: {ckpt_dir}")

    run_dirs = sorted(d for d in ckpt_dir.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"[cross-study] no run directories under {ckpt_dir}")
        return

    todo = []
    for run_dir in run_dirs:
        final_ckpt = run_dir / "inverse_lbfgs.pth"
        if not final_ckpt.exists():
            print(f"[cross-study] skip {run_dir.name}: no inverse_lbfgs.pth")
            continue
        cross_metrics = run_dir / "cross_metrics.json"
        if cross_metrics.exists() and not args.force:
            print(f"[cross-study] skip {run_dir.name}: cross_metrics.json exists")
            continue
        todo.append(run_dir)

    if not todo:
        print("[cross-study] nothing to evaluate")
    else:
        print(f"[cross-study] {len(todo)} runs to evaluate")
        for run_dir in todo:
            final_ckpt = run_dir / "inverse_lbfgs.pth"
            cmd = [args.python, str(CROSS_EVAL),
                   "--ckpt", str(final_ckpt), "--stage", args.stage]
            print(f"[cross-study] {' '.join(cmd)}")
            rc = subprocess.call(cmd, cwd=str(ckpt_dir.parent.parent))
            if rc != 0:
                print(f"[cross-study] WARNING: {run_dir.name} failed (rc={rc})")

    if not args.no_report:
        print("[cross-study] running cross_report.py")
        rc = subprocess.call([args.python, str(CROSS_REPORT),
                              "--ckpt-dir", str(ckpt_dir)],
                             cwd=str(ckpt_dir.parent.parent))
        if rc != 0:
            raise SystemExit(f"cross_report.py failed (rc={rc})")


if __name__ == "__main__":
    main()
