#!/usr/bin/env python
# =============================================================================
# launch_parallel_v2.py — A2 observer v2 parallel-sweep launcher.
#
# Standalone v2 launcher; v1 launch_parallel.py is left untouched.  Reuses the
# shared parallel_sweep core.  Default campaign: S1 + S2 folds, window 32,
# max-norm, residual physics with the promoted roller term.
#
# Run from code_insights/:
#   python observer_v1_py/launch_parallel_v2.py --warm-cache --max-parallel 2
#   python observer_v1_py/launch_parallel_v2.py --dry-run
# =============================================================================
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # code_insights/
import parallel_sweep as ps                                        # noqa: E402

PKG = "observer_v1_py"
ENTRY = f"{PKG}/train_observer_v2.py"
REGIMES = f"{PKG}/regimes"


def build_jobs(args):
    windows = [int(w) for w in args.windows.split(",")]
    jobs = []

    def add(regime, w):
        # v2 run-tag convention: S{fold}_train_w{W}_gamma_v2_phys_max_norm
        label = f"{regime}_w{w}_gamma_v2_phys_max_norm" + args.tag_suffix
        a = [
            "--regime", f"{REGIMES}/{regime}.toml",
            "--window", str(w),
            "--stride-frac", str(args.stride_frac),
            "--cache-dir", args.cache_dir,
            "--batch-size", str(args.per_run_batch),
            "--jobs", str(args.dl_workers),
            "--norm", "max",
            "--out-dir", args.out_dir,
            "--run-tag", label,
            "--scaler-csv", args.scaler_csv,
            "--physics-variant", args.physics_variant,
            "--w-roller", str(args.w_roller),
        ]
        if args.warm_from is not None:
            a += ["--warm-from", args.warm_from]
        if args.no_roller_slip_weighting:
            a += ["--no-roller-slip-weighting"]
        jobs.append(ps.Job(label=label, pkg_dir=PKG, entry=ENTRY, args=a,
                            run_dir=f"{args.out_dir}/{label}"))

    for regime in args.regimes.split(","):
        regime = regime.strip()
        if not regime:
            continue
        for w in windows:
            add(regime, w)
    return jobs


def main():
    ap = argparse.ArgumentParser(description="A2 observer v2 parallel sweep.")
    ap.add_argument("--max-parallel", type=int, default=2,
                    help="N concurrent runs; pick from nvidia-smi GPU util.")
    ap.add_argument("--regimes", default="S1_train,S2_train",
                    help="comma-separated regime TOML stems")
    ap.add_argument("--windows", default="32",
                    help="comma-separated window sizes")
    ap.add_argument("--per-run-batch", type=int, default=4096)
    ap.add_argument("--dl-workers", type=int, default=4,
                    help="dataloader workers PER job (keep N*dl_workers <= cores, <=8)")
    ap.add_argument("--stride-frac", type=float, default=0.5)
    ap.add_argument("--cache-dir", default=f"{PKG}/cache_decim")
    ap.add_argument("--scaler-csv",
                    default="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/variable_scaler_percentiles.csv")
    ap.add_argument("--tag-suffix", default="",
                    help="appended to every run label")
    ap.add_argument("--out-dir", default=f"{PKG}/runs")
    ap.add_argument("--cores-per-job", type=int, default=4)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--log-dir", default=f"{PKG}/runs/_parallel_logs_v2")
    ap.add_argument("--csv", default=f"{PKG}/runs/sweep_results_v2.csv")
    ap.add_argument("--warm-cache", action="store_true",
                    help="single-process decimated-cache pre-build before fan-out")
    ap.add_argument("--physics-variant", default="residual", choices=["residual", "integrated"])
    ap.add_argument("--w-roller", type=float, default=1.0,
                    help="weight on promoted roller residual term")
    ap.add_argument("--no-roller-slip-weighting", action="store_true",
                    help="disable slip weighting on roller term")
    ap.add_argument("--warm-from", type=str, default=None,
                    help="weights-only warm-start from a v1 checkpoint (one per launcher call)")
    ap.add_argument("--heartbeat", type=float, default=120.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even completed jobs")
    args = ap.parse_args()

    jobs = build_jobs(args)
    print(f"[launch-a2-v2] {len(jobs)} jobs over regimes={args.regimes} "
          f"W={args.windows} (stride={args.stride_frac}W, batch={args.per_run_batch})")

    if args.warm_cache and not args.dry_run:
        ps.run_blocking("warm_a2_v2", PKG, ENTRY,
                        ["--warm-cache-only", "--cache-dir", args.cache_dir,
                         "--scaler-csv", args.scaler_csv],
                        args.python, ps.ROOT / args.log_dir / "warm_a2_v2.log")

    ps.run_sweep(jobs, max_parallel=args.max_parallel, python=args.python,
                 log_dir=args.log_dir, cores_per_job=args.cores_per_job,
                 csv_path=args.csv, dry_run=args.dry_run, force=args.force,
                 heartbeat_seconds=args.heartbeat)


if __name__ == "__main__":
    main()
