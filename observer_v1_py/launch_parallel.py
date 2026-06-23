#!/usr/bin/env python
# =============================================================================
# launch_parallel.py — A2 (state observer) adapter for the shared sweep core.
#
# One launcher idiom, ONE machine knob: N = --max-parallel (degree of
# parallelism). There is NO 6-vs-24 dichotomy — the model is tiny (~6 k params)
# and launch-bound, so a single run under-utilises ANY GPU; concurrent runs fill
# it. N just scales with the machine (pick it from `nvidia-smi` GPU util, NOT from
# VRAM: at batch ~4 k a run is ~0.3-0.5 GB; CPU/IO bind first).
#
# Flow:  build the decimated cache ONCE (--warm-cache, single process), then fan
# out N concurrent `train_observer.py` jobs that all read the warm .npz cache.
#
# Run from code_insights/ :
#   python observer_v1_py/launch_parallel.py --warm-cache --max-parallel 3
#   python observer_v1_py/launch_parallel.py --dry-run
# (Linux) start NVIDIA MPS first for true concurrent kernels:
#   nvidia-cuda-mps-control -d   ... echo quit | nvidia-cuda-mps-control
#
# Default sweep = the W-ablation: W in {8,16,32} x {S1_train, S2_train,
# S3_chi_kfold x 4 chi-folds} = 18 jobs, stride=0.5W, 120-epoch curriculum.
# Resume-safe: a run whose runs/<label>/metrics.json exists is skipped.
# =============================================================================
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # code_insights/
import parallel_sweep as ps                                        # noqa: E402

PKG = "observer_v1_py"
ENTRY = f"{PKG}/train_observer.py"
REGIMES = f"{PKG}/regimes"


def build_jobs(args):
    windows = [int(w) for w in args.windows.split(",")]
    chis = [float(c) for c in args.chis.split(",")] if args.chis else []
    jobs = []

    def add(regime, w, test_chi=None):
        label = (f"{regime}_w{w}" + (f"_chi{test_chi:g}" if test_chi is not None else "")
                 + args.tag_suffix)
        a = ["--regime", f"{REGIMES}/{regime}.toml",
             "--window", str(w), "--stride-frac", str(args.stride_frac),
             "--phase-epochs", str(args.phase_epochs),
             "--cache-dir", args.cache_dir, "--batch-size", str(args.per_run_batch),
             "--jobs", str(args.dl_workers), "--norm", args.norm,
             "--out-dir", args.out_dir, "--run-tag", label]
        if args.norm == "max":
            a += ["--scaler-csv", args.scaler_csv]
        if test_chi is not None:
            a += ["--test-chi", str(test_chi)]
        jobs.append(ps.Job(label=label, pkg_dir=PKG, entry=ENTRY, args=a,
                            run_dir=f"{args.out_dir}/{label}"))

    for regime in args.regimes.split(","):
        regime = regime.strip()
        if not regime:
            continue
        if regime.startswith("S3"):
            for w in windows:
                for c in chis:
                    add(regime, w, test_chi=c)
        else:
            for w in windows:
                add(regime, w)
    return jobs


def main():
    ap = argparse.ArgumentParser(description="A2 observer parallel sweep (shared core).")
    ap.add_argument("--max-parallel", type=int, default=2,
                    help="N concurrent runs; pick from nvidia-smi GPU util (NOT VRAM).")
    ap.add_argument("--regimes", default="S1_train,S2_train,S3_chi_kfold")
    ap.add_argument("--windows", default="8,16,32")
    ap.add_argument("--chis", default="0.0,0.002,0.005,0.008")
    ap.add_argument("--phase-epochs", type=int, default=120)
    ap.add_argument("--stride-frac", type=float, default=0.5)
    ap.add_argument("--per-run-batch", type=int, default=4096,
                    help="per-job batch size (a throughput knob, not a memory one)")
    ap.add_argument("--dl-workers", type=int, default=4,
                    help="dataloader workers PER job (keep N*dl_workers <= cores, <=8)")
    ap.add_argument("--cache-dir", default=f"{PKG}/cache_decim")
    ap.add_argument("--norm", default="var", choices=["var", "max"],
                    help="var = z-score (fit on train); max = frozen p95 scaler from --scaler-csv")
    ap.add_argument("--scaler-csv",
                    default="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/variable_scaler_percentiles.csv")
    ap.add_argument("--tag-suffix", default="",
                    help="appended to every run label, e.g. _non_phys_max_norm")
    ap.add_argument("--out-dir", default=f"{PKG}/runs")
    ap.add_argument("--cores-per-job", type=int, default=4)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--log-dir", default=f"{PKG}/runs/_parallel_logs")
    ap.add_argument("--csv", default=f"{PKG}/runs/sweep_results.csv")
    ap.add_argument("--warm-cache", action="store_true",
                    help="single-process decimated-cache pre-build before fan-out")
    ap.add_argument("--heartbeat", type=float, default=120.0,
                    help="seconds between terminal heartbeats + sweep_status.txt refresh")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even completed jobs")
    args = ap.parse_args()

    jobs = build_jobs(args)
    print(f"[launch-a2] {len(jobs)} jobs over regimes={args.regimes} "
          f"W={args.windows} (stride={args.stride_frac}W, batch={args.per_run_batch})")

    if args.warm_cache and not args.dry_run:
        ps.run_blocking("warm_a2", PKG, ENTRY,
                        ["--warm-cache-only", "--cache-dir", args.cache_dir],
                        args.python, ps.ROOT / args.log_dir / "warm_a2.log")

    ps.run_sweep(jobs, max_parallel=args.max_parallel, python=args.python,
                 log_dir=args.log_dir, cores_per_job=args.cores_per_job,
                 csv_path=args.csv, dry_run=args.dry_run, force=args.force,
                 heartbeat_seconds=args.heartbeat)


if __name__ == "__main__":
    main()
