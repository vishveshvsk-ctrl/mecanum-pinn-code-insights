#!/usr/bin/env python
# =============================================================================
# launch_parallel.py — A1 (force-recon PINN) adapter for the shared sweep core.
#
# Mirrors observer_v1_py/launch_parallel.py: same idiom, same shared core
# (code_insights/parallel_sweep.py). N = --max-parallel is the only machine knob.
# Build the decimated cache ONCE (--warm-cache), then fan out N independent
# `train.py both` runs over the regime grid (× optional SSM-dim model sweep).
#
# Run from code_insights/ :
#   # build the cache once (single process), then run 4 at a time:
#   python Mecanum_PINN_Mamba_ForceRecon_v1/launch_parallel.py --warm-cache --max-parallel 4
#   python Mecanum_PINN_Mamba_ForceRecon_v1/launch_parallel.py --dry-run
#   # A2-like window ablation (DEFAULT): seq_len in {8,16,32}, stride=0.5*seq_len,
#   # crossed with {S1,S2} = 6 runs (S3 chi k-fold excluded by default). Override
#   # with --windows / --stride-frac / --regimes.
#   python Mecanum_PINN_Mamba_ForceRecon_v1/launch_parallel.py --windows 8,16,32
#   # model sweep: ssm_d_model x ssm_d_state grid, crossed with windows x regimes
#   python Mecanum_PINN_Mamba_ForceRecon_v1/launch_parallel.py --ssm-dims 32x16,48x16,32x24
#
# Resume-safe: runs whose Mecanum_PINN_Mamba_ForceRecon_v1/runs/checkpoints/<label>/metrics.json
# exist are skipped. Results land in one ranking CSV (MAE(inv)/MAE(fwd)/inv-fwd div + losses).
# =============================================================================
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # code_insights/
import parallel_sweep as ps                                        # noqa: E402

PKG = "Mecanum_PINN_Mamba_ForceRecon_v1"
ENTRY = f"{PKG}/train.py"
REGIMES = "observer_v1_py/regimes"          # shared regime TOMLs (same selection as A2)
PKG_ROOT = Path(__file__).resolve().parent   # absolute path to Mecanum_PINN_Mamba_ForceRecon_v1/


def build_jobs(args):
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    chis = [float(c) for c in args.chis.split(",")] if args.chis else []
    dims = [d.strip() for d in args.ssm_dims.split(",") if d.strip()]
    windows = [int(w) for w in args.windows.split(",")]
    multi_dim = len(dims) > 1
    jobs = []

    def add(regime, dim, window, test_chi=None):
        D, N = (int(x) for x in dim.lower().split("x"))
        # A2-like window ablation: A1's seq_len IS the lookback window; stride =
        # round(stride_frac * window) (windows/epoch ~ 1/stride, so total scan-steps
        # ~ constant across windows -> the 3 window runs cost roughly equally).
        stride = max(1, round(args.stride_frac * window))
        label = f"a1_{regime}"
        if multi_dim:
            label += f"_d{D}n{N}"
        label += f"_w{window}"
        if test_chi is not None:
            label += f"_chi{test_chi:g}"
        a = ["both", "--regime", f"{REGIMES}/{regime}.toml",
             "--cache-dir", args.cache_dir, "--batch-size", str(args.per_run_batch),
             "--run-tag", label,
             "--set", f"seq_len={window}", "--set", f"stride={stride}",
             "--set", f"ssm_d_model={D}", "--set", f"ssm_d_state={N}",
             "--set", f"num_workers={args.dl_workers}"]
        if test_chi is not None:
            a += ["--test-chi", str(test_chi)]
        if args.vram is not None:
            a += ["--vram", str(args.vram)]
        if args.no_lbfgs:
            a += ["--no-lbfgs"]
        if args.epoch_scale is not None:
            a += ["--epoch-scale", str(args.epoch_scale)]
        a += list(args.extra)                       # extra --set passthrough (applied to all)
        jobs.append(ps.Job(label=label, pkg_dir=PKG, entry=ENTRY, args=a,
                            run_dir=f"{args.ckpt_dir}/{label}"))

    for window in windows:
        for dim in dims:
            for regime in regimes:
                if regime.startswith("S3"):
                    for c in chis:                  # one k-fold run per held-out chi
                        add(regime, dim, window, test_chi=c)
                else:
                    add(regime, dim, window)
    return jobs


def main():
    ap = argparse.ArgumentParser(description="A1 force-recon parallel sweep (shared core).")
    ap.add_argument("--max-parallel", type=int, default=4,
                    help="N concurrent runs; set from nvidia-smi GPU util (NOT VRAM).")
    ap.add_argument("--regimes", default="S1_train,S2_train")
    ap.add_argument("--chis", default="0.0,0.002,0.005,0.008", help="S3 held-out chi folds")
    ap.add_argument("--windows", default="8,16,32",
                    help="A2-like window ablation: seq_len values to sweep (= lookback)")
    ap.add_argument("--stride-frac", type=float, default=0.5,
                    help="per job: stride = round(stride_frac * window) (A2 idiom; 0.5W)")
    ap.add_argument("--ssm-dims", default="32x16",
                    help="comma list of ssm_d_model x ssm_d_state (e.g. 32x16,48x16)")
    ap.add_argument("--per-run-batch", type=int, default=1024,
                    help="per-job batch size (model is tiny; this is a throughput knob)")
    ap.add_argument("--dl-workers", type=int, default=2,
                    help="dataloader workers PER job (keep N*dl_workers <= cores, <=8)")
    ap.add_argument("--vram", type=int, choices=[6, 12, 24], default=None,
                    help="VRAM tier for amp/workers; --per-run-batch still wins.")
    ap.add_argument("--no-lbfgs", action="store_true", help="disable L-BFGS refine (faster)")
    ap.add_argument("--epoch-scale", type=float, default=None,
                    help="scale all phase epochs (0.5 ~halves -> ~220 ep; ~0.27 -> ~120 ep)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra tokens appended verbatim to each job (e.g. --extra --set lr=2e-3)")
    ap.add_argument("--cache-dir", default=str(PKG_ROOT / "cache_decim"))
    ap.add_argument("--ckpt-dir", default=str(PKG_ROOT / "runs" / "checkpoints"))
    ap.add_argument("--cores-per-job", type=int, default=4)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--log-dir", default=str(PKG_ROOT / "runs" / "_parallel_logs"))
    ap.add_argument("--csv", default=str(PKG_ROOT / "runs" / "sweep_results.csv"))
    ap.add_argument("--warm-cache", action="store_true",
                    help="single-process decimated-cache pre-build before fan-out")
    ap.add_argument("--heartbeat", type=float, default=120.0,
                    help="seconds between terminal heartbeats + sweep_status.txt refresh")
    ap.add_argument("--cross-eval", action="store_true",
                    help="after the main sweep, evaluate each S1/S2 checkpoint on the "
                         "opposite regime's test split (via cross_eval.py)")
    ap.add_argument("--cross-report", action="store_true",
                    help="after the sweep (and --cross-eval if set), run cross_report.py")
    ap.add_argument("--cross-stage", choices=["forward", "inverse", "both"], default="both",
                    help="stage passed to cross_eval.py (default: both)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even completed jobs")
    args = ap.parse_args()

    jobs = build_jobs(args)
    print(f"[launch-a1] {len(jobs)} jobs over regimes={args.regimes} "
          f"windows={args.windows} (stride={args.stride_frac}W) "
          f"ssm_dims={args.ssm_dims} (batch={args.per_run_batch})")

    if args.warm_cache and not args.dry_run:
        warm_args = ["both", "--warm-cache-only", "--cache-dir", args.cache_dir]
        if args.vram is not None:
            warm_args += ["--vram", str(args.vram)]
        ps.run_blocking("warm_a1", PKG, ENTRY, warm_args, args.python,
                        ps.ROOT / args.log_dir / "warm_a1.log")

    result = ps.run_sweep(jobs, max_parallel=args.max_parallel, python=args.python,
                          log_dir=args.log_dir, cores_per_job=args.cores_per_job,
                          csv_path=args.csv, dry_run=args.dry_run, force=args.force,
                          heartbeat_seconds=args.heartbeat)

    if args.cross_eval and not args.dry_run:
        done_labels = set(result.get("done", []))
        cross_jobs = []
        for j in jobs:
            if j.label not in done_labels:
                continue
            regime = Path(j.args[j.args.index("--regime") + 1]).stem
            if regime not in ("S1_train", "S2_train"):
                continue
            ckpt = f"{args.ckpt_dir}/{j.label}/inverse_lbfgs.pth"
            cross_jobs.append(ps.Job(
                label=f"cross_{j.label}", pkg_dir=PKG,
                entry=f"{PKG}/cross_eval.py",
                args=["--ckpt", ckpt, "--stage", args.cross_stage],
                run_dir=None))
        if cross_jobs:
            print(f"[launch-a1] cross-eval: {len(cross_jobs)} jobs")
            ps.run_sweep(cross_jobs, max_parallel=args.max_parallel, python=args.python,
                         log_dir=args.log_dir, cores_per_job=args.cores_per_job,
                         csv_path=None, dry_run=False, force=args.force,
                         heartbeat_seconds=args.heartbeat)

    if args.cross_report and not args.dry_run:
        ps.run_blocking("cross_report", PKG, f"{PKG}/cross_report.py",
                        ["--ckpt-dir", args.ckpt_dir], args.python,
                        ps.ROOT / args.log_dir / "cross_report.log")


if __name__ == "__main__":
    main()
