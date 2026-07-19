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
ENTRY_V2 = f"{PKG}/train_observer_v2.py"
ENTRY_V2HY3 = f"{PKG}/train_observer_v2hy3.py"
ENTRY_V2HY3_GAMMAKIN = f"{PKG}/train_observer_v2hy3_gammakin.py"
REGIMES = f"{PKG}/regimes"


def build_jobs(args):
    windows = [int(w) for w in args.windows.split(",")]
    jobs = []
    # --gammakin is the gamma-RESIDUAL variant; it takes the same hy3 knobs, but its
    # own entry point / config (uniform dv_scale, dgamma_scale, vy_label ramp come
    # from ObserverConfigV2Hy3GammaKin defaults).
    gk = getattr(args, "gammakin", False)
    hy3 = getattr(args, "hy3", False) or gk
    entry = ENTRY_V2HY3_GAMMAKIN if gk else (ENTRY_V2HY3 if hy3 else ENTRY_V2)

    def add(regime, w):
        if gk:
            label = f"{regime}_w{w}_gamma_dv_v2hy3_gammakin" + args.tag_suffix
        elif hy3:
            # v2-Hy3 run-tag convention: S{fold}_train_w{W}_gamma_dv_v2hy3_phys_max_norm
            label = f"{regime}_w{w}_gamma_dv_v2hy3_phys_max_norm" + args.tag_suffix
        else:
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
        ]
        if args.limit_files > 0:
            a += ["--limit-files", str(args.limit_files)]
        if hy3:
            a += [
                "--w-dv", str(args.w_dv),
                "--w-slip", str(args.w_slip),
                "--gate-center", str(args.gate_center),
                "--gate-width", str(args.gate_width),
                "--mindlin-iters", str(args.mindlin_iters),
                "--crossover-hz", str(args.crossover_hz),
                "--integrator", args.integrator,
            ]
            if args.no_grounding_early_stop:
                a += ["--no-grounding-early-stop"]
            if args.grounding_patience is not None:
                a += ["--grounding-patience", str(args.grounding_patience)]
            if args.grounding_min_epochs is not None:
                a += ["--grounding-min-epochs", str(args.grounding_min_epochs)]
        else:
            a += [
                "--physics-variant", args.physics_variant,
                "--w-roller", str(args.w_roller),
            ]
            if args.no_roller_slip_weighting:
                a += ["--no-roller-slip-weighting"]
        if args.warm_from is not None:
            a += ["--warm-from", args.warm_from]
        jobs.append(ps.Job(label=label, pkg_dir=PKG, entry=entry, args=a,
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
    ap.add_argument("--hy3", action="store_true",
                    help="launch Observer v2-Hy3 jobs instead of base v2")
    ap.add_argument("--gammakin", action="store_true",
                    help="launch the v2-Hy3 gamma-RESIDUAL variant (nd711 5.1 Model 1 "
                         "no-slip base + learned dgamma); implies --hy3 knobs but uses "
                         "train_observer_v2hy3_gammakin.py and its own config defaults "
                         "(uniform dv_scale, dgamma_scale, vy_label ramp)")
    ap.add_argument("--physics-variant", default="residual", choices=["residual", "integrated"])
    ap.add_argument("--w-roller", type=float, default=1.0,
                    help="weight on promoted roller residual term")
    ap.add_argument("--no-roller-slip-weighting", action="store_true",
                    help="disable slip weighting on roller term")
    ap.add_argument("--w-dv", type=float, default=1.0,
                    help="(Hy3 only) weight on supervised ΔV term")
    ap.add_argument("--w-slip", type=float, default=0.0,
                    help="(Hy3 only) weight on the derived-slip consistency term "
                         "(|Vp|(gamma,dV) -> vpm). ~0.02 balances it against gamma's "
                         "gradient; 0 = off.")
    ap.add_argument("--no-grounding-early-stop", action="store_true",
                    help="(Hy3 only) disable grounding early-stop so every fold runs "
                         "the full budget (keeps cross-fold comparisons even)")
    ap.add_argument("--grounding-patience", type=int, default=None,
                    help="(Hy3 only) supervised-val no-improve epochs before "
                         "grounding early-stop fires")
    ap.add_argument("--grounding-min-epochs", type=int, default=None,
                    help="(Hy3 only) floor before grounding early-stop can fire")
    ap.add_argument("--gate-center", type=float, default=0.01,
                    help="(Hy3 only) slip-gate sigmoid center (m/s)")
    ap.add_argument("--gate-width", type=float, default=0.01,
                    help="(Hy3 only) slip-gate sigmoid width (m/s)")
    ap.add_argument("--mindlin-iters", type=int, default=2,
                    help="(Hy3 only) Mindlin Picard iterations")
    ap.add_argument("--crossover-hz", type=float, default=1.0,
                    help="(Hy3 only) complementary-filter crossover frequency")
    ap.add_argument("--integrator", default="rot_ab2", choices=["euler", "rot_ab2"],
                    help="(Hy3 only) strapdown integrator")
    ap.add_argument("--warm-from", type=str, default=None,
                    help="weights-only warm-start from a v1 checkpoint (one per launcher call)")
    ap.add_argument("--limit-files", type=int, default=0,
                    help="debug/smoke subset: cap discovered files per job (0=all)")
    ap.add_argument("--heartbeat", type=float, default=120.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even completed jobs")
    args = ap.parse_args()

    jobs = build_jobs(args)
    label = "v2hy3" if args.hy3 else "v2"
    print(f"[launch-a2-{label}] {len(jobs)} jobs over regimes={args.regimes} "
          f"W={args.windows} (stride={args.stride_frac}W, batch={args.per_run_batch})")

    if args.warm_cache and not args.dry_run:
        warm_entry = ENTRY_V2HY3 if args.hy3 else ENTRY_V2
        warm_name = "warm_a2_v2hy3" if args.hy3 else "warm_a2_v2"
        warm_args = ["--warm-cache-only", "--cache-dir", args.cache_dir,
                     "--scaler-csv", args.scaler_csv]
        if args.hy3:
            # Match the hy3 cache KEY to the jobs' front-end (crossover/integrator).
            warm_args += ["--crossover-hz", str(args.crossover_hz),
                          "--integrator", args.integrator]
        ps.run_blocking(warm_name, PKG, warm_entry, warm_args,
                        args.python, ps.ROOT / args.log_dir / f"{warm_name}.log")

    ps.run_sweep(jobs, max_parallel=args.max_parallel, python=args.python,
                 log_dir=args.log_dir, cores_per_job=args.cores_per_job,
                 csv_path=args.csv, dry_run=args.dry_run, force=args.force,
                 heartbeat_seconds=args.heartbeat)


if __name__ == "__main__":
    main()
