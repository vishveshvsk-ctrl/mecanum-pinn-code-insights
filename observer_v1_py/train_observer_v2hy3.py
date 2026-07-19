#!/usr/bin/env python
# =============================================================================
# train_observer_v2hy3.py — CLI entry point for Observer v2-Hy3.
#
# Mirrors train_observer_v2.py but wires ObserverConfigV2Hy3 + train_v2hy3.
# =============================================================================
from __future__ import annotations

# Import pyarrow BEFORE torch (Windows native-loader crash otherwise).
import pyarrow.feather  # noqa: F401

import argparse
from pathlib import Path

from mecanum_observer.config import CHI_GRID
from mecanum_observer.config_v2hy3 import ObserverConfigV2Hy3
from mecanum_observer.data import discover, load_regime, regime_to_kwargs
from mecanum_observer.data_v2hy3 import load_max_scaler, warm_hy3_cache
from mecanum_observer.training_v2hy3 import train_v2hy3


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the γ + ΔV Observer v2-Hy3.")
    ap.add_argument("--data-dir", type=Path,
                    default=Path("../data/Simulation_Data_MecanumSlipSpin_LugreAdamov"))
    ap.add_argument("--whitelist", type=Path, default=Path("diagnostics_combined.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("observer_v1_py/runs"))
    ap.add_argument("--regime", type=Path, default=None,
                    help="regime TOML (observer_v1_py/regimes/*.toml); merged over base.toml")
    ap.add_argument("--model", choices=["ssm", "gru"], default="ssm")
    ap.add_argument("--window", type=int, default=None, help="override regime window")
    ap.add_argument("--stride-frac", type=float, default=None,
                    help="stride = round(frac*window), e.g. 0.5")
    ap.add_argument("--epochs", type=int, default=0,
                    help="override total epochs (0 = use 200-epoch schedule)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default=None)
    ap.add_argument("--cache-dir", type=str, default=None,
                    help="decimated-500Hz cache dir ('' disables)")
    ap.add_argument("--norm", choices=["max"], default="max",
                    help="v2hy3 pins max-norm (frozen p95 scaler)")
    ap.add_argument("--scaler-csv", type=str, default=None,
                    help="max-norm: path to variable_scaler_percentiles.csv")
    ap.add_argument("--w-dv", type=float, default=None,
                    help="weight on supervised ΔV term")
    ap.add_argument("--w-slip", type=float, default=None,
                    help="weight on derived-slip consistency term (|Vp|(γ̂,ΔV̂) -> vpm)")
    ap.add_argument("--gate-center", type=float, default=None)
    ap.add_argument("--gate-width", type=float, default=None)
    ap.add_argument("--mindlin-iters", type=int, default=None)
    ap.add_argument("--crossover-hz", type=float, default=None,
                    help="complementary-filter crossover frequency")
    ap.add_argument("--integrator", choices=["euler", "rot_ab2"], default=None)
    ap.add_argument("--noise-stage", choices=["none", "real"], default=None)
    ap.add_argument("--grounding-min-epochs", type=int, default=None,
                    help="floor before grounding early-stop can fire")
    ap.add_argument("--grounding-patience", type=int, default=None,
                    help="supervised-val no-improve epochs to early-stop grounding")
    ap.add_argument("--gradnorm", action="store_true",
                    help="adaptive gradient-balanced weighting between gamma and dV")
    ap.add_argument("--gradnorm-alpha", type=float, default=None,
                    help="GradNorm restoring force (default 1.5)")
    ap.add_argument("--gradnorm-lr", type=float, default=None,
                    help="LR for the GradNorm weight params (default 0.025)")
    ap.add_argument("--cosine-diag", action="store_true",
                    help="log cos(grad L_gamma, grad L_dv) at the shared encoder on "
                         "heartbeat epochs (diagnostic only; never changes the loss)")
    ap.add_argument("--cosine-diag-every", type=int, default=None,
                    help="cosine-diag heartbeat in epochs (default 10)")
    ap.add_argument("--cosine-diag-batches", type=int, default=None,
                    help="max batches measured per heartbeat epoch (default 20)")
    ap.add_argument("--no-grounding-early-stop", action="store_true",
                    help="disable grounding early-stop (run full grounding budget)")
    ap.add_argument("--no-lr-reset", action="store_true",
                    help="disable LR/scheduler reset at the grounding->physics hand-off")
    ap.add_argument("--warm-from", type=str, default=None,
                    help="weights-only warm-start from a v1 3-state checkpoint")
    ap.add_argument("--require-gpu", action="store_true",
                    help="hard-fail if CUDA is unavailable")
    ap.add_argument("--jobs", type=int, default=None, help="dataloader workers (<=8)")
    ap.add_argument("--limit-files", type=int, default=0, help="debug subset")
    ap.add_argument("--seed", type=int, default=None, help="override regime seed")
    ap.add_argument("--run-tag", type=str, default=None,
                    help="explicit run_tag (overrides derived one)")
    ap.add_argument("--warm-cache-only", action="store_true",
                    help="single-process pre-build of the decimated cache over the FULL "
                         "mu/chi grid, then exit")
    args = ap.parse_args()

    if args.warm_cache_only:
        base = dict(data_dir=args.data_dir, whitelist_csv=args.whitelist,
                    out_dir=args.out_dir, norm_method="max")
        if args.scaler_csv is not None:
            base["scaler_csv"] = args.scaler_csv
        if args.cache_dir is not None:
            base["cache_dir"] = args.cache_dir
        # Front-end knobs are folded into the hy3 cache KEY, so the warm pass must
        # match the training front-end or the jobs will miss the cache and rebuild.
        for arg_name, field in [("crossover_hz", "vel_filter_crossover_hz"),
                                ("integrator", "vel_filter_integrator"),
                                ("noise_stage", "noise_stage")]:
            v = getattr(args, arg_name)
            if v is not None:
                base[field] = v
        base["mu_values"] = [0.3, 0.5, 0.8]
        base["chi_values"] = list(CHI_GRID)
        wc = ObserverConfigV2Hy3(**base).resolved()
        files = discover(wc)
        warm_hy3_cache(files, wc)
        print(f"[warm-cache-v2hy3] {len(files)} files selected -> {wc.cache_dir or 'off'}")
        return

    kw = dict(data_dir=args.data_dir, whitelist_csv=args.whitelist,
              out_dir=args.out_dir, model=args.model, epochs=args.epochs,
              lr=args.lr, limit_files=args.limit_files, require_gpu=args.require_gpu,
              norm_method="max")
    if args.regime is not None:
        kw.update(regime_to_kwargs(load_regime(args.regime)))
    for arg_name, field in [("batch_size", "batch_size"), ("jobs", "jobs"),
                            ("precision", "precision"), ("window", "window"),
                            ("stride_frac", "stride_frac"), ("cache_dir", "cache_dir"),
                            ("seed", "seed"), ("run_tag", "run_tag_override"),
                            ("scaler_csv", "scaler_csv"),
                            ("w_dv", "w_dv"),
                            ("w_slip", "w_slip"),
                            ("gate_center", "gate_center"),
                            ("gate_width", "gate_width"),
                            ("mindlin_iters", "mindlin_iters"),
                            ("crossover_hz", "vel_filter_crossover_hz"),
                            ("integrator", "vel_filter_integrator"),
                            ("noise_stage", "noise_stage"),
                            ("grounding_min_epochs", "grounding_min_epochs"),
                            ("grounding_patience", "grounding_patience"),
                            ("gradnorm_alpha", "gradnorm_alpha"),
                            ("gradnorm_lr", "gradnorm_lr"),
                            ("cosine_diag_every", "cosine_diag_every"),
                            ("cosine_diag_batches", "cosine_diag_batches"),
                            ("warm_from", "warm_from")]:
        v = getattr(args, arg_name)
        if v is not None:
            kw[field] = v
    if args.gradnorm:
        kw["use_gradnorm"] = True
    if args.cosine_diag:
        kw["cosine_diag"] = True
    if args.no_grounding_early_stop:
        kw["grounding_early_stop"] = False
    if args.no_lr_reset:
        kw["lr_reset_at_physics"] = False

    cfg = ObserverConfigV2Hy3(**kw)
    print(f"[cli-v2hy3] model={cfg.model} window={cfg.window} stride={cfg.eff_stride} "
          f"batch={cfg.batch_size} jobs={cfg.jobs} epochs={cfg.total_epochs} "
          f"w_dv={cfg.w_dv} gate=({cfg.gate_center},{cfg.gate_width}) "
          f"mindlin={cfg.mindlin_iters} crossover={cfg.vel_filter_crossover_hz} "
          f"integrator={cfg.vel_filter_integrator} noise={cfg.noise_stage} "
          f"warm_from={cfg.warm_from or 'none'} cache={cfg.cache_dir or 'off'}")
    train_v2hy3(cfg)


if __name__ == "__main__":
    main()
