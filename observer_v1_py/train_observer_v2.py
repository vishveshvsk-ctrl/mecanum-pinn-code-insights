#!/usr/bin/env python
# =============================================================================
# train_observer_v2.py — CLI entry point for the γ-only Observer v2.
#
# Mirrors v1 CLI semantics but wires ObserverConfigV2 + train_v2.  Runs on the
# WSL/Windows machine; --precision auto picks bf16 on Ampere and fp32 elsewhere.
# =============================================================================
from __future__ import annotations

# Import pyarrow BEFORE torch (Windows native-loader crash otherwise).
import pyarrow.feather  # noqa: F401

import argparse
from pathlib import Path

from mecanum_observer.config import CHI_GRID
from mecanum_observer.config_v2 import ObserverConfigV2
from mecanum_observer.data import discover, load_regime, regime_to_kwargs, warm_cache
from mecanum_observer.data_v2 import load_max_scaler
from mecanum_observer.training_v2 import train_v2


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the γ-only Observer v2.")
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
                    help="override total epochs (0 = use v2 200-epoch schedule)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default=None)
    ap.add_argument("--cache-dir", type=str, default=None,
                    help="decimated-500Hz cache dir ('' disables)")
    ap.add_argument("--norm", choices=["max"], default="max",
                    help="v2 pins max-norm (frozen p95 scaler)")
    ap.add_argument("--scaler-csv", type=str, default=None,
                    help="max-norm: path to variable_scaler_percentiles.csv")
    ap.add_argument("--physics-variant", choices=["residual", "integrated"],
                    default=None, help="physics-loss variant")
    ap.add_argument("--w-roller", type=float, default=None,
                    help="weight on promoted roller residual term")
    ap.add_argument("--no-roller-slip-weighting", action="store_true",
                    help="disable slip weighting on the roller term")
    ap.add_argument("--warm-from", type=str, default=None,
                    help="weights-only warm-start from a v1 3-state checkpoint")
    ap.add_argument("--require-gpu", action="store_true",
                    help="hard-fail if CUDA is unavailable")
    ap.add_argument("--jobs", type=int, default=None, help="dataloader workers (<=8)")
    ap.add_argument("--limit-files", type=int, default=0, help="debug subset")
    ap.add_argument("--seed", type=int, default=None, help="override regime seed")
    ap.add_argument("--run-tag", type=str, default=None,
                    help="explicit run_tag (overrides the derived one)")
    ap.add_argument("--warm-cache-only", action="store_true",
                    help="single-process pre-build of the decimated cache over the FULL "
                         "mu/chi grid, then exit")
    args = ap.parse_args()

    # --warm-cache-only: decimate the superset once, single-process.
    if args.warm_cache_only:
        base = dict(data_dir=args.data_dir, whitelist_csv=args.whitelist,
                    out_dir=args.out_dir, norm_method="max")
        if args.scaler_csv is not None:
            base["scaler_csv"] = args.scaler_csv
        if args.cache_dir is not None:
            base["cache_dir"] = args.cache_dir
        base["mu_values"] = [0.3, 0.5, 0.8]
        base["chi_values"] = list(CHI_GRID)
        wc = ObserverConfigV2(**base).resolved()
        files = discover(wc)
        warm_cache(files, wc.cache_dir)
        print(f"[warm-cache-v2] {len(files)} files selected -> {wc.cache_dir or 'off'}")
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
                            ("physics_variant", "physics_variant"),
                            ("w_roller", "w_roller"),
                            ("warm_from", "warm_from")]:
        v = getattr(args, arg_name)
        if v is not None:
            kw[field] = v
    if args.no_roller_slip_weighting:
        kw["roller_slip_weighting"] = False

    cfg = ObserverConfigV2(**kw)
    print(f"[cli-v2] model={cfg.model} window={cfg.window} stride={cfg.eff_stride} "
          f"batch={cfg.batch_size} jobs={cfg.jobs} epochs={cfg.total_epochs} "
          f"physics_variant={cfg.physics_variant} w_roller={cfg.w_roller} "
          f"roller_slip_weighting={cfg.roller_slip_weighting} "
          f"warm_from={cfg.warm_from or 'none'} cache={cfg.cache_dir or 'off'}")
    train_v2(cfg)


if __name__ == "__main__":
    main()
