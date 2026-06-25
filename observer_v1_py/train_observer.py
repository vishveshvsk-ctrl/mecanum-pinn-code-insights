#!/usr/bin/env python
# =============================================================================
# train_observer.py — CLI for the Approach-2 causal state observer.
#
# Streaming, resume-aware (re-run the same command to continue from the last
# epoch checkpoint). One model x one window per invocation; run twice per model
# for the window ablation, e.g.:
#
#   python train_observer.py --model ssm --window 8
#   python train_observer.py --model ssm --window 32
#   python train_observer.py --model gru --window 8
#   python train_observer.py --model gru --window 32
#
# Runs on the WSL machine (two GPUs). --precision auto picks bf16 on the Ampere
# RTX 3060 and fp16 on the Turing Quadro RTX 6000 (no native bf16).
# =============================================================================
from __future__ import annotations

# Import pyarrow BEFORE anything that pulls in torch (Windows native-loader
# crash otherwise) — same load-order lock as train_GPU_PINN_v14_py/train.py.
import pyarrow.feather  # noqa: F401

import argparse
from pathlib import Path

from mecanum_observer.config import CHI_GRID, ObserverConfig, VRAM_PRESETS
from mecanum_observer.data import discover, load_regime, regime_to_kwargs, warm_cache
from mecanum_observer.training import train


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Approach-2 state observer.")
    ap.add_argument("--data-dir", type=Path,
                    default=Path("../data/Simulation_Data_MecanumSlipSpin_LugreAdamov"))
    ap.add_argument("--whitelist", type=Path, default=Path("diagnostics_combined.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("observer_v1_py/runs"))
    ap.add_argument("--regime", type=Path, default=None,
                    help="regime TOML (observer_v1_py/regimes/*.toml); merged over base.toml")
    ap.add_argument("--vram", type=int, choices=[6, 24], default=None,
                    help="machine preset: 6=RTX3060/16GB (bf16), 24=Quadro RTX6000 (fp32). "
                         "Sets batch/jobs/precision/cache; explicit flags still win.")
    ap.add_argument("--model", choices=["ssm", "gru"], default="ssm")
    ap.add_argument("--window", type=int, default=None, help="override regime window")
    ap.add_argument("--stride-frac", type=float, default=None,
                    help="stride = round(frac*window), e.g. 0.5")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--phase-epochs", type=int, default=None,
                    help="scale the 5-phase schedule to this total (e.g. 120)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default=None)
    ap.add_argument("--cache-dir", type=str, default=None,
                    help="decimated-500Hz cache dir ('' disables)")
    ap.add_argument("--norm", choices=["var", "max"], default=None,
                    help="normalization: 'var' (z-score, fit on train) or 'max' (frozen p95 scaler)")
    ap.add_argument("--scaler-csv", type=str, default=None,
                    help="max-norm: path to variable_scaler_percentiles.csv")
    ap.add_argument("--velocity-prop-loss", action="store_true",
                    help="enable analytical one-step velocity-propagation loss/metric")
    ap.add_argument("--require-gpu", action="store_true",
                    help="hard-fail if CUDA is unavailable")
    ap.add_argument("--jobs", type=int, default=None, help="dataloader workers (<=8)")
    ap.add_argument("--limit-files", type=int, default=0, help="debug subset")
    ap.add_argument("--seed", type=int, default=None, help="override regime seed")
    ap.add_argument("--test-chi", type=float, default=None,
                    help="S3: held-out χ for this k-fold run (overrides regime)")
    ap.add_argument("--run-tag", type=str, default=None,
                    help="explicit run_tag (overrides the derived one); the parallel "
                         "launcher pins it to the job label for a deterministic run dir")
    ap.add_argument("--warm-cache-only", action="store_true",
                    help="single-process pre-build of the decimated cache over the FULL "
                         "μ/χ grid (so every regime+S3-fold run hits warm cache), then exit")
    args = ap.parse_args()

    # --warm-cache-only: decimate the superset (all μ, all χ, all profiles) once,
    # single-process, so the fan-out jobs never race to write the same .npz.
    if args.warm_cache_only:
        base = dict(data_dir=args.data_dir, whitelist_csv=args.whitelist,
                    out_dir=args.out_dir)
        if args.vram is not None:
            base.update(VRAM_PRESETS[args.vram])
        if args.cache_dir is not None:
            base["cache_dir"] = args.cache_dir
        base["mu_values"] = [0.3, 0.5, 0.8]
        base["chi_values"] = list(CHI_GRID)
        wc = ObserverConfig(**base).resolved()
        files = discover(wc)
        warm_cache(files, wc.cache_dir)
        print(f"[warm-cache] {len(files)} files selected -> {wc.cache_dir or 'off'}")
        return

    # Precedence: dataclass defaults < --vram preset < regime TOML < explicit CLI.
    kw = dict(data_dir=args.data_dir, whitelist_csv=args.whitelist,
              out_dir=args.out_dir, model=args.model, epochs=args.epochs,
              lr=args.lr, limit_files=args.limit_files, require_gpu=args.require_gpu)
    if args.vram is not None:
        kw.update(VRAM_PRESETS[args.vram])
    if args.regime is not None:
        kw.update(regime_to_kwargs(load_regime(args.regime)))
    for arg_name, field in [("batch_size", "batch_size"), ("jobs", "jobs"),
                            ("precision", "precision"), ("window", "window"),
                            ("stride_frac", "stride_frac"), ("cache_dir", "cache_dir"),
                            ("phase_epochs", "phase_total_epochs"),
                            ("seed", "seed"), ("test_chi", "chi_fold_test"),
                            ("run_tag", "run_tag_override"),
                            ("norm", "norm_method"), ("scaler_csv", "scaler_csv"),
                            ("velocity_prop_loss", "velocity_prop_loss")]:
        v = getattr(args, arg_name)
        if v is not None:
            kw[field] = v
    cfg = ObserverConfig(**kw)
    print(f"[cli] vram={args.vram} regime={cfg.regime_name} model={cfg.model} "
          f"window={cfg.window} stride={cfg.eff_stride} batch={cfg.batch_size} "
          f"jobs={cfg.jobs} phases={cfg.phases}({cfg.phase_total_epochs or 'full'}) "
          f"physics={cfg.physics_loss} vp_loss={cfg.velocity_prop_loss} "
          f"cache={cfg.cache_dir or 'off'}")
    train(cfg)


if __name__ == "__main__":
    main()
