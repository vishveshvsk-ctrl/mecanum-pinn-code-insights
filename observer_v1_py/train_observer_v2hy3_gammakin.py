#!/usr/bin/env python
# =============================================================================
# train_observer_v2hy3_gammakin.py — CLI for the gamma-RESIDUAL variant.
#
# Separate entry point so train_observer_v2hy3.py / ObserverConfigV2Hy3 /
# training_v2hy3.py stay byte-identical and every run to date stays reproducible.
# Wires ObserverConfigV2Hy3GammaKin + train_v2hy3_gammakin.
#
#   gamma_hat = gamma_noslip(V_y_used) + dgamma_hat*dgamma_scale     (nd711 5.1 Model 1)
#   V_y_used  = V_hat_y + (lam*dV_true + (1-lam)*dV_hat)_y*dv_scale  (lam: label -> deployable)
#
# Grounding-only 80ep, both folds (matches the v2hy3 grnd80_slip02 baseline):
#   OBS_PHASE_PLAN='80,0,0,0,0' python observer_v1_py/train_observer_v2hy3_gammakin.py \
#     --regime observer_v1_py/regimes/S1_train_hy3.toml \
#     --scaler-csv ../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/variable_scaler_percentiles.csv \
#     --cache-dir C:/Users/vishv/mecanum_cache_decim --stride-frac 0.5 \
#     --w-dv 1.0 --w-slip 0.02 --no-grounding-early-stop --require-gpu
# =============================================================================
from __future__ import annotations

# Import pyarrow BEFORE torch (Windows native-loader crash otherwise).
import pyarrow.feather  # noqa: F401

import argparse
from pathlib import Path

from mecanum_observer.config_v2hy3_gammakin import ObserverConfigV2Hy3GammaKin
from mecanum_observer.data import load_regime, regime_to_kwargs
from mecanum_observer.training_v2hy3_gammakin import train_v2hy3_gammakin


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train the gamma-RESIDUAL (no-slip base) Observer v2-Hy3.")
    ap.add_argument("--data-dir", type=Path,
                    default=Path("../data/Simulation_Data_MecanumSlipSpin_LugreAdamov"))
    ap.add_argument("--whitelist", type=Path, default=Path("diagnostics_combined.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("observer_v1_py/runs"))
    ap.add_argument("--regime", type=Path, default=None)
    ap.add_argument("--model", choices=["ssm", "gru"], default="ssm")
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--stride-frac", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default=None)
    ap.add_argument("--cache-dir", type=str, default=None)
    ap.add_argument("--norm", choices=["max"], default="max")
    ap.add_argument("--scaler-csv", type=str, default=None)
    ap.add_argument("--w-dv", type=float, default=None)
    ap.add_argument("--w-slip", type=float, default=None)
    ap.add_argument("--gate-center", type=float, default=None)
    ap.add_argument("--gate-width", type=float, default=None)
    ap.add_argument("--mindlin-iters", type=int, default=None)
    ap.add_argument("--crossover-hz", type=float, default=None)
    ap.add_argument("--integrator", choices=["euler", "rot_ab2"], default=None)
    ap.add_argument("--noise-stage", choices=["none", "real"], default=None)
    ap.add_argument("--grounding-min-epochs", type=int, default=None)
    ap.add_argument("--grounding-patience", type=int, default=None)
    ap.add_argument("--no-grounding-early-stop", action="store_true")
    ap.add_argument("--no-lr-reset", action="store_true")
    ap.add_argument("--gradnorm", action="store_true")
    ap.add_argument("--cosine-diag", action="store_true",
                    help="log pairwise cos(grad) at the shared encoder on heartbeats")
    ap.add_argument("--cosine-diag-every", type=int, default=None)
    ap.add_argument("--cosine-diag-batches", type=int, default=None)
    # --- variant-specific ---
    ap.add_argument("--dgamma-scale", type=float, default=None,
                    help="residual scale (default gamma_p95/10 = 8.2806 rad/s)")
    ap.add_argument("--dv-scale", type=float, default=None,
                    help="UNIFORM dV scale for BOTH axes "
                         "(default mean(p95 Vx, p95 Vy) = 1.2967 m/s)")
    ap.add_argument("--vy-label-start", type=float, default=None,
                    help="lambda during grounding (1 = bootstrap on TRUE V_y label)")
    ap.add_argument("--vy-label-end", type=float, default=None,
                    help="lambda once physics engages (0 = deployable base)")
    ap.add_argument("--vy-label-ramp-epochs", type=int, default=None,
                    help="ramp lambda start->end linearly over this many GROUNDING "
                         "epochs, then hold at end (0 = constant). For the gamma-only "
                         "phase-out fine-tune.")
    ap.add_argument("--finetune-gamma-from", type=str, default=None,
                    help="warm-start the FULL model from this trained gammakin "
                         "checkpoint (e.g. .../checkpoint.pt), then fine-tune")
    ap.add_argument("--freeze-encoder-dv", action="store_true",
                    help="freeze encoder + feat + wheel_emb + head_dv; train ONLY "
                         "head_gamma (dv_hat becomes a fixed base for gamma_noslip)")
    ap.add_argument("--no-gamma-base-detach", action="store_true",
                    help="let gamma's gradient flow into dv_hat through the no-slip "
                         "base (only matters once lambda < 1)")
    ap.add_argument("--gamma-high-slip-upweight", type=float, default=None)
    ap.add_argument("--warm-from", type=str, default=None)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-tag", type=str, default=None)
    args = ap.parse_args()

    kw = dict(data_dir=args.data_dir, whitelist_csv=args.whitelist,
              out_dir=args.out_dir, model=args.model, epochs=args.epochs,
              lr=args.lr, limit_files=args.limit_files,
              require_gpu=args.require_gpu, norm_method="max")
    if args.regime is not None:
        kw.update(regime_to_kwargs(load_regime(args.regime)))
    for arg_name, field in [("batch_size", "batch_size"), ("jobs", "jobs"),
                            ("precision", "precision"), ("window", "window"),
                            ("stride_frac", "stride_frac"), ("cache_dir", "cache_dir"),
                            ("seed", "seed"), ("run_tag", "run_tag_override"),
                            ("scaler_csv", "scaler_csv"), ("w_dv", "w_dv"),
                            ("w_slip", "w_slip"), ("gate_center", "gate_center"),
                            ("gate_width", "gate_width"),
                            ("mindlin_iters", "mindlin_iters"),
                            ("crossover_hz", "vel_filter_crossover_hz"),
                            ("integrator", "vel_filter_integrator"),
                            ("noise_stage", "noise_stage"),
                            ("grounding_min_epochs", "grounding_min_epochs"),
                            ("grounding_patience", "grounding_patience"),
                            ("cosine_diag_every", "cosine_diag_every"),
                            ("cosine_diag_batches", "cosine_diag_batches"),
                            ("dgamma_scale", "dgamma_scale"),
                            ("vy_label_start", "vy_label_start"),
                            ("vy_label_end", "vy_label_end"),
                            ("vy_label_ramp_epochs", "vy_label_ramp_epochs"),
                            ("finetune_gamma_from", "finetune_gamma_from"),
                            ("gamma_high_slip_upweight", "gamma_high_slip_upweight"),
                            ("warm_from", "warm_from")]:
        v = getattr(args, arg_name)
        if v is not None:
            kw[field] = v
    if args.dv_scale is not None:                      # uniform on BOTH axes
        kw["dv_scale"] = (args.dv_scale, args.dv_scale)
    if args.gradnorm:
        kw["use_gradnorm"] = True
    if args.cosine_diag:
        kw["cosine_diag"] = True
    if args.no_grounding_early_stop:
        kw["grounding_early_stop"] = False
    if args.no_lr_reset:
        kw["lr_reset_at_physics"] = False
    if args.no_gamma_base_detach:
        kw["gamma_base_detach"] = False
    if args.freeze_encoder_dv:
        kw["freeze_encoder_dv"] = True

    cfg = ObserverConfigV2Hy3GammaKin(**kw)
    print(f"[cli-gammakin] model={cfg.model} window={cfg.window} stride={cfg.eff_stride} "
          f"batch={cfg.batch_size} epochs={cfg.total_epochs} w_dv={cfg.w_dv} "
          f"w_slip={cfg.w_slip} upweight={cfg.gamma_high_slip_upweight} "
          f"dgamma_scale={cfg.dgamma_scale:.4f} dv_scale={cfg.dv_scale} "
          f"lam={cfg.vy_label_start}->{cfg.vy_label_end} "
          f"detach={cfg.gamma_base_detach} cache={cfg.cache_dir or 'off'}")
    train_v2hy3_gammakin(cfg)


if __name__ == "__main__":
    main()
