#!/usr/bin/env python
# =============================================================================
# training_v2.py — 5-phase γ-only observer training with physics-regularised
# refinement.
#
# Key differences from v1:
#   * single γ target and single γ head
#   * 200-epoch schedule (80/24/40/24/32)
#   * supervised weight floor W_SUP_MIN = 0.1 (never zero)
#   * ONE ReduceLROnPlateau scheduler across all phases (no per-phase lr_scale)
#   * plateau counter frozen during phys_rampup / grnd_rampdown
#   * validation metric is terminal_val_loss (phase-invariant)
#   * promoted roller-residual term is trained
# =============================================================================
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.feather  # noqa: F401  (before torch)
import torch
from torch import _dynamo  # noqa: F401
from torch.utils.data import DataLoader

from . import config as C
from . import data_v2 as D2
from .config_v2 import ObserverConfigV2, W_SUP_MIN
from .losses_v2 import physics_loss_v2, supervised_gamma_loss
from .models_v2 import build_model_v2, load_warm_start_v2


def _cfg_dict(cfg: ObserverConfigV2) -> Dict[str, object]:
    """Return config as a JSON/pickle-safe dict with Path objects as strings.

    Avoids pickling pathlib.Path, whose internal module path changed in
    Python 3.13 and breaks cross-version checkpoint loading."""
    d = asdict(cfg)
    for k, v in d.items():
        if isinstance(v, Path):
            d[k] = str(v)
    return d


def _phase_plan(cfg: ObserverConfigV2) -> List[Dict[str, object]]:
    """Build per-epoch plan with (phase, w_sup, w_phys).  LR is scheduler-owned."""
    plan: List[Dict[str, object]] = []
    total = cfg.total_epochs
    consumed = 0
    for name, n in cfg.phase_plan:
        if consumed + n > total:
            n = total - consumed
        if n <= 0:
            break
        for e in range(n):
            frac = e / max(n - 1, 1)
            if name == "grounding":
                w_sup, w_phys = 1.0, 0.0
            elif name == "phys_rampup":
                w_sup, w_phys = 1.0, frac
            elif name == "overlap":
                w_sup, w_phys = 1.0, 1.0
            elif name == "grnd_rampdown":
                w_sup, w_phys = 1.0 - (1.0 - W_SUP_MIN) * frac, 1.0
            elif name == "physics":
                w_sup, w_phys = W_SUP_MIN, 1.0
            else:  # pragma: no cover
                w_sup, w_phys = 1.0, 0.0
            assert w_sup >= W_SUP_MIN - 1e-9, f"w_sup {w_sup} below floor {W_SUP_MIN}"
            plan.append(dict(phase=name, w_sup=float(w_sup), w_phys=float(w_phys)))
        consumed += n
    return plan


# Ramp phases where the plateau counter is frozen.
_RAMP_PHASES = {"phys_rampup", "grnd_rampdown"}
# Constant-objective phases where scheduler.step() may fire.
_STEP_PHASES = {"grounding", "overlap", "physics"}


def _resolve_precision(cfg: ObserverConfigV2, device: torch.device) -> Optional[torch.dtype]:
    """Same logic as v1: bf16 only on Ampere+; otherwise fp32."""
    if device.type != "cuda":
        return None
    p = cfg.precision
    if p == "fp32":
        return None
    if p == "bf16":
        return torch.bfloat16
    if p == "fp16":
        return torch.float16
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else None


def _ckpt_path(cfg: ObserverConfigV2) -> Path:
    return Path(cfg.out_dir) / cfg.run_tag / "checkpoint.pt"


def _best_ckpt_path(cfg: ObserverConfigV2) -> Path:
    return Path(cfg.out_dir) / cfg.run_tag / "checkpoint_best.pt"


def _phys_batch(batch: Dict[str, torch.Tensor], device: torch.device
                ) -> Dict[str, torch.Tensor]:
    return {
        "psi_dot": batch["ph_psi_dot"].to(device),
        "Vpx0": batch["ph_Vpx0"].to(device),
        "Vpy0": batch["ph_Vpy0"].to(device),
        "cti": batch["ph_cti"].to(device),
        "sti": batch["ph_sti"].to(device),
        "Msat": batch["ph_Msat"].to(device),
        "w": batch["ph_w"].to(device),
        "w_dot": batch["ph_w_dot"].to(device),
        "mu": batch["ph_mu"].to(device),
        "chi": batch["ph_chi"].to(device),
        "Vx": batch["ph_Vx"].to(device),
        "Vy": batch["ph_Vy"].to(device),
        "dVx": batch["ph_dVx"].to(device),
        "dVy": batch["ph_dVy"].to(device),
        "dpsi_dot": batch["ph_dpsi_dot"].to(device),
        "Vx_next": batch["ph_Vx_next"].to(device),
        "Vy_next": batch["ph_Vy_next"].to(device),
        "psi_dot_next": batch["ph_psi_dot_next"].to(device),
        "w_next": batch["ph_w_next"].to(device),
        "zx_lab": batch["zx_lab"].to(device),
        "zy_lab": batch["zy_lab"].to(device),
        "slip_mag": batch["slip_mag"].to(device),
    }


@torch.no_grad()
def terminal_val_loss(model, val_loader: DataLoader, device: torch.device,
                      amp_dtype: Optional[torch.dtype], cfg: ObserverConfigV2,
                      gamma_std_t: torch.Tensor, gamma_mean_t: torch.Tensor,
                      Minv: torch.Tensor) -> float:
    """Validation loss at TERMINAL phase weights (w_sup=0.1, w_phys=1, incl. roller).

    This phase-invariant metric is fed to ReduceLROnPlateau and the best-checkpoint
    selector so ramps cannot fake an improvement.
    """
    model.eval()
    tot, nb = 0.0, 0
    for batch in val_loader:
        Gw = batch["Gw"].to(device)
        Pw = batch["Pw"].to(device)
        y = batch["y"].to(device)
        supw = batch["sup_weight"].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            gamma_hat = model(Gw, Pw).float()
            l_sup, _ = supervised_gamma_loss(gamma_hat, y, supw)
            gamma_hat_phys = gamma_hat * gamma_std_t + gamma_mean_t
            # Physics recompute always in fp32 (autocast-exempt).
            l_phys, _ = physics_loss_v2(
                gamma_hat_phys, _phys_batch(batch, device),
                variant=cfg.physics_variant, w_roller=cfg.w_roller,
                roller_slip_weighting=cfg.roller_slip_weighting, Minv=Minv)
            loss = W_SUP_MIN * l_sup + 1.0 * l_phys
        tot += float(loss)
        nb += 1
    return tot / max(nb, 1)


def train_v2(cfg: ObserverConfigV2) -> None:
    cfg = cfg.resolved()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    run_dir = Path(cfg.out_dir) / cfg.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        cap = torch.cuda.get_device_capability(0)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.manual_seed_all(cfg.seed)
        print(f"[train-v2] device=cuda:0 {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
    else:
        device = torch.device("cpu")
        if cfg.require_gpu:
            raise RuntimeError("CUDA not available but --require-gpu set.")
        print("[train-v2] WARNING: CUDA unavailable -> training on CPU (slow)")

    files = D2.discover_and_split(cfg)
    print(f"[train-v2] {sum(len(v) for v in files.values())} files -> "
          f"train {len(files['train'])} val {len(files['val'])} test {len(files['test'])}")
    with open(run_dir / "split.json", "w") as fh:
        json.dump({k: [p.name for p in v] for k, v in files.items()}, fh, indent=0)

    nrm_path = run_dir / "norm.npz"
    nrm = D2.load_max_scaler(cfg.scaler_csv)
    nrm.to_npz(nrm_path)
    print(f"[train-v2] MAX-norm (frozen p95) <- {cfg.scaler_csv}")

    # Gamma de-normalisation tensors (index 0 in v1 y-scale corresponds to gamma).
    gamma_std_t = torch.tensor(nrm.y_std[0], dtype=torch.float32, device=device).view(1, 1)
    gamma_mean_t = torch.tensor(nrm.y_mean[0], dtype=torch.float32, device=device).view(1, 1)

    with open(run_dir / "LOSS_AND_NORM.md", "w", encoding="utf-8") as fh:
        fh.write(
            f"# Run: {cfg.run_tag}\n\n"
            f"- model={cfg.model} window={cfg.window} stride={cfg.eff_stride} regime={cfg.regime_name}\n"
            f"- **normalization:** MAX (frozen p95) <- {cfg.scaler_csv}\n"
            f"- **loss:** γ-only supervised + physics (5-phase ramp, W_SUP_MIN={W_SUP_MIN})\n"
            f"- physics_variant={cfg.physics_variant} w_roller={cfg.w_roller} "
            f"roller_slip_weighting={cfg.roller_slip_weighting}\n"
            f"- AdamW lr={cfg.lr} wd={cfg.weight_decay} grad_clip={cfg.grad_clip}; "
            f"ReduceLROnPlateau factor={cfg.sched_factor} patience={cfg.sched_patience} "
            f"min_lr={cfg.sched_min_lr}\n"
            f"- warm_from={cfg.warm_from or 'none'}\n")

    model = build_model_v2(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg.sched_factor, patience=cfg.sched_patience,
        threshold=cfg.sched_rel_threshold, min_lr=cfg.sched_min_lr, verbose=False)

    Minv = torch.tensor(C.M_BODY_INV, dtype=torch.float32, device=device)
    amp_dtype = _resolve_precision(cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    plan = _phase_plan(cfg)

    print(f"[train-v2] model={cfg.model} params={sum(p.numel() for p in model.parameters()):,} "
          f"epochs={len(plan)} batch={cfg.batch_size} physics_variant={cfg.physics_variant} "
          f"warm_from={cfg.warm_from or 'none'}")

    start_epoch = 0
    ckpt = _ckpt_path(cfg)
    best_ckpt = _best_ckpt_path(cfg)
    best_metric = float("inf")
    lr_trace: List[float] = []

    # Warm-start from v1 checkpoint before normal resume.
    if cfg.warm_from and not ckpt.exists():
        load_warm_start_v2(model, cfg.warm_from)
        print(f"[train-v2] warm-started encoder from {cfg.warm_from} (γ head fresh)")

    if ckpt.exists():
        stt = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(stt["model"])
        opt.load_state_dict(stt["opt"])
        scheduler.load_state_dict(stt["scheduler"])
        start_epoch = stt["epoch"] + 1
        best_metric = stt.get("best_metric", float("inf"))
        lr_trace = list(stt.get("lr_trace", []))
        print(f"[train-v2] resumed from epoch {start_epoch}/{len(plan)}")

    train_loader, val_loader = D2.make_loaders(files, nrm, cfg)

    for ge in range(start_epoch, len(plan)):
        ph = plan[ge]
        model.train()
        agg_sup = 0.0
        agg_phys = 0.0
        log_sum: Dict[str, float] = {}
        nb = 0

        for batch in train_loader:
            Gw = batch["Gw"].to(device, non_blocking=True)
            Pw = batch["Pw"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            supw = batch["sup_weight"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                gamma_hat = model(Gw, Pw)
            gamma_hat = gamma_hat.float()
            l_sup, _ = supervised_gamma_loss(gamma_hat, y, supw)
            loss = ph["w_sup"] * l_sup

            if ph["w_phys"] > 0.0 and cfg.physics_loss:
                gamma_hat_phys = gamma_hat * gamma_std_t + gamma_mean_t
                l_phys, phys_log = physics_loss_v2(
                    gamma_hat_phys, _phys_batch(batch, device),
                    variant=cfg.physics_variant, w_roller=cfg.w_roller,
                    roller_slip_weighting=cfg.roller_slip_weighting, Minv=Minv)
                loss = loss + ph["w_phys"] * l_phys
                agg_phys += float(l_phys.detach())
                for k, v in phys_log.items():
                    log_sum[k] = log_sum.get(k, 0.0) + v

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            agg_sup += float(l_sup.detach())
            nb += 1

        nb = max(nb, 1)

        # Terminal validation metric (phase-invariant).
        val_term = terminal_val_loss(model, val_loader, device, amp_dtype, cfg,
                                     gamma_std_t, gamma_mean_t, Minv)

        # Scheduler: step only in constant-objective phases; freeze counter in ramps.
        if ph["phase"] in _STEP_PHASES:
            scheduler.step(val_term)
        current_lr = opt.param_groups[0]["lr"]
        lr_trace.append(float(current_lr))

        # Best-checkpoint selection uses the same terminal metric.
        if val_term < best_metric:
            best_metric = val_term
            torch.save(dict(model=model.state_dict(), epoch=ge, cfg=_cfg_dict(cfg),
                            best_metric=best_metric),
                       best_ckpt)

        # Build a concise physics summary for the log line.
        phys_summary = ""
        if cfg.physics_loss and log_sum:
            if cfg.physics_variant == "residual":
                wheel = sum(log_sum.get(f"phys_wheel_w{i}", 0.0) for i in range(1, 5)) / nb
                body_x = log_sum.get("phys_body_x", 0.0) / nb
                body_y = log_sum.get("phys_body_y", 0.0) / nb
                body_yaw = log_sum.get("phys_body_yaw", 0.0) / nb
                roller = sum(log_sum.get(f"phys_roller_w{i}", 0.0) for i in range(1, 5)) / nb
                phys_summary = (f"phys(w {wheel:.4f} bx {body_x:.4f} by {body_y:.4f} "
                                f"byaw {body_yaw:.4f} roll {roller:.4f})")
            else:  # integrated
                int_Vx = log_sum.get("phys_int_Vx", 0.0) / nb
                int_Vy = log_sum.get("phys_int_Vy", 0.0) / nb
                int_pd = log_sum.get("phys_int_psidot", 0.0) / nb
                int_w = sum(log_sum.get(f"phys_int_w{i}", 0.0) for i in range(1, 5)) / nb
                roller = sum(log_sum.get(f"phys_roller_w{i}", 0.0) for i in range(1, 5)) / nb
                phys_summary = (f"phys(Vx {int_Vx:.4f} Vy {int_Vy:.4f} pd {int_pd:.4f} "
                                f"w {int_w:.4f} roll {roller:.4f})")
        else:
            phys_summary = "phys n/a"

        print(f"[{ge:3d}/{len(plan)} {ph['phase']:<13} "
              f"ws{ph['w_sup']:.2f} wp{ph['w_phys']:.2f} lr{current_lr:.2e}] "
              f"sup {agg_sup/nb:.5f} {phys_summary} "
              f"val_term {val_term:.5f} best {best_metric:.5f}")

        # Latest checkpoint (resume).
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        scheduler=scheduler.state_dict(), epoch=ge,
                        cfg=_cfg_dict(cfg), best_metric=best_metric,
                        lr_trace=lr_trace),
                   ckpt)

        # Per-phase snapshot.
        is_last = (ge == len(plan) - 1)
        next_phase = plan[ge + 1]["phase"] if not is_last else None
        if is_last or next_phase != ph["phase"]:
            snap_dir = run_dir / "phase_ckpts"
            snap_dir.mkdir(exist_ok=True)
            torch.save(dict(model=model.state_dict(), epoch=ge, phase=ph["phase"],
                            w_sup=ph["w_sup"], w_phys=ph["w_phys"],
                            cfg=_cfg_dict(cfg)),
                       snap_dir / f"{ph['phase']}_ep{ge:03d}.pt")

    print(f"[train-v2] done -> {ckpt}")

    # Final metrics.json
    metrics = dict(
        val_loss=float(val_term),
        val_state_loss=float(val_term),
        epochs=len(plan),
        model="ssm_v2_gamma" if cfg.model == "ssm" else "gru_v2_gamma",
        window=cfg.window,
        stride=cfg.eff_stride,
        regime=cfg.regime_name,
        chi_fold_test=cfg.chi_fold_test,
        physics_loss=cfg.physics_loss,
        physics_variant=cfg.physics_variant,
        norm_method=cfg.norm_method,
        w_roller=cfg.w_roller,
        roller_slip_weighting=cfg.roller_slip_weighting,
        sched_factor=cfg.sched_factor,
        sched_patience=cfg.sched_patience,
        sched_min_lr=cfg.sched_min_lr,
        lr_trace=lr_trace,
        warm_from=cfg.warm_from,
        params=sum(p.numel() for p in model.parameters()),
        cfg=_cfg_dict(cfg),
    )
    with open(run_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=0)
    print(f"[train-v2] metrics -> {run_dir / 'metrics.json'}")
