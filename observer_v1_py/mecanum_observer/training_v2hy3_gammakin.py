#!/usr/bin/env python
# =============================================================================
# training_v2hy3_gammakin.py — 5-phase training for the gamma-RESIDUAL variant.
#
# SEPARATE MODULE ON PURPOSE. training_v2hy3.py is untouched so every v2hy3 run so
# far stays reproducible. Unchanged helpers (_phys_batch, _resolve_precision,
# _new_scheduler, _gradnorm_step, _cosine_conflict, ...) are IMPORTED from it rather
# than copied, so there is exactly one implementation of each.
#
# The only real difference: the gamma head predicts a RESIDUAL off the closed-form
# no-slip roller spin (nd711 sec 5.1 Model 1) instead of gamma itself.
#
#   gamma_noslip = -Vpy0_u / (cos delta * (R*cos theta_t - Rd))       [rad/s]
#   gamma_hat    = gamma_noslip(V_y_used) + dgamma_hat * dgamma_scale
#   V_y_used     = V_hat_y + (lam*dV_true + (1-lam)*dV_hat)_y * dv_scale
#
# lam ("vy_label") is the sim-to-real phase-out: 1.0 during grounding (bootstrap on
# the TRUE V_y label), ramping to 0.0 as physics engages (base built only from V_hat
# + the model's own dV_hat). Measured motivation: gamma_noslip alone predicts gamma
# to 0.285 rad/s RMS in STICK vs the v2hy3 head's 5.56 rad/s (~19x), and stick is
# where the |Vp| gate decides.
#
# Supervision moves to dgamma space:
#     dgamma_target = (gamma_true - gamma_noslip) / dgamma_scale
# With a detached base this is EXACTLY the gamma-space MSE rescaled by
# (gamma_p95/dgamma_scale)^2 = 100 -> the head's gradient is ~100x better scaled
# than it would be under gamma_p95 normalization. mse_gamma is ALSO reported in the
# old gamma_p95-normalized units so the logs stay comparable to prior runs.
# =============================================================================
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.feather  # noqa: F401
import torch
from torch.utils.data import DataLoader

from . import config as C
from . import data_v2hy3 as D2H
from . import physics_v2hy3 as P
from .config_v2 import W_SUP_MIN
from .config_v2hy3_gammakin import ObserverConfigV2Hy3GammaKin
from .losses_v2hy3 import (physics_loss_hy3, derived_contact_slip,
                           slip_consistency_loss)
from .models_v2hy3 import build_model_v2hy3, load_warm_start_v2hy3
# Unchanged machinery — imported, never re-implemented.
from .training_v2hy3 import (_cfg_dict, _phys_batch, _resolve_precision,
                             _new_scheduler, _gradnorm_step, _cosine_conflict,
                             _first_non_grounding, _ckpt_path, _best_ckpt_path)


def _phase_plan_gk(cfg: ObserverConfigV2Hy3GammaKin) -> List[Dict[str, object]]:
    """Per-epoch (phase, w_sup, w_phys, vy_label).

    Same schedule as training_v2hy3._phase_plan, plus the vy_label lambda: the V_y
    LABEL is ramped out exactly as physics ramps in, so the base is deployable
    (V_hat + dV_hat only) by the time the physics curriculum owns the objective.
    """
    plan: List[Dict[str, object]] = []
    total = cfg.total_epochs
    lam0, lam1 = cfg.vy_label_start, cfg.vy_label_end
    ramp = int(getattr(cfg, "vy_label_ramp_epochs", 0) or 0)
    consumed = 0
    for name, n in cfg.phase_plan:
        if consumed + n > total:
            n = total - consumed
        if n <= 0:
            break
        for e in range(n):
            frac = e / max(n - 1, 1)
            if name == "grounding":
                # Ramp lambda linearly start->end over `ramp` grounding epochs, then
                # hold at end (decoupled from the physics ramp). ge = global epoch.
                ge_ = consumed + e
                if ramp > 0:
                    lam = lam0 + (lam1 - lam0) * min(ge_ / ramp, 1.0)
                else:
                    lam = lam0
                w_sup, w_phys = 1.0, 0.0
            elif name == "phys_rampup":
                w_sup, w_phys, lam = 1.0, frac, lam0 + (lam1 - lam0) * frac
            elif name == "overlap":
                w_sup, w_phys, lam = 1.0, 1.0, lam1
            elif name == "grnd_rampdown":
                w_sup, w_phys, lam = 1.0 - (1.0 - W_SUP_MIN) * frac, 1.0, lam1
            elif name == "physics":
                w_sup, w_phys, lam = W_SUP_MIN, 1.0, lam1
            else:
                w_sup, w_phys, lam = 1.0, 0.0, lam0
            assert w_sup >= W_SUP_MIN - 1e-9
            # A NON-STATIONARY epoch: either lambda is mid-ramp (grounding) or a physics
            # ramp phase. The objective shifts each epoch, so a plateau LR scheduler / ES
            # / best-ckpt selection would misread the expected metric rise as "no
            # progress" and prematurely anneal or stop. They are FROZEN while ramping.
            lam_ramping = (ramp > 0 and name == "grounding" and (consumed + e) < ramp)
            phase_ramping = name in ("phys_rampup", "grnd_rampdown")
            plan.append(dict(phase=name, w_sup=float(w_sup), w_phys=float(w_phys),
                             vy_label=float(lam),
                             ramping=bool(lam_ramping or phase_ramping)))
        consumed += n
    return plan


def assemble_gamma(dgamma_hat: torch.Tensor, dv_hat: torch.Tensor,
                   y_dv: torch.Tensor, phys: Dict[str, torch.Tensor],
                   cfg: ObserverConfigV2Hy3GammaKin, dv_scale_t: torch.Tensor,
                   lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """(gamma_phys [B,4] rad/s, gamma_noslip [B,4] rad/s) from base + residual.

    lam = 1 -> base from the TRUE V_y label (no gradient path into dv_hat at all).
    lam < 1 -> base uses dv_hat; d(gamma_noslip)/dV_y ~ 97.5 rad/s per m/s, i.e. MORE
    leverage onto dV than the slip term's ~70:1 that was measured corrupting dV
    (4.2x at w_slip=0.1). cfg.gamma_base_detach cuts that path; it is moot while
    lam = 1 (grounding), which is why the policy is deferred to the phase-out.
    """
    dv_used = lam * y_dv + (1.0 - lam) * dv_hat            # normalized [B,2]
    Vpy0_u = phys["Vpy0_hat"] + (dv_used * dv_scale_t)[:, 1:2]
    g_ns = P.gamma_noslip(Vpy0_u, phys["cti"])
    if cfg.gamma_base_detach:
        g_ns = g_ns.detach()
    return g_ns + dgamma_hat * cfg.dgamma_scale, g_ns


def _gamma_terms(dgamma_hat, dv_hat, y_dv, y_gamma, phys, cfg, dv_scale_t,
                 gamma_std_t, gamma_mean_t, lam):
    """Shared assembly + dgamma-space residual. Returns
    (gamma_hat_norm, se_dgamma [B], se_gamma_gp95 [B], gamma_rmse_phys scalar)."""
    gamma_phys, g_ns = assemble_gamma(dgamma_hat, dv_hat, y_dv, phys, cfg,
                                      dv_scale_t, lam)
    gamma_true = y_gamma * gamma_std_t + gamma_mean_t          # [B,4] rad/s
    dgamma_target = (gamma_true - g_ns.detach()) / cfg.dgamma_scale
    se_dgamma = ((dgamma_hat - dgamma_target) ** 2).mean(dim=-1)
    err = gamma_phys - gamma_true
    # Same error in the OLD gamma_p95-normalized units, for log comparability.
    se_gp95 = ((err / gamma_std_t) ** 2).mean(dim=-1)
    gamma_hat_norm = (gamma_phys - gamma_mean_t) / gamma_std_t
    return gamma_hat_norm, se_dgamma, se_gp95, float(torch.sqrt((err ** 2).mean()).detach())


@torch.no_grad()
def supervised_val_metrics_gk(model, val_loader: DataLoader, device, amp_dtype,
                              cfg, gamma_std_t, gamma_mean_t, dv_scale_t,
                              vhat_scale_t, lam: float) -> Dict[str, float]:
    """Supervised-only val for the residual variant. `total` drives grounding
    early-stop / scheduler / best-ckpt, exactly as on the v2hy3 path."""
    model.eval()
    tot = g_sum = dv_sum = slip_sum = grms_sum = 0.0
    nb = 0
    for batch in val_loader:
        Gw = batch["Gw"].to(device); Pw = batch["Pw"].to(device)
        y_gamma = batch["y_gamma"].to(device); y_dv = batch["y_dv"].to(device)
        supw = batch["sup_weight"].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            dgamma_hat, dv_hat = model(Gw, Pw)
        dgamma_hat = dgamma_hat.float(); dv_hat = dv_hat.float()
        phys = _phys_batch(batch, device)
        gh_norm, se_dg, se_gp95, grms = _gamma_terms(
            dgamma_hat, dv_hat, y_dv, y_gamma, phys, cfg, dv_scale_t,
            gamma_std_t, gamma_mean_t, lam)
        l_gamma = (se_dg * supw).mean()
        se_dv = ((dv_hat - y_dv) ** 2).mean(dim=-1)
        total = l_gamma + cfg.w_dv * se_dv.mean()
        if cfg.w_slip > 0.0:
            V_hat = Gw[:, -1, :2] * vhat_scale_t
            v_slip = derived_contact_slip(gh_norm, dv_hat, V_hat, phys, cfg,
                                          gamma_std_t, gamma_mean_t)
            l_slip = slip_consistency_loss(v_slip, batch["slip_mag"].to(device),
                                           cfg.vpm_scale)
            slip_sum += float(l_slip)
            total = total + cfg.w_slip * l_slip
        tot += float(total); g_sum += float(se_gp95.mean())
        dv_sum += float(se_dv.mean()); grms_sum += grms
        nb += 1
    nb = max(nb, 1)
    return dict(total=tot / nb, gamma_mse=g_sum / nb, dv_mse=dv_sum / nb,
                slip_mse=slip_sum / nb, gamma_rmse_phys=grms_sum / nb)


@torch.no_grad()
def terminal_val_loss_gk(model, val_loader, device, amp_dtype, cfg,
                         gamma_std_t, gamma_mean_t, dv_scale_t, vhat_scale_t,
                         lam: float) -> float:
    """Validation at terminal phase weights (w_sup=0.1, w_phys=1)."""
    model.eval()
    tot, nb = 0.0, 0
    for batch in val_loader:
        Gw = batch["Gw"].to(device); Pw = batch["Pw"].to(device)
        y_gamma = batch["y_gamma"].to(device); y_dv = batch["y_dv"].to(device)
        supw = batch["sup_weight"].to(device)
        V_hat = Gw[:, -1, :2] * vhat_scale_t
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            dgamma_hat, dv_hat = model(Gw, Pw)
        dgamma_hat = dgamma_hat.float(); dv_hat = dv_hat.float()
        phys = _phys_batch(batch, device)
        gh_norm, se_dg, _, _ = _gamma_terms(dgamma_hat, dv_hat, y_dv, y_gamma, phys,
                                            cfg, dv_scale_t, gamma_std_t,
                                            gamma_mean_t, lam)
        l_sup = (se_dg * supw).mean() + cfg.w_dv * ((dv_hat - y_dv) ** 2).mean()
        # NORMALIZED gamma in, physics_loss_hy3 de-normalizes in-graph.
        l_phys, _ = physics_loss_hy3(gh_norm, dv_hat, V_hat, phys, cfg,
                                     gamma_std_t, gamma_mean_t)
        tot += float(W_SUP_MIN * l_sup + 1.0 * l_phys)
        nb += 1
    return tot / max(nb, 1)


def train_v2hy3_gammakin(cfg: ObserverConfigV2Hy3GammaKin) -> None:
    cfg = cfg.resolved()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)

    run_dir = Path(cfg.out_dir) / cfg.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        cap = torch.cuda.get_device_capability(0)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.manual_seed_all(cfg.seed)
        print(f"[train-gammakin] device=cuda:0 {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
    else:
        device = torch.device("cpu")
        if cfg.require_gpu:
            raise RuntimeError("CUDA not available but --require-gpu set.")
        print("[train-gammakin] WARNING: CUDA unavailable -> CPU (slow)")

    files = D2H.discover_and_split(cfg)
    print(f"[train-gammakin] {sum(len(v) for v in files.values())} files -> "
          f"train {len(files['train'])} val {len(files['val'])} test {len(files['test'])}")
    with open(run_dir / "split.json", "w") as fh:
        json.dump({k: [p.name for p in v] for k, v in files.items()}, fh, indent=0)

    nrm = D2H.load_max_scaler(cfg.scaler_csv)
    nrm.to_npz(run_dir / "norm.npz")

    gamma_std_t = torch.tensor(nrm.y_std[0], dtype=torch.float32, device=device).view(1, 1)
    gamma_mean_t = torch.tensor(nrm.y_mean[0], dtype=torch.float32, device=device).view(1, 1)
    dv_scale_t = torch.tensor(cfg.dv_scale, dtype=torch.float32, device=device).view(1, 2)
    vhat_scale_t = torch.tensor([nrm.g_std[0], nrm.g_std[1]],
                                dtype=torch.float32, device=device).view(1, 2)

    with open(run_dir / "LOSS_AND_NORM.md", "w", encoding="utf-8") as fh:
        fh.write(
            f"# Run: {cfg.run_tag}  (gamma-RESIDUAL variant)\n\n"
            f"- model={cfg.model} window={cfg.window} stride={cfg.eff_stride} regime={cfg.regime_name}\n"
            f"- **gamma parametrization:** residual off nd711 sec5.1 Model 1:\n"
            f"  `gamma_hat = gamma_noslip(V_y_used) + dgamma_hat * {cfg.dgamma_scale:.4f}`\n"
            f"  `gamma_noslip = -Vpy0_u / (cos_delta*(R*cos_tt - Rd))`  (V_Y and Omega only)\n"
            f"- **vy_label lambda:** {cfg.vy_label_start} -> {cfg.vy_label_end} "
            f"(TRUE V_y label ramps out as physics ramps in); base_detach={cfg.gamma_base_detach}\n"
            f"- **dv_scale:** UNIFORM {cfg.dv_scale} (isotropic; d|Vp|/ddV ~ 1 on both axes)\n"
            f"- **normalization:** MAX (frozen p95, 5 global) <- {cfg.scaler_csv}\n"
            f"- w_dv={cfg.w_dv} w_slip={cfg.w_slip} upweight={cfg.gamma_high_slip_upweight} "
            f"gate=({cfg.gate_center},{cfg.gate_width})\n"
            f"- AdamW lr={cfg.lr} wd={cfg.weight_decay}; ReduceLROnPlateau "
            f"factor={cfg.sched_factor} patience={cfg.sched_patience}\n")

    model = build_model_v2hy3(cfg).to(device)
    ckpt, best_ckpt = _ckpt_path(cfg), _best_ckpt_path(cfg)

    # gamma-only fine-tune of the phase-out: warm-start the FULL model from a trained
    # gammakin checkpoint, then freeze everything but head_gamma. Distinct from
    # warm_from (v1->v2 encoder transfer). Skipped on a genuine in-place resume.
    if cfg.finetune_gamma_from and not ckpt.exists():
        st = torch.load(cfg.finetune_gamma_from, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        print(f"[train-gammakin] fine-tune warm-start (FULL model) <- {cfg.finetune_gamma_from}")
    if cfg.freeze_encoder_dv:
        n_fz = 0
        for n, p in model.named_parameters():
            if not n.startswith("head_gamma"):
                p.requires_grad_(False); n_fz += 1
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[train-gammakin] FROZEN {n_fz} tensors (encoder+feat+wheel_emb+head_dv); "
              f"training head_gamma only ({n_tr:,} params). dv_hat is now a FIXED "
              f"function -> gamma_noslip base is static.")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _new_scheduler(opt, cfg)
    amp_dtype = _resolve_precision(cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    plan = _phase_plan_gk(cfg)

    shared_params = [p for n, p in model.named_parameters()
                     if p.requires_grad and not (n.startswith("head_gamma")
                                                 or n.startswith("head_dv"))]
    if cfg.cosine_diag:
        print(f"[train-gammakin] cosine-diag ON (every {cfg.cosine_diag_every} ep, "
              f"<={cfg.cosine_diag_batches} batches) over {len(shared_params)} shared tensors")
    gn_w = gn_opt = None
    gn_state: Dict[str, object] = {"L0": None}
    if cfg.use_gradnorm:
        gn_w = torch.nn.Parameter(torch.ones(2, device=device))
        gn_opt = torch.optim.Adam([gn_w], lr=cfg.gradnorm_lr)

    print(f"[train-gammakin] params={sum(p.numel() for p in model.parameters()):,} "
          f"epochs={len(plan)} batch={cfg.batch_size} dgamma_scale={cfg.dgamma_scale:.4f} "
          f"dv_scale={cfg.dv_scale[0]:.4f} lam {cfg.vy_label_start}->{cfg.vy_label_end}")

    start_epoch, best_metric = 0, float("inf")
    lr_trace: List[float] = []
    cosine_trace: List[Dict[str, float]] = []

    if cfg.warm_from and not cfg.finetune_gamma_from and not ckpt.exists():
        load_warm_start_v2hy3(model, cfg.warm_from)
    if ckpt.exists():
        stt = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(stt["model"]); opt.load_state_dict(stt["opt"])
        scheduler.load_state_dict(stt["scheduler"])
        start_epoch = stt["epoch"] + 1
        best_metric = stt.get("best_metric", float("inf"))
        lr_trace = list(stt.get("lr_trace", []))
        cosine_trace = list(stt.get("cosine_trace", []))
        print(f"[train-gammakin] resumed from epoch {start_epoch}/{len(plan)}")

    train_loader, val_loader = D2H.make_loaders(files, nrm, cfg)
    grnd_best, grnd_bad = float("inf"), 0
    lr_reset_done = (start_epoch < len(plan) and plan[start_epoch]["phase"] != "grounding")
    n_executed, val_term = 0, float("nan")

    ge = start_epoch
    while ge < len(plan):
        ph = plan[ge]
        lam = float(ph["vy_label"])

        if (cfg.lr_reset_at_physics and not lr_reset_done and ph["phase"] != "grounding"):
            for g in opt.param_groups:
                g["lr"] = cfg.lr
            scheduler = _new_scheduler(opt, cfg)
            best_metric = float("inf"); lr_reset_done = True
            print(f"[train-gammakin] LR reset -> {cfg.lr:.2e} entering '{ph['phase']}' (ep {ge})")

        model.train()
        agg_sup = agg_phys = agg_gamma = agg_dv = agg_slip = agg_grms = 0.0
        agg_wg = agg_wd = 0.0
        log_sum: Dict[str, float] = {}
        nb = 0
        diag_on = cfg.cosine_diag and (ge % max(cfg.cosine_diag_every, 1) == 0)
        diag_recs: List[Dict[str, float]] = []

        for batch in train_loader:
            Gw = batch["Gw"].to(device, non_blocking=True)
            Pw = batch["Pw"].to(device, non_blocking=True)
            y_gamma = batch["y_gamma"].to(device, non_blocking=True)
            y_dv = batch["y_dv"].to(device, non_blocking=True)
            supw = batch["sup_weight"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                dgamma_hat, dv_hat = model(Gw, Pw)
            dgamma_hat = dgamma_hat.float(); dv_hat = dv_hat.float()
            phys = _phys_batch(batch, device)

            gh_norm, se_dg, se_gp95, grms = _gamma_terms(
                dgamma_hat, dv_hat, y_dv, y_gamma, phys, cfg, dv_scale_t,
                gamma_std_t, gamma_mean_t, lam)
            L_gamma = (se_dg * supw).mean()
            se_dv = ((dv_hat - y_dv) ** 2).mean(dim=-1)
            L_dv = se_dv.mean()

            if cfg.use_gradnorm:
                wg, wd = _gradnorm_step(gn_state, L_gamma, L_dv, shared_params,
                                        gn_w, gn_opt, cfg.gradnorm_alpha)
            else:
                wg, wd = 1.0, cfg.w_dv
            l_sup = wg * L_gamma + wd * L_dv

            # Derived-slip term. slip is a deterministic fn of (gamma,dV), so we log
            # it even when w_slip=0 (to watch it fall out for free). CRITICAL: when
            # w_slip=0 compute it under no_grad — a grad-tracked l_slip that is never
            # backpropped retains its graph and blows peak VRAM (OOM at 2x batch-4096).
            V_hat = Gw[:, -1, :2] * vhat_scale_t
            slip_task = None
            if cfg.w_slip > 0.0:
                v_slip = derived_contact_slip(gh_norm, dv_hat, V_hat, phys, cfg,
                                              gamma_std_t, gamma_mean_t)
                l_slip = slip_consistency_loss(v_slip, batch["slip_mag"].to(device),
                                               cfg.vpm_scale)
                l_sup = l_sup + cfg.w_slip * l_slip
                agg_slip += float(l_slip.detach())
                slip_task = l_slip           # has grad -> usable as a cosine task
            else:
                with torch.no_grad():
                    v_slip = derived_contact_slip(gh_norm, dv_hat, V_hat, phys, cfg,
                                                  gamma_std_t, gamma_mean_t)
                    l_slip = slip_consistency_loss(v_slip, batch["slip_mag"].to(device),
                                                   cfg.vpm_scale)
                agg_slip += float(l_slip)

            if diag_on and nb < cfg.cosine_diag_batches:
                _tasks = {"gamma": L_gamma, "dv": L_dv}
                if slip_task is not None:     # only when slip has a grad graph
                    _tasks["slip"] = slip_task
                diag_recs.append(_cosine_conflict(_tasks, shared_params))

            loss = ph["w_sup"] * l_sup
            agg_gamma += float(se_gp95.mean().detach()); agg_dv += float(se_dv.mean().detach())
            agg_grms += grms; agg_wg += float(wg); agg_wd += float(wd)

            if ph["w_phys"] > 0.0 and cfg.physics_loss:
                V_hat = Gw[:, -1, :2] * vhat_scale_t
                l_phys, phys_log = physics_loss_hy3(gh_norm, dv_hat, V_hat, phys, cfg,
                                                    gamma_std_t, gamma_mean_t)
                loss = loss + ph["w_phys"] * l_phys
                agg_phys += float(l_phys.detach())
                for k, v in phys_log.items():
                    log_sum[k] = log_sum.get(k, 0.0) + v

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
            agg_sup += float(l_sup.detach()); nb += 1

        nb = max(nb, 1); n_executed += 1

        svm = sup_val = None
        if ph["phase"] == "grounding":
            svm = supervised_val_metrics_gk(model, val_loader, device, amp_dtype, cfg,
                                            gamma_std_t, gamma_mean_t, dv_scale_t,
                                            vhat_scale_t, lam)
            sup_val = svm["total"]; val_metric = val_term = sup_val
        else:
            val_term = terminal_val_loss_gk(model, val_loader, device, amp_dtype, cfg,
                                            gamma_std_t, gamma_mean_t, dv_scale_t,
                                            vhat_scale_t, lam)
            val_metric = val_term

        # FREEZE the plateau scheduler AND best-ckpt selection while ramping: the
        # objective is non-stationary, so the metric rise is expected, not a plateau.
        # (best-ckpt too, else "best" locks onto the pre-ramp lambda≈1 state.)
        ramping = bool(ph.get("ramping", False))
        if not ramping and ph["phase"] in ("grounding", "overlap", "physics"):
            scheduler.step(val_metric)
        current_lr = opt.param_groups[0]["lr"]
        lr_trace.append(float(current_lr))
        if not ramping and val_metric < best_metric:
            best_metric = val_metric
            torch.save(dict(model=model.state_dict(), epoch=ge, cfg=_cfg_dict(cfg),
                            best_metric=best_metric), best_ckpt)

        phys_summary = "phys n/a"
        if cfg.physics_loss and log_sum:
            stick = sum(log_sum.get(f"phys_stick_w{i}", 0.0) for i in range(1, 5)) / nb
            slip = sum(log_sum.get(f"phys_slip_w{i}", 0.0) for i in range(1, 5)) / nb
            phys_summary = (f"phys(stick {stick:.4f} slip {slip:.4f} "
                            f"g {log_sum.get('g_mean',0.)/nb:.4f} "
                            f"stick_frac {log_sum.get('g_stick_frac',0.)/nb:.3f})")
        sup_str = (f"g_mse {agg_gamma/nb:.5f} g_rmse {agg_grms/nb:.3f}rad/s "
                   f"dv_mse {agg_dv/nb:.5f}")
        sup_str += f" slip_rmse {((agg_slip/nb) ** 0.5) * cfg.vpm_scale:.4f}m/s"
        if cfg.w_slip == 0.0:
            sup_str += "(untrained)"
        grnd_str = ""
        if sup_val is not None:
            grnd_str = (f" | val sup {sup_val:.5f} g_mse {svm['gamma_mse']:.5f} "
                        f"g_rmse {svm['gamma_rmse_phys']:.3f}rad/s dv_mse {svm['dv_mse']:.5f}")
            if cfg.w_slip > 0.0:
                grnd_str += f" slip_rmse {(svm['slip_mse'] ** 0.5) * cfg.vpm_scale:.4f}m/s"

        if diag_on and diag_recs:
            keys = list(diag_recs[0].keys())
            rec: Dict[str, float] = dict(epoch=ge, phase=ph["phase"], vy_label=lam,
                                         n_batches=len(diag_recs),
                                         gamma_mse=agg_gamma / nb, dv_mse=agg_dv / nb)
            cos_parts, norm_parts = [], []
            for k in keys:
                v = np.asarray([d[k] for d in diag_recs])
                if k.startswith("cos_"):
                    rec[f"{k}_mean"] = float(v.mean()); rec[f"{k}_med"] = float(np.median(v))
                    rec[f"{k}_negfrac"] = float((v < 0).mean())
                    cos_parts.append(f"{k[4:]} {v.mean():+.4f}(neg {(v < 0).mean():.2f})")
                else:
                    rec[k] = float(v.mean()); norm_parts.append(f"|d{k[3:]}| {v.mean():.3e}")
            cosine_trace.append(rec)
            with open(run_dir / "cosine_diag.json", "w") as fh:
                json.dump(cosine_trace, fh, indent=1)
            grnd_str += (f"\n      [cos-diag ep{ge}] " + "  ".join(cos_parts) +
                         "  |  " + "  ".join(norm_parts))

        print(f"[{ge:3d}/{len(plan)} {ph['phase']:<13} ws{ph['w_sup']:.2f} "
              f"wp{ph['w_phys']:.2f} lam{lam:.2f} lr{current_lr:.2e}] "
              f"sup {agg_sup/nb:.5f} {sup_str} {phys_summary} "
              f"val_term {val_term:.5f} best {best_metric:.5f}{grnd_str}")

        if ge % 10 == 0 or ge == len(plan) - 1:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                            scheduler=scheduler.state_dict(), epoch=ge,
                            cfg=_cfg_dict(cfg), best_metric=best_metric,
                            lr_trace=lr_trace, cosine_trace=cosine_trace), ckpt)

        early_stop_now = False
        if (ph["phase"] == "grounding" and cfg.grounding_early_stop
                and sup_val is not None and not ramping):
            if sup_val < grnd_best * (1.0 - cfg.grounding_rel_tol):
                grnd_best, grnd_bad = sup_val, 0
            else:
                grnd_bad += 1
            if (ge + 1) >= cfg.grounding_min_epochs and grnd_bad >= cfg.grounding_patience:
                early_stop_now = True

        is_last = (ge == len(plan) - 1)
        next_phase = plan[ge + 1]["phase"] if not is_last else None
        if is_last or next_phase != ph["phase"] or early_stop_now:
            snap = run_dir / "phase_ckpts"; snap.mkdir(exist_ok=True)
            torch.save(dict(model=model.state_dict(), epoch=ge, phase=ph["phase"],
                            w_sup=ph["w_sup"], w_phys=ph["w_phys"], vy_label=lam,
                            cfg=_cfg_dict(cfg)), snap / f"{ph['phase']}_ep{ge:03d}.pt")
        if early_stop_now:
            nxt = _first_non_grounding(plan)
            print(f"[train-gammakin] grounding EARLY-STOP at ep {ge} (sup_val {sup_val:.5f})")
            ge = nxt
            continue
        ge += 1

    print(f"[train-gammakin] done ({n_executed} epochs) -> {ckpt}")
    metrics = dict(
        val_loss=float(val_term), epochs=n_executed, epochs_planned=len(plan),
        model="ssm_v2hy3_gammakin", window=cfg.window, stride=cfg.eff_stride,
        regime=cfg.regime_name, gamma_parametrization="residual_noslip_model1",
        dgamma_scale=cfg.dgamma_scale, dv_scale=list(cfg.dv_scale),
        vy_label_start=cfg.vy_label_start, vy_label_end=cfg.vy_label_end,
        vy_label_ramp_epochs=cfg.vy_label_ramp_epochs,
        finetune_gamma_from=cfg.finetune_gamma_from,
        freeze_encoder_dv=cfg.freeze_encoder_dv,
        gamma_base_detach=cfg.gamma_base_detach,
        w_dv=cfg.w_dv, w_slip=cfg.w_slip,
        gamma_high_slip_upweight=cfg.gamma_high_slip_upweight,
        physics_loss=cfg.physics_loss, norm_method=cfg.norm_method,
        lr_trace=lr_trace, params=sum(p.numel() for p in model.parameters()),
        cfg=_cfg_dict(cfg))
    with open(run_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=0)
    print(f"[train-gammakin] metrics -> {run_dir / 'metrics.json'}")
