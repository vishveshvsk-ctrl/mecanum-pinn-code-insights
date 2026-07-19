#!/usr/bin/env python
# =============================================================================
# training_v2hy3.py — 5-phase γ + ΔV observer training with regime-split physics.
#
# Inherits from training_v2.py:
#   * 200-epoch schedule (80/24/40/24/32)
#   * supervised weight floor W_SUP_MIN = 0.1
#   * ONE ReduceLROnPlateau scheduler across all phases
#   * plateau counter frozen during phys_rampup / grnd_rampdown
#   * terminal_val_loss is phase-invariant
#
# Hy3-specific changes:
#   * model outputs (γ̂, ΔV̂)
#   * supervised loss spans both heads
#   * physics loss is the regime-split steady-state loss
#   * terminal metric includes the ΔV supervised term
# =============================================================================
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyarrow.feather  # noqa: F401
import torch
from torch.utils.data import DataLoader

from . import config as C
from . import data_v2hy3 as D2H
from .config_v2 import GAMMA_P95_DEFAULT, W_SUP_MIN
from .config_v2hy3 import ObserverConfigV2Hy3
from .losses_v2hy3 import (physics_loss_hy3, supervised_loss,
                           derived_contact_slip, slip_consistency_loss)
from .models_v2hy3 import build_model_v2hy3, load_warm_start_v2hy3


def _cfg_dict(cfg: ObserverConfigV2Hy3) -> Dict[str, object]:
    """JSON/pickle-safe dict; Path -> str."""
    d = asdict(cfg)
    for k, v in d.items():
        if isinstance(v, Path):
            d[k] = str(v)
    return d


def _phase_plan(cfg: ObserverConfigV2Hy3) -> List[Dict[str, object]]:
    """Per-epoch (phase, w_sup, w_phys).  LR owned by ReduceLROnPlateau."""
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
            else:
                w_sup, w_phys = 1.0, 0.0
            assert w_sup >= W_SUP_MIN - 1e-9
            plan.append(dict(phase=name, w_sup=float(w_sup), w_phys=float(w_phys)))
        consumed += n
    return plan


_RAMP_PHASES = {"phys_rampup", "grnd_rampdown"}
_STEP_PHASES = {"grounding", "overlap", "physics"}


def _resolve_precision(cfg: ObserverConfigV2Hy3, device: torch.device) -> Optional[torch.dtype]:
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


def _ckpt_path(cfg: ObserverConfigV2Hy3) -> Path:
    return Path(cfg.out_dir) / cfg.run_tag / "checkpoint.pt"


def _best_ckpt_path(cfg: ObserverConfigV2Hy3) -> Path:
    return Path(cfg.out_dir) / cfg.run_tag / "checkpoint_best.pt"


def _phys_batch(batch: Dict[str, torch.Tensor], device: torch.device
                ) -> Dict[str, torch.Tensor]:
    return {
        "Vpx0_hat": batch["ph_Vpx0"].to(device),
        "Vpy0_hat": batch["ph_Vpy0"].to(device),
        "cti": batch["ph_cti"].to(device),
        "sti": batch["ph_sti"].to(device),
        "psi_dot": batch["ph_psi_dot"].to(device),
        "mu": batch["ph_mu"].to(device),
        "chi": batch["ph_chi"].to(device),
        "w": batch["ph_w"].to(device),
    }


@torch.no_grad()
def terminal_val_loss(model, val_loader: DataLoader, device: torch.device,
                      amp_dtype: Optional[torch.dtype], cfg: ObserverConfigV2Hy3,
                      gamma_std_t: torch.Tensor, gamma_mean_t: torch.Tensor,
                      dv_scale_t: torch.Tensor) -> float:
    """Validation loss at terminal phase weights (w_sup=0.1, w_phys=1)."""
    model.eval()
    tot, nb = 0.0, 0
    for batch in val_loader:
        Gw = batch["Gw"].to(device)
        Pw = batch["Pw"].to(device)
        y_gamma = batch["y_gamma"].to(device)
        y_dv = batch["y_dv"].to(device)
        supw = batch["sup_weight"].to(device)
        V_hat = Gw[:, -1, :2] * dv_scale_t  # de-normalize V̂ from Gw

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            gamma_hat, dv_hat = model(Gw, Pw)
            gamma_hat = gamma_hat.float()
            dv_hat = dv_hat.float()
            l_sup, _ = supervised_loss(gamma_hat, dv_hat, y_gamma, y_dv, supw, cfg.w_dv)
            # physics_loss_hy3 takes NORMALIZED gamma and de-normalizes in-graph;
            # passing an already-de-normalized gamma double-scales it by gamma_p95.
            l_phys, _ = physics_loss_hy3(
                gamma_hat, dv_hat, V_hat,
                _phys_batch(batch, device), cfg, gamma_std_t, gamma_mean_t)
            loss = W_SUP_MIN * l_sup + 1.0 * l_phys
        tot += float(loss)
        nb += 1
    return tot / max(nb, 1)


@torch.no_grad()
def supervised_val_metrics(model, val_loader: DataLoader, device: torch.device,
                          amp_dtype: Optional[torch.dtype], cfg: ObserverConfigV2Hy3,
                          gamma_std_t: torch.Tensor, gamma_mean_t: torch.Tensor,
                          vhat_scale_t: torch.Tensor) -> Dict[str, float]:
    """Supervised-ONLY val metrics; no physics term. Returns a breakdown so we can
    watch the γ / ΔV 'measured labels' separately from the derived-slip term.
    `total` = MSE(gamma;sup_w) + w_dv*MSE(dV) + w_slip*slip_mse — drives grounding
    early-stop (the terminal metric is physics-dominated and flat during grounding)."""
    model.eval()
    tot = g_sum = dv_sum = slip_sum = 0.0
    nb = 0
    for batch in val_loader:
        Gw = batch["Gw"].to(device)
        Pw = batch["Pw"].to(device)
        y_gamma = batch["y_gamma"].to(device)
        y_dv = batch["y_dv"].to(device)
        supw = batch["sup_weight"].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            gamma_hat, dv_hat = model(Gw, Pw)
        gamma_hat = gamma_hat.float(); dv_hat = dv_hat.float()
        l_sup, sup_log = supervised_loss(gamma_hat, dv_hat, y_gamma, y_dv, supw, cfg.w_dv)
        g_sum += sup_log["mse_gamma"]; dv_sum += sup_log["mse_dv"]
        total = l_sup
        if cfg.w_slip > 0.0:
            V_hat = Gw[:, -1, :2] * vhat_scale_t
            v_slip = derived_contact_slip(gamma_hat, dv_hat, V_hat,
                                          _phys_batch(batch, device), cfg,
                                          gamma_std_t, gamma_mean_t)
            l_slip = slip_consistency_loss(v_slip, batch["slip_mag"].to(device), cfg.vpm_scale)
            slip_sum += float(l_slip)
            total = total + cfg.w_slip * l_slip
        tot += float(total)
        nb += 1
    nb = max(nb, 1)
    return dict(total=tot / nb, gamma_mse=g_sum / nb, dv_mse=dv_sum / nb,
                slip_mse=slip_sum / nb)


def _new_scheduler(opt, cfg: ObserverConfigV2Hy3):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg.sched_factor, patience=cfg.sched_patience,
        threshold=cfg.sched_rel_threshold, min_lr=cfg.sched_min_lr)


def _gradnorm_step(state: Dict[str, object], L_gamma: torch.Tensor, L_dv: torch.Tensor,
                   shared_params: List[torch.Tensor], gn_w: torch.Tensor,
                   gn_opt, alpha: float):
    """One GradNorm update of the 2 target weights (gamma, dV). Returns detached
    (w_gamma, w_dv). The weights are trained so each task's weighted grad-norm at
    the shared layer matches Ḡ·r^alpha (r = relative inverse training rate);
    slower tasks get up-weighted. gn (per-task grad norm) is detached, so gn_w is
    updated purely toward the gradnorm target — it cannot be gamed by shrinking a
    weight to lower the task loss."""
    gns = []
    for Li in (L_gamma, L_dv):
        grads = torch.autograd.grad(Li, shared_params, retain_graph=True,
                                    create_graph=False, allow_unused=True)
        sq = sum((g.detach() ** 2).sum() for g in grads if g is not None)
        gns.append(torch.sqrt(sq + 1e-12))
    gn = torch.stack(gns)                                 # [2], detached norms
    Lvec = torch.stack([L_gamma.detach(), L_dv.detach()])
    if state["L0"] is None:
        state["L0"] = Lvec.clamp_min(1e-8)
    G = gn_w * gn                                          # weighted grad norms (grad wrt gn_w)
    Gbar = G.mean().detach()
    ratio = Lvec / state["L0"]
    r = ratio / ratio.mean().clamp_min(1e-8)
    target = (Gbar * r ** alpha).detach()
    L_grad = (G - target).abs().sum()
    gn_opt.zero_grad()
    L_grad.backward()
    gn_opt.step()
    with torch.no_grad():
        gn_w.clamp_(min=1e-3)
        gn_w.mul_(2.0 / gn_w.sum())                       # renormalize sum -> 2
    return gn_w[0].detach(), gn_w[1].detach()


def _cosine_conflict(tasks: Dict[str, torch.Tensor],
                     shared_params: List[torch.Tensor]) -> Dict[str, float]:
    """PAIRWISE gradient-DIRECTION conflict between task losses at the shared trunk.

    Returns cos(grad L_a, grad L_b) for every pair over the flattened shared-parameter
    gradient, plus each task's RAW (unweighted) grad norm. Cosine is scale-invariant,
    so the loss weights cannot affect it — it measures representation tension only:
        cos < 0  -> tasks pull the shared encoder opposite ways -> split/adapter helps
        cos >= 0 -> trunk serves both; imbalance is a weighting problem, not geometry
    The norms are what GradNorm equalizes, reported alongside so the two effects can be
    read apart. With w_slip>0 the slip term joins as a third task: it is a function of
    BOTH heads, so cos(slip,gamma) / cos(slip,dv) reveal which head it actually pulls
    (the ~70:1 dVp/ddV vs dVp/dgamma leverage predicts it loads onto dV).
    Called under retain_graph so the main backward still owns the graph."""
    def _flat(Li: torch.Tensor) -> torch.Tensor:
        gs = torch.autograd.grad(Li, shared_params, retain_graph=True,
                                 create_graph=False, allow_unused=True)
        return torch.cat([g.detach().reshape(-1) for g in gs if g is not None])

    flat = {k: _flat(v) for k, v in tasks.items()}
    out: Dict[str, float] = {f"gn_{k}": float(v.norm()) for k, v in flat.items()}
    keys = list(flat)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            na, nb = flat[a].norm(), flat[b].norm()
            out[f"cos_{a}_{b}"] = float((flat[a] @ flat[b]) / (na * nb + 1e-12))
    return out


def _first_non_grounding(plan: List[Dict[str, object]]) -> int:
    for i, ph in enumerate(plan):
        if ph["phase"] != "grounding":
            return i
    return len(plan)


def train_v2hy3(cfg: ObserverConfigV2Hy3) -> None:
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
        print(f"[train-v2hy3] device=cuda:0 {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
    else:
        device = torch.device("cpu")
        if cfg.require_gpu:
            raise RuntimeError("CUDA not available but --require-gpu set.")
        print("[train-v2hy3] WARNING: CUDA unavailable -> training on CPU (slow)")

    files = D2H.discover_and_split(cfg)
    print(f"[train-v2hy3] {sum(len(v) for v in files.values())} files -> "
          f"train {len(files['train'])} val {len(files['val'])} test {len(files['test'])}")
    with open(run_dir / "split.json", "w") as fh:
        json.dump({k: [p.name for p in v] for k, v in files.items()}, fh, indent=0)

    nrm_path = run_dir / "norm.npz"
    nrm = D2H.load_max_scaler(cfg.scaler_csv)
    nrm.to_npz(nrm_path)
    print(f"[train-v2hy3] MAX-norm (frozen p95, 5 global) <- {cfg.scaler_csv}")

    # De-normalization tensors.
    gamma_std_t = torch.tensor(nrm.y_std[0], dtype=torch.float32, device=device).view(1, 1)
    gamma_mean_t = torch.tensor(nrm.y_mean[0], dtype=torch.float32, device=device).view(1, 1)
    dv_scale_t = torch.tensor(cfg.dv_scale, dtype=torch.float32, device=device).view(1, 2)
    # V̂ is the first two channels of Gw; their p95 scales are the Vx/Vy entries.
    vhat_scale_t = torch.tensor([nrm.g_std[0], nrm.g_std[1]],
                                dtype=torch.float32, device=device).view(1, 2)

    with open(run_dir / "LOSS_AND_NORM.md", "w", encoding="utf-8") as fh:
        fh.write(
            f"# Run: {cfg.run_tag}\n\n"
            f"- model={cfg.model} window={cfg.window} stride={cfg.eff_stride} regime={cfg.regime_name}\n"
            f"- **normalization:** MAX (frozen p95, 5 global channels) <- {cfg.scaler_csv}\n"
            f"- **loss:** γ + ΔV supervised + regime-split steady-state physics "
            f"(5-phase ramp, W_SUP_MIN={W_SUP_MIN})\n"
            f"- w_dv={cfg.w_dv} gate_center={cfg.gate_center} gate_width={cfg.gate_width} "
            f"mindlin_iters={cfg.mindlin_iters}\n"
            f"- vel_filter_crossover_hz={cfg.vel_filter_crossover_hz} "
            f"integrator={cfg.vel_filter_integrator} noise_stage={cfg.noise_stage}\n"
            f"- AdamW lr={cfg.lr} wd={cfg.weight_decay} grad_clip={cfg.grad_clip}; "
            f"ReduceLROnPlateau factor={cfg.sched_factor} patience={cfg.sched_patience} "
            f"min_lr={cfg.sched_min_lr}\n"
            f"- warm_from={cfg.warm_from or 'none'}\n")

    model = build_model_v2hy3(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    scheduler = _new_scheduler(opt, cfg)

    amp_dtype = _resolve_precision(cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    plan = _phase_plan(cfg)

    # Shared trunk = everything BOTH heads read from (encoder + feat + wheel_emb),
    # excluding the task-specific heads. Used by GradNorm and the cosine diagnostic.
    shared_params = [p for n, p in model.named_parameters()
                     if p.requires_grad and not (n.startswith("head_gamma")
                                                 or n.startswith("head_dv"))]

    if cfg.cosine_diag:
        print(f"[train-v2hy3] cosine-diag ON (every {cfg.cosine_diag_every} epochs, "
              f"<={cfg.cosine_diag_batches} batches/heartbeat) over "
              f"{len(shared_params)} shared tensors")

    # GradNorm state (adaptive gamma/dV weights).
    gn_w = gn_opt = None
    gn_state: Dict[str, object] = {"L0": None}
    if cfg.use_gradnorm:
        gn_w = torch.nn.Parameter(torch.ones(2, device=device))
        gn_opt = torch.optim.Adam([gn_w], lr=cfg.gradnorm_lr)
        print(f"[train-v2hy3] GradNorm ON (alpha={cfg.gradnorm_alpha} "
              f"lr={cfg.gradnorm_lr}) over gamma/dV; {len(shared_params)} shared tensors; "
              f"slip stays FIXED at w_slip={cfg.w_slip}")

    print(f"[train-v2hy3] model={cfg.model} params={sum(p.numel() for p in model.parameters()):,} "
          f"epochs={len(plan)} batch={cfg.batch_size} "
          f"warm_from={cfg.warm_from or 'none'}")

    start_epoch = 0
    ckpt = _ckpt_path(cfg)
    best_ckpt = _best_ckpt_path(cfg)
    best_metric = float("inf")
    lr_trace: List[float] = []
    cosine_trace: List[Dict[str, float]] = []

    if cfg.warm_from and not ckpt.exists():
        load_warm_start_v2hy3(model, cfg.warm_from)
        print(f"[train-v2hy3] warm-started encoder from {cfg.warm_from} (heads fresh)")

    if ckpt.exists():
        stt = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(stt["model"])
        opt.load_state_dict(stt["opt"])
        scheduler.load_state_dict(stt["scheduler"])
        start_epoch = stt["epoch"] + 1
        best_metric = stt.get("best_metric", float("inf"))
        lr_trace = list(stt.get("lr_trace", []))
        if cfg.use_gradnorm and stt.get("gn_w") is not None:
            with torch.no_grad():
                gn_w.copy_(torch.as_tensor(stt["gn_w"], device=device))
            if stt.get("gn_L0") is not None:
                gn_state["L0"] = torch.as_tensor(stt["gn_L0"], device=device)
            print(f"[train-v2hy3] restored GradNorm weights {gn_w.tolist()}")
        print(f"[train-v2hy3] resumed from epoch {start_epoch}/{len(plan)}")

    train_loader, val_loader = D2H.make_loaders(files, nrm, cfg)

    # Grounding early-stop state (supervised-val plateau) + LR-reset latch.
    grnd_best = float("inf")
    grnd_bad = 0
    lr_reset_done = (start_epoch < len(plan)
                     and plan[start_epoch]["phase"] != "grounding")
    n_executed = 0
    val_term = float("nan")

    ge = start_epoch
    while ge < len(plan):
        ph = plan[ge]

        # LR reset + fresh scheduler at the grounding -> physics hand-off, so the
        # physics curriculum starts at full LR regardless of grounding's collapse.
        if (cfg.lr_reset_at_physics and not lr_reset_done
                and ph["phase"] != "grounding"):
            for g in opt.param_groups:
                g["lr"] = cfg.lr
            scheduler = _new_scheduler(opt, cfg)
            best_metric = float("inf")        # grounding tracked supervised; physics
            lr_reset_done = True              # tracks terminal -> reset (diff scales)
            print(f"[train-v2hy3] LR reset -> {cfg.lr:.2e} + fresh scheduler "
                  f"entering phase '{ph['phase']}' (epoch {ge})")

        model.train()
        agg_sup = 0.0
        agg_phys = 0.0
        agg_gamma = 0.0
        agg_dv = 0.0
        agg_slip = 0.0
        agg_wg = 0.0
        agg_wd = 0.0
        log_sum: Dict[str, float] = {}
        nb = 0

        # Cosine-conflict heartbeat: measure only on every Nth epoch, capped batches.
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
                gamma_hat, dv_hat = model(Gw, Pw)
            gamma_hat = gamma_hat.float()
            dv_hat = dv_hat.float()

            # Per-target supervised losses, kept SEPARATE for GradNorm.
            se_gamma = ((gamma_hat - y_gamma) ** 2).mean(dim=-1)
            L_gamma = (se_gamma * supw).mean()
            se_dv = ((dv_hat - y_dv) ** 2).mean(dim=-1)
            L_dv = se_dv.mean()

            # Adaptive gamma/dV weighting (GradNorm) or fixed (1, w_dv).
            if cfg.use_gradnorm:
                wg, wd = _gradnorm_step(gn_state, L_gamma, L_dv, shared_params,
                                        gn_w, gn_opt, cfg.gradnorm_alpha)
            else:
                wg, wd = 1.0, cfg.w_dv
            l_sup = wg * L_gamma + wd * L_dv

            # Derived-slip consistency term (no new head; FIXED weight, OUTSIDE the
            # adaptive pool): pull |Vp|(γ̂,ΔV̂) -> vpm.
            l_slip = None
            if cfg.w_slip > 0.0:
                V_hat = Gw[:, -1, :2] * vhat_scale_t
                v_slip = derived_contact_slip(gamma_hat, dv_hat, V_hat,
                                              _phys_batch(batch, device), cfg,
                                              gamma_std_t, gamma_mean_t)
                l_slip = slip_consistency_loss(v_slip, batch["slip_mag"].to(device),
                                               cfg.vpm_scale)
                l_sup = l_sup + cfg.w_slip * l_slip
                agg_slip += float(l_slip.detach())

            # Cosine-conflict measurement (pure diagnostic; does NOT touch the loss).
            # Placed after l_slip so the slip term can join as a third task, and
            # before the main backward frees the graph.
            if diag_on and nb < cfg.cosine_diag_batches:
                _tasks = {"gamma": L_gamma, "dv": L_dv}
                if l_slip is not None:
                    _tasks["slip"] = l_slip
                diag_recs.append(_cosine_conflict(_tasks, shared_params))

            loss = ph["w_sup"] * l_sup
            agg_gamma += float(se_gamma.mean().detach())
            agg_dv += float(se_dv.mean().detach())
            agg_wg += float(wg); agg_wd += float(wd)

            if ph["w_phys"] > 0.0 and cfg.physics_loss:
                # V̂ from the final step of Gw (first two channels), de-normalized.
                V_hat = Gw[:, -1, :2] * vhat_scale_t
                # NORMALIZED gamma: physics_loss_hy3 de-normalizes in-graph. Passing
                # gamma already in rad/s double-scales it by gamma_p95 (~83x) and
                # pins the slip gate at g=1 (the run-1 "stick_frac=0" symptom).
                l_phys, phys_log = physics_loss_hy3(
                    gamma_hat, dv_hat, V_hat,
                    _phys_batch(batch, device), cfg, gamma_std_t, gamma_mean_t)
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
        n_executed += 1

        # Validation metric. During grounding, physics is OFF and the terminal
        # metric is physics-dominated (flat/meaningless), so drive the scheduler,
        # best-epoch and early-stop off the SUPERVISED metric. Once physics
        # engages, use the terminal metric.
        svm = None
        sup_val = None
        if ph["phase"] == "grounding":
            svm = supervised_val_metrics(model, val_loader, device, amp_dtype, cfg,
                                         gamma_std_t, gamma_mean_t, vhat_scale_t)
            sup_val = svm["total"]
            val_metric = val_term = sup_val
        else:
            val_term = terminal_val_loss(model, val_loader, device, amp_dtype, cfg,
                                         gamma_std_t, gamma_mean_t, dv_scale_t)
            val_metric = val_term

        # Scheduler steps on the phase-appropriate metric (constant-objective phases).
        if ph["phase"] in _STEP_PHASES:
            scheduler.step(val_metric)
        current_lr = opt.param_groups[0]["lr"]
        lr_trace.append(float(current_lr))

        # Best-epoch checkpoint (overwrites checkpoint_best.pt) on the active metric.
        if val_metric < best_metric:
            best_metric = val_metric
            torch.save(dict(model=model.state_dict(), epoch=ge, cfg=_cfg_dict(cfg),
                            best_metric=best_metric),
                       best_ckpt)

        # Concise physics summary.
        phys_summary = ""
        if cfg.physics_loss and log_sum:
            stick = sum(log_sum.get(f"phys_stick_w{i}", 0.0) for i in range(1, 5)) / nb
            slip = sum(log_sum.get(f"phys_slip_w{i}", 0.0) for i in range(1, 5)) / nb
            gmean = log_sum.get("g_mean", 0.0) / nb
            stick_frac = log_sum.get("g_stick_frac", 0.0) / nb   # fraction g<0.5 (stick)
            deep_frac = log_sum.get("g_lt01_frac", 0.0) / nb     # fraction g<0.1
            phys_summary = (f"phys(stick {stick:.4f} slip {slip:.4f} g {gmean:.4f} "
                            f"stick_frac {stick_frac:.3f} g<0.1 {deep_frac:.3f})")
        else:
            phys_summary = "phys n/a"

        # Train supervised breakdown (γ/ΔV normalized MSE; slip physical RMSE).
        sup_str = (f"g_mse {agg_gamma/nb:.5f} dv_mse {agg_dv/nb:.5f}")
        if cfg.w_slip > 0.0:
            sup_str += f" slip_rmse {((agg_slip/nb) ** 0.5) * cfg.vpm_scale:.4f}m/s"
        if cfg.use_gradnorm:
            sup_str += f" gn(wg {agg_wg/nb:.3f} wd {agg_wd/nb:.3f})"
        grnd_str = ""
        if sup_val is not None:
            grnd_str = f" | val sup {sup_val:.5f} g_mse {svm['gamma_mse']:.5f} dv_mse {svm['dv_mse']:.5f}"
            if cfg.w_slip > 0.0:
                grnd_str += f" slip_rmse {(svm['slip_mse'] ** 0.5) * cfg.vpm_scale:.4f}m/s"

        # Cosine-conflict heartbeat line + persisted trace (pairwise over all tasks).
        if diag_on and diag_recs:
            keys = list(diag_recs[0].keys())
            rec: Dict[str, float] = dict(epoch=ge, phase=ph["phase"],
                                         n_batches=len(diag_recs),
                                         gamma_mse=agg_gamma / nb, dv_mse=agg_dv / nb)
            cos_parts, norm_parts = [], []
            for k in keys:
                v = np.asarray([d[k] for d in diag_recs])
                if k.startswith("cos_"):
                    rec[f"{k}_mean"] = float(v.mean())
                    rec[f"{k}_med"] = float(np.median(v))
                    rec[f"{k}_p10"] = float(np.percentile(v, 10))
                    rec[f"{k}_p90"] = float(np.percentile(v, 90))
                    rec[f"{k}_negfrac"] = float((v < 0).mean())
                    cos_parts.append(f"{k[4:]} {v.mean():+.4f}"
                                     f"(med {np.median(v):+.4f} neg {(v < 0).mean():.2f})")
                else:
                    rec[k] = float(v.mean())
                    norm_parts.append(f"|d{k[3:]}| {v.mean():.3e}")
            if "gn_gamma" in rec and "gn_dv" in rec:
                rec["gn_ratio"] = rec["gn_gamma"] / max(rec["gn_dv"], 1e-12)
                norm_parts.append(f"ratio {rec['gn_ratio']:.2f}")
            cosine_trace.append(rec)
            with open(run_dir / "cosine_diag.json", "w") as fh:
                json.dump(cosine_trace, fh, indent=1)
            grnd_str += (f"\n      [cos-diag ep{ge}] " + "  ".join(cos_parts) +
                         "\n      [cos-diag ep{}] ".format(ge) + "  ".join(norm_parts) +
                         f"  (n={len(diag_recs)} batches)")
        print(f"[{ge:3d}/{len(plan)} {ph['phase']:<13} "
              f"ws{ph['w_sup']:.2f} wp{ph['w_phys']:.2f} lr{current_lr:.2e}] "
              f"sup {agg_sup/nb:.5f} {sup_str} {phys_summary} "
              f"val_term {val_term:.5f} best {best_metric:.5f}{grnd_str}")

        # Resumable checkpoint: overwrite checkpoint.pt every 10 epochs (+ last).
        if ge % 10 == 0 or ge == len(plan) - 1:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                            scheduler=scheduler.state_dict(), epoch=ge,
                            cfg=_cfg_dict(cfg), best_metric=best_metric,
                            lr_trace=lr_trace,
                            gn_w=(gn_w.detach().cpu() if cfg.use_gradnorm else None),
                            gn_L0=(gn_state["L0"].cpu() if cfg.use_gradnorm
                                   and gn_state["L0"] is not None else None)),
                       ckpt)

        # Grounding early-stop decision (supervised-val plateau, floored at min).
        early_stop_now = False
        if ph["phase"] == "grounding" and cfg.grounding_early_stop and sup_val is not None:
            if sup_val < grnd_best * (1.0 - cfg.grounding_rel_tol):
                grnd_best = sup_val
                grnd_bad = 0
            else:
                grnd_bad += 1
            if (ge + 1) >= cfg.grounding_min_epochs and grnd_bad >= cfg.grounding_patience:
                early_stop_now = True

        # Phase snapshot at phase end / run end / grounding early-stop.
        is_last = (ge == len(plan) - 1)
        next_phase = plan[ge + 1]["phase"] if not is_last else None
        if is_last or next_phase != ph["phase"] or early_stop_now:
            snap_dir = run_dir / "phase_ckpts"
            snap_dir.mkdir(exist_ok=True)
            torch.save(dict(model=model.state_dict(), epoch=ge, phase=ph["phase"],
                            w_sup=ph["w_sup"], w_phys=ph["w_phys"],
                            cfg=_cfg_dict(cfg)),
                       snap_dir / f"{ph['phase']}_ep{ge:03d}.pt")

        if early_stop_now:
            nxt = _first_non_grounding(plan)
            tail = "physics phases" if nxt < len(plan) else "end (grounding-only)"
            print(f"[train-v2hy3] grounding EARLY-STOP at epoch {ge} "
                  f"(sup_val {sup_val:.5f}; {grnd_bad} epochs no-improve >= "
                  f"patience {cfg.grounding_patience}) -> {tail}")
            ge = nxt
            continue

        ge += 1

    print(f"[train-v2hy3] done ({n_executed} epochs executed) -> {ckpt}")

    metrics = dict(
        val_loss=float(val_term),
        val_state_loss=float(val_term),
        epochs=n_executed,
        epochs_planned=len(plan),
        grounding_early_stop=cfg.grounding_early_stop,
        lr_reset_at_physics=cfg.lr_reset_at_physics,
        model="ssm_v2hy3_gamma_dv" if cfg.model == "ssm" else "gru_v2hy3_gamma_dv",
        window=cfg.window,
        stride=cfg.eff_stride,
        regime=cfg.regime_name,
        chi_fold_test=cfg.chi_fold_test,
        physics_loss=cfg.physics_loss,
        norm_method=cfg.norm_method,
        w_dv=cfg.w_dv,
        gate_center=cfg.gate_center,
        gate_width=cfg.gate_width,
        mindlin_iters=cfg.mindlin_iters,
        sched_factor=cfg.sched_factor,
        sched_patience=cfg.sched_patience,
        sched_min_lr=cfg.sched_min_lr,
        lr_trace=lr_trace,
        warm_from=cfg.warm_from,
        vel_filter_crossover_hz=cfg.vel_filter_crossover_hz,
        vel_filter_integrator=cfg.vel_filter_integrator,
        noise_stage=cfg.noise_stage,
        params=sum(p.numel() for p in model.parameters()),
        cfg=_cfg_dict(cfg),
    )
    with open(run_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=0)
    print(f"[train-v2hy3] metrics -> {run_dir / 'metrics.json'}")
