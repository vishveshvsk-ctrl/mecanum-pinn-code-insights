#!/usr/bin/env python
# =============================================================================
# training.py — streaming window dataset + 5-phase causal-observer curriculum.
#
# Memory-safe: WindowStream is an IterableDataset that reads one Arrow file at a
# time, builds causal windows, and serves them through a bounded shuffle buffer.
# Files are sharded across (<=8) workers. Resume-aware: the latest checkpoint
# carries model+optimiser+global-epoch+config; per-phase snapshots are saved at
# phase boundaries for ablation.
#
# Curriculum mirrors train_GPU_PINN_v14's forward schedule (grounding ->
# phys_rampup -> overlap -> grnd_rampdown -> physics) with its lr scaling. The
# physics weight ramps 0->1 over phys_rampup and stays 1; the supervised weight
# ramps 1->W_SUP_MIN over grnd_rampdown. With physics_loss=False the physics
# weight is forced 0 and supervised stays 1 (the schedule then just provides the
# lr/epoch budget). Physics loss = quasi-static roller + wheel torque-balance
# residuals from the predicted states, with chi carried per-sample (loss-side
# only). Mz term excluded (low-SNR).
# =============================================================================
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

import pyarrow.feather  # noqa: F401  (import before torch — CLAUDE.md §7)
import torch
from torch import _dynamo  # noqa: F401  (Windows/WSL gotcha — CLAUDE.md §7)
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from . import config as C
from . import data as D
from .config import ObserverConfig
from .losses import observer_loss, physics_loss
from .models import build_model

_PHYS_KEYS = ["ph_psi_dot", "ph_Vpx0", "ph_Vpy0", "ph_cti", "ph_sti",
              "ph_Msat", "ph_w", "ph_w_dot", "ph_mu", "ph_chi",
              "ph_Vx", "ph_Vy", "ph_dVx", "ph_dVy", "ph_dpsi_dot",
              "ph_Vx_next", "ph_Vy_next", "ph_psi_dot_next", "ph_w_next"]


class WindowStream(IterableDataset):
    def __init__(self, files: List[Path], nrm: D.Normalizer, cfg: ObserverConfig,
                 shuffle: bool):
        super().__init__()
        self.files = list(files)
        self.nrm = nrm
        self.cfg = cfg
        self.shuffle = shuffle
        self.keys = ["Gw", "Pw", "Yt"] + (_PHYS_KEYS if cfg.physics_loss else [])

    def _my_files(self) -> List[Path]:
        wi = get_worker_info()
        if wi is None:
            files = list(self.files)
        else:
            files = self.files[wi.id:: wi.num_workers]       # shard by worker
        if self.shuffle:
            random.Random(self.cfg.seed + (0 if wi is None else wi.id)).shuffle(files)
        return files

    def _emit(self, item: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {k: torch.as_tensor(v) for k, v in item.items()}

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buf: List[Dict[str, np.ndarray]] = []
        cap = self.cfg.shuffle_buffer if self.shuffle else 1
        rng = random.Random(self.cfg.seed)
        for path in self._my_files():
            a = D.read_arrays(path, self.cfg.cache_dir)
            w = D.make_windows(a, self.nrm, self.cfg)
            del a
            if w is None:
                continue
            M = w["Gw"].shape[0]
            for j in range(M):
                item = {k: w[k][j] for k in self.keys}
                if not self.shuffle:
                    yield self._emit(item)
                    continue
                buf.append(item)
                if len(buf) >= cap:
                    k = rng.randrange(len(buf))
                    out = buf[k]; buf[k] = buf[-1]; buf.pop()
                    yield self._emit(out)
            del w
        rng.shuffle(buf)
        for out in buf:
            yield self._emit(out)


def _phase_plan(cfg: ObserverConfig) -> List[dict]:
    """Global-epoch schedule: [{phase, lr_scale, w_sup, w_phys}, ...]."""
    if cfg.phases != "a1_5phase":
        return [dict(phase="flat", lr_scale=1.0, w_sup=1.0, w_phys=0.0)
                for _ in range(cfg.epochs)]

    # Warm-start refinement skips grounding and appends a pure-physics tail.
    sched = C.REFINE_SCHEDULE if cfg.warm_from else C.PHASE_SCHEDULE
    if cfg.phase_total_epochs > 0:                          # scale the refine/base phases to this total
        base = sum(n for _, n, _ in sched)
        f = cfg.phase_total_epochs / base
        sched = [(name, max(1, round(n * f)), lrs) for name, n, lrs in sched]

    plan: List[dict] = []
    phys_on = cfg.physics_loss
    for name, n, lrs in sched:
        for e in range(n):
            frac = e / max(n - 1, 1)
            if not phys_on:
                w_sup, w_phys = 1.0, 0.0
            elif name == "grounding":
                w_sup, w_phys = 1.0, 0.0
            elif name == "phys_rampup":
                w_sup, w_phys = 1.0, frac
            elif name == "overlap":
                w_sup, w_phys = 1.0, 1.0
            elif name == "grnd_rampdown":
                w_sup, w_phys = 1.0 - (1.0 - C.W_SUP_MIN) * frac, 1.0
            else:  # physics
                w_sup, w_phys = C.W_SUP_MIN, 1.0
            plan.append(dict(phase=name, lr_scale=lrs, w_sup=w_sup, w_phys=w_phys))

    if cfg.warm_from:
        for _ in range(C.PURE_PHYSICS_EPOCHS):
            plan.append(dict(phase="pure_physics", lr_scale=C.PURE_PHYSICS_LR,
                             w_sup=0.0, w_phys=1.0))
    return plan


def _resolve_precision(cfg: ObserverConfig, device: torch.device) -> Optional[torch.dtype]:
    """auto: bf16 only on Ampere+ (the RTX 3060); the Turing Quadro RTX 6000 has
    no native bf16 -> v14 disables AMP (fp32) rather than fp16+scaler."""
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


def _ckpt_path(cfg: ObserverConfig) -> Path:
    return Path(cfg.out_dir) / cfg.run_tag / "checkpoint.pt"


def _phys_batch(batch, device):
    return dict(
        psi_dot=batch["ph_psi_dot"].to(device), Vpx0=batch["ph_Vpx0"].to(device),
        Vpy0=batch["ph_Vpy0"].to(device), cti=batch["ph_cti"].to(device),
        sti=batch["ph_sti"].to(device), Msat=batch["ph_Msat"].to(device),
        w=batch["ph_w"].to(device), w_dot=batch["ph_w_dot"].to(device),
        mu=batch["ph_mu"].to(device), chi=batch["ph_chi"].to(device),
        Vx=batch["ph_Vx"].to(device), Vy=batch["ph_Vy"].to(device),
        dVx=batch["ph_dVx"].to(device), dVy=batch["ph_dVy"].to(device),
        dpsi_dot=batch["ph_dpsi_dot"].to(device),
        Vx_next=batch["ph_Vx_next"].to(device),
        Vy_next=batch["ph_Vy_next"].to(device),
        psi_dot_next=batch["ph_psi_dot_next"].to(device),
        w_next=batch["ph_w_next"].to(device))


def train(cfg: ObserverConfig) -> None:
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
        print(f"[train] device=cuda:0 {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
    else:
        device = torch.device("cpu")
        if cfg.require_gpu:
            raise RuntimeError("CUDA not available but --require-gpu set.")
        print("[train] WARNING: CUDA unavailable -> training on CPU (slow)")

    files = D.discover(cfg)
    sp = D.split_files(files, cfg)
    print(f"[train] {len(files)} files -> train {len(sp['train'])} "
          f"val {len(sp['val'])} test {len(sp['test'])}")
    with open(run_dir / "split.json", "w") as fh:
        json.dump({k: [p.name for p in v] for k, v in sp.items()}, fh, indent=0)

    nrm_path = run_dir / "norm.npz"
    if cfg.norm_method == "max":
        if not cfg.scaler_csv:
            raise ValueError("norm_method='max' requires scaler_csv (variable_scaler_percentiles.csv)")
        nrm = D.load_max_scaler(cfg.scaler_csv); nrm.to_npz(nrm_path)
        print(f"[train] MAX-norm (frozen p95, sin/cos unscaled) <- {cfg.scaler_csv}")
    elif nrm_path.exists():
        nrm = D.Normalizer.from_npz(nrm_path); print("[train] loaded normaliser")
    else:
        print("[train] fitting normaliser (streaming over train files)...")
        nrm = D.fit_normalizer(sp["train"], cfg.cache_dir); nrm.to_npz(nrm_path)
    with open(run_dir / "LOSS_AND_NORM.md", "w", encoding="utf-8") as fh:
        fh.write(
            f"# Run: {cfg.run_tag}\n\n"
            f"- model={cfg.model} window={cfg.window} stride={cfg.eff_stride} regime={cfg.regime_name}\n"
            f"- **normalization:** {'MAX (frozen p95; sin/cos unscaled) <- ' + cfg.scaler_csv if cfg.norm_method == 'max' else 'VAR (z-score, fit on train)'}\n"
            f"- **loss:** {'supervised + physics (5-phase ramp)' if cfg.physics_loss else 'SUPERVISED ONLY (physics_loss=False)'}\n"
            f"- phases={cfg.phases} phase_total_epochs={cfg.phase_total_epochs}; "
            f"AdamW lr={cfg.lr} wd={cfg.weight_decay} grad_clip={cfg.grad_clip}; no L-BFGS, no early stopping\n")
    y_mean_t = torch.tensor(nrm.y_mean, dtype=torch.float32, device=device).view(1, 1, -1)
    y_std_t = torch.tensor(nrm.y_std, dtype=torch.float32, device=device).view(1, 1, -1)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    print(f"[train] model={cfg.model} params={sum(p.numel() for p in model.parameters()):,} "
          f"phases={cfg.phases} physics_loss={cfg.physics_loss} "
          f"variant={cfg.physics_variant} warm_from={cfg.warm_from or 'none'}")

    # Body mass-matrix inverse (needed by integrated variant).
    Minv = torch.tensor(C.M_BODY_INV, dtype=torch.float32, device=device)

    amp_dtype = _resolve_precision(cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    plan = _phase_plan(cfg)

    start_epoch = 0
    ckpt = _ckpt_path(cfg)

    # Warm-start (weights only) from an external checkpoint before normal resume.
    if cfg.warm_from and not ckpt.exists():
        warm_ckpt = Path(cfg.warm_from)
        stt = torch.load(warm_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(stt["model"])          # weights ONLY — fresh optimiser, epoch 0
        print(f"[train] warm-started weights from {warm_ckpt} (refine schedule, epoch 0)")

    if ckpt.exists():
        stt = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(stt["model"]); opt.load_state_dict(stt["opt"])
        start_epoch = stt["epoch"] + 1
        print(f"[train] resumed from epoch {start_epoch}/{len(plan)}")

    def loader(files_, shuffle):
        ds = WindowStream(files_, nrm, cfg, shuffle)
        kw = dict(batch_size=cfg.batch_size, num_workers=cfg.jobs,
                  drop_last=shuffle, pin_memory=device.type == "cuda")
        if cfg.jobs > 0:
            kw.update(persistent_workers=True, prefetch_factor=4)
        return DataLoader(ds, **kw)

    train_loader, val_loader = loader(sp["train"], True), loader(sp["val"], False)

    vl = None
    epoch_log_totals: Dict[str, tuple] = {}   # key -> (running_sum, count)
    for ge in range(start_epoch, len(plan)):
        ph = plan[ge]
        for grp in opt.param_groups:
            grp["lr"] = cfg.lr * ph["lr_scale"]
        model.train()
        agg: Dict[str, float] = {"sup": 0.0}
        log_sum: Dict[str, float] = {}
        nb = 0
        for batch in train_loader:
            Gw = batch["Gw"].to(device, non_blocking=True)
            Pw = batch["Pw"].to(device, non_blocking=True)
            Yt = batch["Yt"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                pred = model(Gw, Pw)
            pred = pred.float()                              # physics math in fp32
            l_sup, _ = observer_loss(pred, Yt)
            loss = ph["w_sup"] * l_sup
            if ph["w_phys"] > 0.0 and cfg.physics_loss:
                pred_phys = pred * y_std_t + y_mean_t
                l_phys, phys_log = physics_loss(pred_phys, _phys_batch(batch, device),
                                                variant=cfg.physics_variant, Minv=Minv)
                loss = loss + ph["w_phys"] * l_phys
                agg["phys"] = agg.get("phys", 0.0) + float(l_phys.detach())
                for k, v in phys_log.items():
                    log_sum[k] = log_sum.get(k, 0.0) + v
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
            agg["sup"] += float(l_sup.detach()); nb += 1
        vl = evaluate_loss(model, val_loader, device, amp_dtype)
        nb = max(nb, 1)

        # Per-component epoch means.
        phys_summary = ""
        if cfg.physics_loss and log_sum:
            for k, v in log_sum.items():
                epoch_log_totals[k] = (epoch_log_totals.get(k, (0.0, 0))[0] + v,
                                       epoch_log_totals.get(k, (0.0, 0))[1] + 1)
            if cfg.physics_variant == "residual":
                wheel = sum(log_sum.get(f"phys_wheel_w{i}", 0.0) for i in range(1, 5)) / nb
                body_x = log_sum.get("phys_body_x", 0.0) / nb
                body_y = log_sum.get("phys_body_y", 0.0) / nb
                body_yaw = log_sum.get("phys_body_yaw", 0.0) / nb
                phys_summary = f"phys(w {wheel:.4f} bx {body_x:.4f} by {body_y:.4f} byaw {body_yaw:.4f})"
            else:  # integrated
                int_Vx = log_sum.get("phys_int_Vx", 0.0) / nb
                int_Vy = log_sum.get("phys_int_Vy", 0.0) / nb
                int_pd = log_sum.get("phys_int_psidot", 0.0) / nb
                int_w = sum(log_sum.get(f"phys_int_w{i}", 0.0) for i in range(1, 5)) / nb
                phys_summary = f"phys(Vx {int_Vx:.4f} Vy {int_Vy:.4f} pd {int_pd:.4f} w {int_w:.4f})"
        else:
            phys_summary = "phys n/a"

        print(f"[{ge:3d}/{len(plan)} {ph['phase']:<13} lr×{ph['lr_scale']:.2f} "
              f"ws{ph['w_sup']:.2f} wp{ph['w_phys']:.2f}] "
              f"sup {agg['sup']/nb:.5f} {phys_summary} val {vl:.5f}")

        # Latest checkpoint (for resume).
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        epoch=ge, cfg=asdict(cfg)), ckpt)

        # Per-phase snapshot: save at the last epoch of each phase (including the
        # final epoch of training).
        is_last_epoch = (ge == len(plan) - 1)
        next_phase = plan[ge + 1]["phase"] if not is_last_epoch else None
        if is_last_epoch or next_phase != ph["phase"]:
            snap_dir = run_dir / "phase_ckpts"; snap_dir.mkdir(exist_ok=True)
            torch.save(dict(model=model.state_dict(), epoch=ge, phase=ph["phase"],
                            w_sup=ph["w_sup"], w_phys=ph["w_phys"], cfg=asdict(cfg)),
                       snap_dir / f"{ph['phase']}_ep{ge:03d}.pt")
    print(f"[train] done -> {ckpt}")

    # metrics.json = completion marker + per-component epoch means.
    if vl is None:
        vl = evaluate_loss(model, val_loader, device, amp_dtype)
    metrics = dict(val_loss=float(vl), epochs=len(plan), model=cfg.model,
                   window=cfg.window, stride=cfg.eff_stride,
                   regime=cfg.regime_name, chi_fold_test=cfg.chi_fold_test,
                   physics_loss=cfg.physics_loss, physics_variant=cfg.physics_variant,
                   norm_method=cfg.norm_method,
                   params=sum(p.numel() for p in model.parameters()))
    if epoch_log_totals:
        for k, (s, n) in epoch_log_totals.items():
            metrics[k] = s / max(n, 1)
    with open(run_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=0)
    print(f"[train] metrics -> {run_dir / 'metrics.json'}")


@torch.no_grad()
def evaluate_loss(model, loader, device, amp_dtype) -> float:
    model.eval()
    tot, nb = 0.0, 0
    for batch in loader:
        Gw = batch["Gw"].to(device); Pw = batch["Pw"].to(device)
        Yt = batch["Yt"].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            loss, _ = observer_loss(model(Gw, Pw).float(), Yt)
        tot += float(loss); nb += 1
    return tot / max(nb, 1)
