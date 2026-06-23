"""Training: phase schedule (grounding -> rampup -> overlap -> rampdown ->
physics-only), Adam-per-phase + EarlyStopper, L-BFGS refinement, checkpoint I/O.

Schedule + GPU idioms retained from train_GPU_PINN_v14_py (compile, global flags
set in stages.py; per-phase lr_scale; NaN guard; grad clip). Adapted to the
forces / param-ID API: loss functions return components, the trainer weights
them (w_grnd, w_phys ramp per phase; w_cons, w_param_id constant from config).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from .losses import forward_losses, inverse_losses
from .models import MecanumPINN, set_grad, maybe_compile_pinn
from .physics import RobotParams


# ============================================================
# EarlyStopper + checkpoint I/O
# ============================================================
class EarlyStopper:
    def __init__(self, patience=8, min_epochs=20, rel_delta=1e-3):
        self.patience, self.min_epochs, self.rel_delta = patience, min_epochs, rel_delta
        self.best, self.bad, self.epoch = float('inf'), 0, 0

    def step(self, val_loss: float) -> bool:
        self.epoch += 1
        if val_loss < self.best * (1.0 - self.rel_delta):
            self.best, self.bad = val_loss, 0
        else:
            self.bad += 1
        return self.epoch >= self.min_epochs and self.bad > self.patience


def _clean_sd(sd: Dict) -> Dict:
    """Strip torch.compile's `_orig_mod.` prefix so checkpoints load into an
    uncompiled skeleton."""
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}


def save_phase_checkpoint(model, optimizer, history, ckpt_dir, run_tag,
                          stage: str, phase: str, extra: Optional[Dict] = None):
    out_dir = Path(ckpt_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {'model': _clean_sd(model.state_dict()),
               'opt': optimizer.state_dict() if optimizer is not None else None,
               'history': history, 'stage': stage, 'phase': phase}
    if extra:
        payload.update(extra)
    torch.save(payload, out_dir / f"{stage}_{phase}.pth")


def load_phase_checkpoint(model, ckpt_path, map_location=None) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    incompat = model.load_state_dict(ckpt['model'], strict=False)
    if incompat.missing_keys:
        print(f"[checkpoint] missing keys: {incompat.missing_keys[:5]}")
    if incompat.unexpected_keys:
        print(f"[checkpoint] unexpected keys: {incompat.unexpected_keys[:5]}")
    return ckpt


def _new_history():
    return {'train': defaultdict(list), 'val': defaultdict(list), 'phase_boundaries': []}


# ============================================================
# Stage param selection + loss composition
# ============================================================
def _trainable_params(model: MecanumPINN, stage: str) -> List[nn.Parameter]:
    m = model.forward_model if stage == 'forward' else model.inverse_model
    return list(m.parameters())


def _set_stage_grad(model: MecanumPINN, stage: str) -> None:
    set_grad(model.forward_model, stage == 'forward')
    set_grad(model.inverse_model, stage == 'inverse')


def _compute_components(model, batch, rp, cfg, stage) -> Dict[str, torch.Tensor]:
    S, U, _T, _S_next, _F_sim, mu, chi = batch
    if stage == 'forward':
        F_phys, _shapes = model.forward_model(S, U, mu, chi)
        return forward_losses(F_phys, batch, rp, cfg)
    with torch.no_grad():                          # frozen forward backbone
        F_fwd, _shapes = model.forward_model(S, U, mu, chi)
    F_inv = model.inverse_model(S, U)
    return inverse_losses(F_inv, F_fwd, batch, rp, cfg)


def _aggregate(comp, stage, cfg, w_grnd, w_phys) -> torch.Tensor:
    if stage == 'forward':
        return w_grnd * comp['grnd'] + w_phys * comp['phys']
    # param-ID is test-only (not trained); consistency monitored at w_cons=0.
    return (w_grnd * comp['grnd'] + w_phys * comp['phys']
            + cfg['w_cons'] * comp['cons'])


# ============================================================
# Epoch loop
# ============================================================
def _epoch_loop(model, loader, rp, cfg, stage, w_grnd, w_phys,
                optimizer=None, train=True, desc=""):
    model.train() if train else model.eval()
    device = cfg['device']
    sums = defaultdict(float)
    n = 0
    trainable = _trainable_params(model, stage)
    grad_clip = cfg[stage]['grad_clip']
    ctx = torch.enable_grad() if train else torch.no_grad()
    # disable=None -> auto-disable the live bar when stdout isn't a TTY (i.e. when
    # the launcher redirects to a per-job log), keeping the file clean; the
    # per-epoch tqdm.write() summary lines below still print either way.
    bbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=None)
    with ctx:
        for batch in bbar:
            batch = tuple(b.to(device) if torch.is_tensor(b) else b for b in batch)
            comp = _compute_components(model, batch, rp, cfg, stage)
            total = _aggregate(comp, stage, cfg, w_grnd, w_phys)
            if train and not torch.isfinite(total):
                bbar.close()
                raise RuntimeError(
                    f"[NaN guard] non-finite loss in {stage}; lower lr / w_phys_max.")
            if train:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                optimizer.step()
            for k, v in comp.items():
                if torch.is_tensor(v) and v.dim() == 0:
                    sums[k] += float(v.detach())
            sums['total'] += float(total.detach())
            n += 1
            if n % 5 == 0:
                bbar.set_postfix({'loss': f"{sums['total']/n:.2e}"})
    bbar.close()
    return {k: v / max(1, n) for k, v in sums.items()}


# ============================================================
# Phase runner + schedules
# ============================================================
def _lin(a, b, ep, n):
    return a + (b - a) * (ep + 1) / max(1, n)


def _run_phase(model, tr_loader, va_loader, rp, cfg, stage, history, next_ep,
               phase_name, n_epochs,
               w_grnd_fn: Callable[[int, int], float],
               w_phys_fn: Callable[[int, int], float],
               early_stop=False):
    if n_epochs <= 0:
        return next_ep
    _set_stage_grad(model, stage)
    lr = cfg['lr'] * cfg[stage]['lr_scale'][phase_name]
    optimizer = optim.Adam(_trainable_params(model, stage), lr=lr)
    stopper = (EarlyStopper(cfg[stage]['patience'], cfg[stage]['min_epochs'],
                            cfg[stage]['rel_delta']) if early_stop else None)
    h = history[stage]
    h['phase_boundaries'].append((phase_name, next_ep))
    print(f"\n--- Phase {phase_name} ({stage}) | epochs {next_ep}..{next_ep+n_epochs-1} "
          f"| lr={lr:.1e} | ES={'on' if stopper else 'off'} ---")
    ep = -1
    for ep in range(n_epochs):
        w_g, w_p = float(w_grnd_fn(ep, n_epochs)), float(w_phys_fn(ep, n_epochs))
        tr = _epoch_loop(model, tr_loader, rp, cfg, stage, w_g, w_p,
                         optimizer=optimizer, train=True,
                         desc=f"{stage}/{phase_name} ep{ep+1}/{n_epochs}")
        va = _epoch_loop(model, va_loader, rp, cfg, stage, w_g, w_p,
                         optimizer=None, train=False, desc="  val")
        for k, v in tr.items():
            h['train'][k].append(v)
        for k, v in va.items():
            h['val'][k].append(v)
        tqdm.write(f"[{stage}/{phase_name}] ep {next_ep+ep+1:>3} | "
                   f"tr={tr.get('total', float('nan')):.3e} "
                   f"va={va.get('total', float('nan')):.3e} | wg={w_g:.2f} wp={w_p:.2e}")
        if stopper and stopper.step(va['total']):
            print(f"[ES {stage}/{phase_name}] stop @ ep {next_ep+ep+1} (best={stopper.best:.3e})")
            break
    next_ep += ep + 1
    save_phase_checkpoint(model, optimizer, history, cfg['ckpt_dir'], cfg['run_tag'],
                          stage, phase_name)
    return next_ep


def _run_full_schedule(model, tr_loader, va_loader, rp, cfg, stage, history):
    wmax = cfg['w_phys_max']
    s = cfg[stage]
    ne = 0
    ne = _run_phase(model, tr_loader, va_loader, rp, cfg, stage, history, ne,
                    'grounding', s['grounding_epochs'],
                    lambda e, t: 1.0, lambda e, t: 0.0, early_stop=True)
    ne = _run_phase(model, tr_loader, va_loader, rp, cfg, stage, history, ne,
                    'phys_rampup', s['rampup_epochs'],
                    lambda e, t: 1.0, lambda e, t: _lin(0.0, wmax, e, t))
    ne = _run_phase(model, tr_loader, va_loader, rp, cfg, stage, history, ne,
                    'overlap', s['overlap_epochs'],
                    lambda e, t: 1.0, lambda e, t: wmax)
    ne = _run_phase(model, tr_loader, va_loader, rp, cfg, stage, history, ne,
                    'grnd_rampdown', s['rampdown_epochs'],
                    lambda e, t: _lin(1.0, 0.0, e, t), lambda e, t: wmax)
    ne = _run_phase(model, tr_loader, va_loader, rp, cfg, stage, history, ne,
                    'physics', s['physics_epochs'],          # physics-only final phase
                    lambda e, t: 0.0, lambda e, t: wmax, early_stop=True)
    return model, history


# ============================================================
# L-BFGS refinement
# ============================================================
def run_lbfgs_refine(model, tr_loader, rp, cfg, stage, history):
    s = cfg[stage]
    if not s.get('use_lbfgs', True):
        print(f"[L-BFGS {stage}] disabled — skipping.")
        return model, history
    print(f"\n--- L-BFGS refinement ({stage}) | max_iter={s['lbfgs_max_iter']} ---")
    _set_stage_grad(model, stage)
    trainable = _trainable_params(model, stage)
    optimizer = optim.LBFGS(trainable, lr=s['lbfgs_lr'], max_iter=s['lbfgs_max_iter'],
                            history_size=10, line_search_fn='strong_wolfe')
    wmax = cfg['w_phys_max']
    device = cfg['device']

    def closure():
        optimizer.zero_grad(set_to_none=True)
        nb = len(tr_loader)
        tot = 0.0
        for batch in tr_loader:
            batch = tuple(b.to(device) if torch.is_tensor(b) else b for b in batch)
            comp = _compute_components(model, batch, rp, cfg, stage)
            t = _aggregate(comp, stage, cfg, 0.0, wmax)      # physics-only weighting
            (t / nb).backward()
            tot += float(t.detach())
        torch.nn.utils.clip_grad_norm_(trainable, s['grad_clip'])
        return torch.tensor(tot / max(1, nb), device=device)

    try:
        loss = optimizer.step(closure)
        print(f"[{stage}/lbfgs] final loss = {float(loss):.4e}")
    except Exception as e:                                   # pragma: no cover
        print(f"[{stage}/lbfgs] aborted: {e}")
    save_phase_checkpoint(model, optimizer, history, cfg['ckpt_dir'], cfg['run_tag'],
                          stage, 'lbfgs')
    return model, history


# ============================================================
# Top-level trainers
# ============================================================
def train_forward(rp: RobotParams, tr_loader, va_loader, cfg: Dict,
                  model: Optional[MecanumPINN] = None) -> Tuple[MecanumPINN, Dict]:
    if model is None:
        model = MecanumPINN(cfg, rp).to(cfg['device'])
        model = maybe_compile_pinn(model, cfg)
    history = {'forward': _new_history()}
    model, history = _run_full_schedule(model, tr_loader, va_loader, rp, cfg,
                                        'forward', history)
    if cfg['forward'].get('use_lbfgs', True):
        model, history = run_lbfgs_refine(model, tr_loader, rp, cfg, 'forward', history)
    return model, history


def train_inverse(rp: RobotParams, tr_loader, va_loader, cfg: Dict,
                  model: MecanumPINN) -> Tuple[MecanumPINN, Dict]:
    history = {'inverse': _new_history()}
    model, history = _run_full_schedule(model, tr_loader, va_loader, rp, cfg,
                                        'inverse', history)
    if cfg['inverse'].get('use_lbfgs', True):
        model, history = run_lbfgs_refine(model, tr_loader, rp, cfg, 'inverse', history)
    return model, history
