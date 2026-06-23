"""Training primitives + phase runners + L-BFGS + top-level trainers.

v13: helpers _forward_model_kwargs / _inverse_model_kwargs forward all
new architecture knobs (hidden_dim_wheel, dec1_hidden, dec3_hidden,
inv_hidden, embed_dim) from config to the respective model constructors,
with safe defaults.

v14 θ-rework: dec_theta_hidden is no longer forwarded — the forward
model dropped its learned θ head. Older configs that still set
dec_theta_hidden continue to work because MecanumForwardModel.__init__
silently absorbs deprecated kwargs.

Per-component diagnostics: `_epoch_loop` accumulates any loss-dict key
ending in `_per_comp` into a side-channel dict (separate from `sums` so
it doesn't pollute the loss accounting), and the per-epoch log line in
`_run_phase` prints whichever breakdowns are present. This works
uniformly for forward (state/grnd/phys breakdowns) and inverse
(grnd/cons/phys breakdowns), in both grounding and physics phases.
"""
from __future__ import annotations

import copy
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from .losses import aggregate_total_loss, compute_losses
from .models import (MecanumForwardModel, MecanumInverseModel, MecanumPINN,
                     maybe_compile_pinn, set_grad)
from .physics import Geometry, RobotParams, make_geometry


# ============================================================
# Model construction kwargs sourced from config
# ============================================================
def _forward_model_kwargs(config: Dict, rp: RobotParams) -> Dict[str, Any]:
    """Forward-model constructor kwargs sourced from `config`.

    Note: `dec_theta_hidden` is no longer forwarded — the θ head was
    removed in v12.5's θ-rework. If older configs still set it, the
    MecanumForwardModel constructor silently absorbs it via **_legacy_kwargs.
    """
    return dict(
        raw_state_dim=config['raw_state_dim'],
        net_state_dim=config['net_state_dim'],
        ctrl_dim=config['ctrl_dim'],
        force_dim=config['force_dim'],
        hidden_dim=config['hidden_dim'],
        n_wheels=config['n_wheels'],
        hidden_dim_wheel=config.get('hidden_dim_wheel'),
        dec1_hidden=config.get('dec1_hidden', 32),
        dec3_hidden=config.get('dec3_hidden', 8),
        embed_dim=config.get('embed_dim', 4),
        F_max=rp.F_max,
        Mz_max=rp.Mz_max,
    )


def _inverse_model_kwargs(config: Dict, geom: Geometry, use_H: bool
                          ) -> Dict[str, Any]:
    return dict(
        geom=geom,
        raw_state_dim=config['raw_state_dim'],
        net_state_dim=config['net_state_dim'],
        ctrl_dim=config['ctrl_dim'],
        force_dim=config['force_dim'],
        hidden_dim=config['hidden_dim'],
        n_wheels=config['n_wheels'],
        hidden_dim_wheel=config.get('hidden_dim_wheel'),
        inv_hidden=config.get('inv_hidden', 32),
        embed_dim=config.get('embed_dim', 4),
        use_H=use_H,
    )


# ============================================================
# EarlyStopper + checkpoint I/O
# ============================================================
class EarlyStopper:
    def __init__(self, patience: int = 8, min_epochs: int = 20,
                 rel_delta: float = 1e-3):
        self.patience = patience
        self.min_epochs = min_epochs
        self.rel_delta = rel_delta
        self.best = float('inf')
        self.bad = 0
        self.epoch = 0

    def step(self, val_loss: float) -> bool:
        self.epoch += 1
        improved = val_loss < self.best * (1.0 - self.rel_delta)
        if improved:
            self.best = val_loss
            self.bad = 0
        else:
            self.bad += 1
        if self.epoch < self.min_epochs:
            return False
        return self.bad > self.patience


def save_phase_checkpoint(model: nn.Module, optimizer, history,
                          ckpt_dir: Path, run_tag: str,
                          stage: str, phase: str,
                          extra: Optional[Dict] = None):
    out_dir = Path(ckpt_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'model':    model.state_dict(),
        'opt':      optimizer.state_dict() if optimizer is not None else None,
        'history':  history,
        'stage':    stage,
        'phase':    phase,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, out_dir / f"{stage}_{phase}.pth")


def load_phase_checkpoint(model: nn.Module, ckpt_path: Path,
                          map_location=None) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    # `strict=False` so old checkpoints that include `dec_theta.*` keys
    # still load cleanly into the v12.5 (post-θ-rework) skeleton, which
    # no longer has those parameters. The unexpected_keys warning is
    # logged but doesn't block loading.
    incompat = model.load_state_dict(ckpt['model'], strict=False)
    if incompat.missing_keys or incompat.unexpected_keys:
        unexp = [k for k in incompat.unexpected_keys if 'dec_theta' in k]
        other_unexp = [k for k in incompat.unexpected_keys if 'dec_theta' not in k]
        if unexp:
            print(f"[checkpoint] ignoring {len(unexp)} legacy dec_theta keys "
                  f"(θ head was removed in v12.5)")
        if other_unexp:
            print(f"[checkpoint] WARNING: unexpected keys in state_dict: "
                  f"{other_unexp[:5]}{'...' if len(other_unexp) > 5 else ''}")
        if incompat.missing_keys:
            print(f"[checkpoint] WARNING: missing keys in state_dict: "
                  f"{incompat.missing_keys[:5]}"
                  f"{'...' if len(incompat.missing_keys) > 5 else ''}")
    return ckpt


def _new_history():
    return {'train': defaultdict(list), 'val': defaultdict(list),
            'phase_boundaries': []}


# ============================================================
# Stage parameter selection
# ============================================================
def _trainable_params(model: MecanumPINN, stage: str) -> List[nn.Parameter]:
    if stage == 'forward':
        return list(model.forward_model.parameters())
    elif stage == 'inverse':
        return list(model.inverse_model.parameters())
    raise ValueError(stage)


def _set_stage_grad(model: MecanumPINN, stage: str) -> None:
    if stage == 'forward':
        set_grad(model.forward_model, True)
        set_grad(model.inverse_model, False)
    else:
        set_grad(model.forward_model, False)
        set_grad(model.inverse_model, True)


# ============================================================
# Per-component diagnostic formatting
# ============================================================
def _format_per_comp(label: str, pc: torch.Tensor) -> str:
    """Format a per-component tensor into a one-line diagnostic string.

    Shape determines the layout:
       7 — V_X, V_Y, Ω, ω_1..4   (state / forward-phys / inverse-phys)
      12 — Fx_1..4, Fy_1..4, Mz_1..4   (grnd / cons)
    """
    n = pc.numel()
    if n == 7:
        return (f"           {label}: "
                f"V_X={pc[0]:.2e} V_Y={pc[1]:.2e} Ω={pc[2]:.2e} | "
                f"ω={pc[3:7].mean():.2e} (max={pc[3:7].max():.2e})")
    elif n == 12:
        return (f"           {label}: "
                f"Fx={pc[0:4].mean():.2e} (max={pc[0:4].max():.2e}) | "
                f"Fy={pc[4:8].mean():.2e} (max={pc[4:8].max():.2e}) | "
                f"Mz={pc[8:12].mean():.2e} (max={pc[8:12].max():.2e})")
    else:
        return f"           {label}: <unexpected shape {tuple(pc.shape)}>"


# Order in which to display per-comp breakdowns (only those present are shown).
_PER_COMP_DISPLAY_ORDER = [
    ('state_per_comp', 'state per-comp'),
    ('grnd_per_comp',  'grnd  per-comp'),
    ('cons_per_comp',  'cons  per-comp'),
    ('phys_per_comp',  'phys  per-comp'),
]


# ============================================================
# Epoch loop
# ============================================================
def _epoch_loop(model, loader, rp: RobotParams, config: Dict, stage: str,
                w_grnd: float, w_phys: float, w_cons: float = 0.0,
                optimizer=None, train: bool = True,
                desc: str = ""):
    if train:
        model.train()
    else:
        model.eval()

    sums = defaultdict(float)
    diag_sums: Dict[str, torch.Tensor] = {}    # '*_per_comp' tensors — diagnostic only
    n = 0
    grad_clip = config[stage]['grad_clip']
    trainable = _trainable_params(model, stage)

    bbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in bbar:
            losses = compute_losses(model, batch, rp, config, stage,
                                    w_phys=w_phys)
            total  = aggregate_total_loss(losses, stage, w_grnd, w_phys, w_cons)

            # NaN guard: once any weight goes non-finite, all subsequent
            # forward passes produce NaN and the optimizer happily corrupts
            # things further. Abort cleanly with diagnostic info instead.
            if train and not torch.isfinite(total):
                bbar.close()
                comp = {k: (v.item() if isinstance(v, torch.Tensor) and v.dim() == 0
                            else v) for k, v in losses.items()}
                raise RuntimeError(
                    f"[NaN guard] non-finite loss in {stage}: total={total.item()}\n"
                    f"  loss components: {comp}\n"
                    f"  Most likely causes: LR too high, w_phys too high for current "
                    f"weights, or AMP overflow. Lower lr, lower w_phys_max, or "
                    f"disable AMP for this run."
                )

            if train:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                optimizer.step()

            # Scalar loss accumulators
            for k, v in losses.items():
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    sums[k] += float(v.detach().float().item())
            sums['total'] += float(total.detach().float().item())

            # Per-component diagnostic accumulators (any '*_per_comp' key,
            # 1-D tensor). Kept separate from `sums` so they never enter
            # loss accounting; the only purpose is the per-epoch print.
            for k, v in losses.items():
                if (k.endswith('_per_comp')
                    and isinstance(v, torch.Tensor) and v.dim() == 1):
                    v_cpu = v.detach().float().cpu()
                    diag_sums[k] = v_cpu.clone() if k not in diag_sums \
                                   else diag_sums[k] + v_cpu

            n += 1

            if n % 5 == 0:
                bbar.set_postfix({'loss': f"{float(total.detach().float().item()):.2e}"})

    bbar.close()
    out = {k: v / max(1, n) for k, v in sums.items()}
    out.update({k: v / max(1, n) for k, v in diag_sums.items()})
    return out


# ============================================================
# Phase runner (Adam)
# ============================================================
def _run_phase(model, train_loader, val_loader, rp: RobotParams,
               config: Dict, stage: str,
               history: Dict, ema_scale, next_ep: int,
               phase_name: str, n_epochs: int,
               w_grnd_fn: Callable[[int, int], float],
               w_phys_fn: Callable[[int, int], float],
               early_stop_on: Optional[str] = None,
               print_every: int = 1):
    _set_stage_grad(model, stage)

    lr = config['lr'] * config[stage]['lr_scale'][phase_name]
    optimizer = optim.Adam(_trainable_params(model, stage), lr=lr)

    stopper = None
    if early_stop_on:
        stopper = EarlyStopper(patience=config[stage]['patience'],
                               min_epochs=config[stage]['min_epochs'],
                               rel_delta=config[stage]['rel_delta'])

    h = history[stage]
    h['phase_boundaries'].append((phase_name, next_ep))
    w_cons = config['w_cons']

    print(f"\n--- Phase: {phase_name} ({stage}) | epochs {next_ep}..{next_ep + n_epochs - 1} "
          f"| lr={lr:.1e} | ES={'on' if stopper else 'off'} ---")

    pbar = tqdm(range(n_epochs), desc=f"{stage}/{phase_name}", leave=True,
                dynamic_ncols=True)
    for ep in pbar:
        w_g = float(w_grnd_fn(ep, n_epochs))
        w_p = float(w_phys_fn(ep, n_epochs))

        tr = _epoch_loop(model, train_loader, rp, config, stage,
                         w_g, w_p, w_cons, optimizer=optimizer, train=True,
                         desc=f"  {phase_name}/train ep{ep+1}/{n_epochs}")
        va = _epoch_loop(model, val_loader,   rp, config, stage,
                         w_g, w_p, w_cons, optimizer=None,    train=False,
                         desc=f"  {phase_name}/val   ep{ep+1}/{n_epochs}")

        for k, v in tr.items(): h['train'][k].append(v)
        for k, v in va.items(): h['val'][k].append(v)

        pbar.set_postfix({
            'tr_total': f"{tr.get('total', float('nan')):.2e}",
            'va_total': f"{va.get('total', float('nan')):.2e}",
            'wg':       f"{w_g:.2f}",
            'wp':       f"{w_p:.2e}",
        })

        # Per-epoch summary line — only scalar components present in `tr`
        # are shown (state for forward, cons for inverse, etc.)
        parts = [f"tr_total={tr.get('total', float('nan')):.3e}",
                 f"va_total={va.get('total', float('nan')):.3e}"]
        comp_parts = []
        for k in ('state', 'grnd', 'cons', 'phys'):
            if k in tr:
                comp_parts.append(f"{k}={tr[k]:.3e}")
        if comp_parts:
            parts.append(" ".join(comp_parts))
        tqdm.write(
            f"[{stage}/{phase_name}] ep {next_ep + ep + 1:>3} | "
            + " | ".join(parts)
            + f" | wg={w_g:.2f} wp={w_p:.2e}"
        )

        # Per-component breakdowns — print whichever ones are present.
        # On forward stage:  state_per_comp, grnd_per_comp, (phys_per_comp if w_phys>0)
        # On inverse stage:  grnd_per_comp, cons_per_comp,  (phys_per_comp if w_phys>0)
        for key, label in _PER_COMP_DISPLAY_ORDER:
            if key in tr:
                tqdm.write(_format_per_comp(label, tr[key]))

        if stopper and stopper.step(va['total']):
            print(f"[ES {stage}/{phase_name}] stopping at epoch {next_ep + ep + 1}"
                  f" (best val={stopper.best:.3e})")
            break

    next_ep += (ep + 1)
    pbar.close()

    save_phase_checkpoint(model, optimizer, history,
                          config['ckpt_dir'], config['run_tag'],
                          stage, phase_name)
    return model, history, ema_scale, next_ep


# ============================================================
# Loss-weight schedules per phase
# ============================================================
def _lin(a, b, ep, n):
    return a + (b - a) * (ep + 1) / max(1, n)


def run_grounding(model, train_loader, val_loader, rp, config, stage,
                  history, ema_scale, next_ep):
    n = config[stage]['grounding_epochs']
    if n <= 0: return model, history, ema_scale, next_ep
    return _run_phase(
        model, train_loader, val_loader, rp, config, stage,
        history, ema_scale, next_ep, 'grounding', n,
        w_grnd_fn=lambda e, t: 1.0,
        w_phys_fn=lambda e, t: 0.0,
        early_stop_on='total',
    )


def run_phys_rampup(model, train_loader, val_loader, rp, config, stage,
                    history, ema_scale, next_ep):
    n = config[stage]['rampup_epochs']
    if n <= 0: return model, history, ema_scale, next_ep
    wmax = config['w_phys_max']
    return _run_phase(
        model, train_loader, val_loader, rp, config, stage,
        history, ema_scale, next_ep, 'phys_rampup', n,
        w_grnd_fn=lambda e, t: 1.0,
        w_phys_fn=lambda e, t: _lin(0.0, wmax, e, t),
    )


def run_overlap(model, train_loader, val_loader, rp, config, stage,
                history, ema_scale, next_ep):
    n = config[stage]['overlap_epochs']
    if n <= 0: return model, history, ema_scale, next_ep
    wmax = config['w_phys_max']
    return _run_phase(
        model, train_loader, val_loader, rp, config, stage,
        history, ema_scale, next_ep, 'overlap', n,
        w_grnd_fn=lambda e, t: 1.0,
        w_phys_fn=lambda e, t: wmax,
    )


def run_grnd_rampdown(model, train_loader, val_loader, rp, config, stage,
                      history, ema_scale, next_ep):
    n = config[stage]['rampdown_epochs']
    if n <= 0: return model, history, ema_scale, next_ep
    wmax = config['w_phys_max']
    return _run_phase(
        model, train_loader, val_loader, rp, config, stage,
        history, ema_scale, next_ep, 'grnd_rampdown', n,
        w_grnd_fn=lambda e, t: _lin(1.0, 0.0, e, t),
        w_phys_fn=lambda e, t: wmax,
    )


def run_physics(model, train_loader, val_loader, rp, config, stage,
                history, ema_scale, next_ep):
    n = config[stage]['physics_epochs']
    if n <= 0: return model, history, ema_scale, next_ep
    wmax = config['w_phys_max']
    return _run_phase(
        model, train_loader, val_loader, rp, config, stage,
        history, ema_scale, next_ep, 'physics', n,
        w_grnd_fn=lambda e, t: 0.0,
        w_phys_fn=lambda e, t: wmax,
        early_stop_on='total',
    )


# ============================================================
# L-BFGS refinement
# ============================================================
def _lbfgs_closure_factory(model, train_loader, rp, config, stage,
                           w_phys, call_log=None):
    device = config['device']
    grad_clip = config[stage]['grad_clip']
    trainable = _trainable_params(model, stage)
    w_cons = config['w_cons']

    def closure():
        for p in trainable:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        total_scalar = 0.0
        n = len(train_loader)
        nan_hit = False

        inner = tqdm(train_loader, desc=f"  lbfgs/{stage} closure",
                     leave=False, dynamic_ncols=True)
        for batch in inner:
            losses = compute_losses(model, batch, rp, config, stage, w_phys=w_phys)
            t = aggregate_total_loss(losses, stage,
                                     w_grnd=0.0, w_phys=w_phys, w_cons=w_cons)

            if not torch.isfinite(t):
                nan_hit = True
                inner.close()
                if call_log is not None:
                    call_log['n_calls'] += 1
                    call_log['loss'].append(float('nan'))
                    call_log['nan_hit'] = True
                print("[lbfgs] non-finite loss in batch; aborting closure")
                return torch.tensor(float('nan'), device=device)

            (t / n).backward()
            total_scalar += float(t.detach().float().item())

        inner.close()

        if nan_hit:
            return torch.tensor(float('nan'), device=device)

        avg_loss = total_scalar / max(1, n)
        gn = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)

        if call_log is not None:
            call_log['n_calls'] += 1
            call_log['loss'].append(avg_loss)
            call_log['grad_norm'].append(float(gn))

        return torch.tensor(avg_loss, device=device)

    return closure


def run_lbfgs_refine(model, train_loader, val_loader, rp: RobotParams,
                     config: Dict, stage: str, history: Dict, ema_scale=None):
    cfg = config[stage]
    if not cfg.get('use_lbfgs', True):
        print(f"[L-BFGS {stage}] disabled by config — skipping.")
        return model, history

    print(f"\n--- L-BFGS refinement ({stage}) | max_iter={cfg['lbfgs_max_iter']} ---")
    _set_stage_grad(model, stage)
    optimizer = optim.LBFGS(_trainable_params(model, stage),
                            lr=cfg['lbfgs_lr'],
                            max_iter=cfg['lbfgs_max_iter'],
                            history_size=10,
                            line_search_fn='strong_wolfe')

    call_log = {'n_calls': 0, 'loss': [], 'grad_norm': [], 'nan_hit': False}
    closure = _lbfgs_closure_factory(model, train_loader, rp, config, stage,
                                     w_phys=config['w_phys_max'],
                                     call_log=call_log)

    outer = tqdm(total=cfg['lbfgs_max_iter'], desc=f"L-BFGS {stage}",
                 leave=True, dynamic_ncols=True)
    try:
        loss = optimizer.step(closure)
        outer.update(min(call_log['n_calls'], cfg['lbfgs_max_iter']))
        outer.set_postfix({
            'final_loss': f"{float(loss):.3e}",
            'calls':      call_log['n_calls'],
            'gnorm':      f"{(call_log['grad_norm'][-1] if call_log['grad_norm'] else float('nan')):.2e}",
        })
        print(f"[{stage}/lbfgs] final loss = {float(loss):.4e} "
              f"(closure calls = {call_log['n_calls']})")
    except Exception as e:
        print(f"[{stage}/lbfgs] aborted: {e}")
    outer.close()

    history[stage]['train'].setdefault('lbfgs_closure_loss', []).extend(call_log['loss'])
    history[stage]['train'].setdefault('lbfgs_closure_grad_norm', []).extend(call_log['grad_norm'])

    save_phase_checkpoint(model, optimizer, history,
                          config['ckpt_dir'], config['run_tag'],
                          stage, 'lbfgs')
    return model, history


# ============================================================
# Top-level trainers
# ============================================================
def train_forward(rp: RobotParams, train_loader, val_loader, config: Dict,
                  geom: Optional[Geometry] = None,
                  model: Optional[MecanumPINN] = None
                  ) -> Tuple[MecanumPINN, Dict]:
    if geom is None:
        geom = make_geometry(rp)
    if model is None:
        fwd = MecanumForwardModel(**_forward_model_kwargs(config, rp)).to(config['device'])
        inv = MecanumInverseModel(
            **_inverse_model_kwargs(config, geom, use_H=True)
        ).to(config['device'])
        model = MecanumPINN(fwd, inv).to(config['device'])
        model = maybe_compile_pinn(model, config)

    history = {'forward': _new_history()}
    ema_scale = None
    next_ep = 0
    stage = 'forward'

    model, history, ema_scale, next_ep = run_grounding    (model, train_loader, val_loader, rp, config, stage, history, ema_scale, next_ep)
    model, history, ema_scale, next_ep = run_phys_rampup  (model, train_loader, val_loader, rp, config, stage, history, ema_scale, next_ep)
    model, history, ema_scale, next_ep = run_overlap      (model, train_loader, val_loader, rp, config, stage, history, ema_scale, next_ep)
    model, history, ema_scale, next_ep = run_grnd_rampdown(model, train_loader, val_loader, rp, config, stage, history, ema_scale, next_ep)
    model, history, ema_scale, next_ep = run_physics      (model, train_loader, val_loader, rp, config, stage, history, ema_scale, next_ep)

    if config['forward'].get('use_lbfgs', True):
        model, history = run_lbfgs_refine(model, train_loader, val_loader, rp,
                                          config, stage, history, ema_scale=ema_scale)
    return model, history


def train_inverse(rp: RobotParams, train_loader, val_loader, config: Dict,
                  model: MecanumPINN) -> Tuple[MecanumPINN, Dict]:
    history = {'inverse': _new_history()}
    ema_scale = None
    next_ep = 0
    stage = 'inverse'

    model, history, ema_scale, next_ep = run_grounding    (model, train_loader, val_loader, rp, config, stage, history, ema_scale, next_ep)
    model, history, ema_scale, next_ep = run_phys_rampup  (model, train_loader, val_loader, rp, config, stage, history, ema_scale, next_ep)
    model, history, ema_scale, next_ep = run_overlap      (model, train_loader, val_loader, rp, config, stage, history, ema_scale, next_ep)
    model, history, ema_scale, next_ep = run_grnd_rampdown(model, train_loader, val_loader, rp, config, stage, history, ema_scale, next_ep)
    model, history, ema_scale, next_ep = run_physics      (model, train_loader, val_loader, rp, config, stage, history, ema_scale, next_ep)

    if config['inverse'].get('use_lbfgs', True):
        model, history = run_lbfgs_refine(model, train_loader, val_loader, rp,
                                          config, stage, history, ema_scale=ema_scale)
    return model, history


def train_inverse_ablation(rp: RobotParams, train_loader, val_loader, config: Dict,
                           forward_trained: MecanumPINN, geom: Geometry):
    """Train two inverse variants on the SAME forward backbone for comparison.

    Returns (model_with_H, history_with_H, model_without_H, history_without_H).
    """
    fwd_state = forward_trained.forward_model.state_dict()

    def _build_pair(use_H: bool) -> MecanumPINN:
        fwd = MecanumForwardModel(**_forward_model_kwargs(config, rp)).to(config['device'])
        fwd.load_state_dict(fwd_state)

        inv = MecanumInverseModel(
            **_inverse_model_kwargs(config, geom, use_H=use_H)
        ).to(config['device'])
        model = MecanumPINN(fwd, inv).to(config['device'])
        return maybe_compile_pinn(model, config)

    print("=" * 60)
    print("Training inverse with use_H=True")
    print("=" * 60)
    config_with_H = copy.deepcopy(config)
    config_with_H['run_tag'] = config['run_tag'] + '_invH'
    m_H = _build_pair(use_H=True)
    m_H, h_H = train_inverse(rp, train_loader, val_loader, config_with_H, m_H)

    print("=" * 60)
    print("Training inverse with use_H=False")
    print("=" * 60)
    config_no_H = copy.deepcopy(config)
    config_no_H['run_tag'] = config['run_tag'] + '_invNoH'
    m_NH = _build_pair(use_H=False)
    m_NH, h_NH = train_inverse(rp, train_loader, val_loader, config_no_H, m_NH)

    return m_H, h_H, m_NH, h_NH
