"""Test-set evaluation, OOD evaluation, and (mu, chi) estimators.

Functions here operate on either DataLoaders (for batched mean-loss
evaluation) or single trajectory dicts (for per-trajectory plots and
parameter recovery).

For x-y trajectory rollouts with RMSE diagnostics, see
mecanum_pinn.trajectory_eval — that module is structured around the
single question of "how does the model's predicted path compare to the
sim's path in the world frame, with errors over time?"
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from . import data as _data
from .data import (MecanumTrajectoryDataset, control_max, force_max,
                   make_loader, state_max)
from .losses import compute_losses
from .models import MecanumPINN
from .physics import RobotParams
from .plotting import save_figure


# ============================================================
# Test-set / OOD aggregate evaluation
# ============================================================
def evaluate_on_test(model, test_loader, rp: RobotParams, config: Dict,
                     stage: str, desc: str = "test") -> Dict[str, float]:
    """Run trained model on test split, return mean component losses.

    Forces phys computation regardless of skip-phys logic by passing
    w_phys=config['w_phys_max'], so reported numbers are full-weight
    diagnostic values comparable across phases.
    """
    model.eval()
    accum = defaultdict(float)
    n = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=desc, leave=False, dynamic_ncols=True):
            losses = compute_losses(
                model, batch, rp, config, stage,
                w_phys=config['w_phys_max'],
            )
            for k, v in losses.items():
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    accum[k] += float(v.detach().float().item())
            n += 1

    out = {k: v / max(n, 1) for k, v in accum.items()}

    if stage == 'forward':
        out['total'] = (out.get('state', 0.0)
                        + out.get('grnd', 0.0)
                        + config['w_phys_max'] * out.get('phys', 0.0))
    else:
        out['total'] = (out.get('grnd', 0.0)
                        + config['w_phys_max'] * out.get('phys', 0.0)
                        + config['w_cons']     * out.get('cons', 0.0))
    return out


def evaluate_ood(model, ood_trajs: List[Dict[str, Any]], rp: RobotParams,
                 config: Dict, stage: str,
                 batch_size: Optional[int] = None,
                 num_workers: Optional[int] = None,
                 desc: str = "ood") -> Dict[str, float]:
    """Run test-loss metrics on the OOD pool. Same return shape as evaluate_on_test."""
    if len(ood_trajs) == 0:
        print("[evaluate_ood] empty OOD pool")
        return {}

    ood_ds = MecanumTrajectoryDataset(
        ood_trajs,
        seq_len=config['seq_len'],
        stride=config.get('stride', 1),
    )

    cfg = dict(config)
    if batch_size  is not None: cfg['batch_size']  = batch_size
    if num_workers is not None: cfg['num_workers'] = num_workers
    ood_loader = make_loader(ood_ds, cfg, shuffle=False, drop_last=False)

    return evaluate_on_test(model, ood_loader, rp, config, stage, desc=desc)


# ============================================================
# Per-trajectory state + force plot (the original predictions plot)
# ============================================================
def plot_test_trajectory_predictions(model: MecanumPINN, trajectories,
                                     rp: RobotParams, config: Dict,
                                     num_cases: int = 4,
                                     time_window: float = 6.0,
                                     seed: int = 0,
                                     stratify: bool = True,
                                     show_inverse: bool = True):
    """Pick trajectories from the test pool, plot pred-vs-truth state and forces+Mz.

    Plots time-series of platform velocities and per-wheel Fx, Mz. The
    NEW world-frame x-y mapping with RMSE-vs-time lives in
    mecanum_pinn.trajectory_eval — this function intentionally keeps the
    body-frame, all-component view it has always had.

    Note: S_pred from forward_path is 7-dim (V + ω). θ is consumed as
    input by the model but not predicted. This function only plots the
    velocity components and forces, none of which need θ from S_pred —
    θ is still read from the trajectory data for the inverse model's
    sin/cos features in `show_inverse=True` mode.
    """
    model.eval()
    rng = np.random.default_rng(seed)

    if stratify:
        by_cell = defaultdict(list)
        for i, t in enumerate(trajectories):
            by_cell[(t['motion'], round(t['chi'], 4))].append(i)
        cells = sorted(by_cell.keys())
        rng.shuffle(cells)
        picks: List[int] = []
        for cell in cells:
            if len(picks) >= num_cases:
                break
            idx_pool = by_cell[cell]
            picks.append(int(rng.choice(idx_pool)))
        if len(picks) < num_cases:
            extras = rng.choice(len(trajectories),
                                size=num_cases - len(picks), replace=False)
            picks.extend(int(e) for e in extras if int(e) not in picks)
    else:
        picks = rng.choice(len(trajectories),
                           size=min(num_cases, len(trajectories)),
                           replace=False).tolist()

    for idx in picks:
        traj = trajectories[idx]
        T_full = traj['times'].squeeze()
        dt = float(np.median(np.diff(T_full)))
        n_window = max(1, int(time_window / dt))
        L = min(n_window, traj['states'].shape[0] - 2)

        S_np  = (traj['states'][:L]    / state_max).astype(np.float32)
        U_np  = (traj['controls'][:L]  / control_max).astype(np.float32)
        Sn_np = (traj['states'][1:L+1] / state_max).astype(np.float32)
        Tt    = traj['times'][:L].astype(np.float32)

        S_t  = torch.tensor(S_np,  device=config['device']).unsqueeze(0)
        U_t  = torch.tensor(U_np,  device=config['device']).unsqueeze(0)
        Sn_t = torch.tensor(Sn_np, device=config['device']).unsqueeze(0)
        T_t  = torch.tensor(Tt,    device=config['device']).unsqueeze(0)
        mu_t = torch.tensor([traj['mu']],  dtype=torch.float32, device=config['device'])
        ch_t = torch.tensor([traj['chi']], dtype=torch.float32, device=config['device'])

        with torch.no_grad():
            S_pred, F_fwd, H_t = model.forward_path(
                S_t, U_t, T_t.requires_grad_(False),
                mu_t, ch_t, rp.N_per_roller,
            )
            F_inv = None
            if show_inverse:
                H_d = H_t.detach() if model.inverse_model.use_H else None
                F_inv = model.inverse_path(
                    S_curr=S_t, S_next=Sn_t, U=U_t,
                    N_per_wheel=rp.N_per_roller, H_detached=H_d,
                )

        # S_pred is 7-dim (V + ω) — θ is not predicted by the model.
        # Un-normalize against the first 7 entries of state_max.
        S_pred_np = (S_pred.squeeze(0).cpu().numpy() * state_max[:7])
        F_pred_np = (F_fwd.squeeze(0).cpu().numpy() * force_max)
        F_inv_np  = (F_inv.squeeze(0).cpu().numpy() * force_max) if F_inv is not None else None
        F_true_np = traj['forces'][:L]

        n_rows = 4 if show_inverse else 3
        fig, axs = plt.subplots(n_rows, 1, figsize=(10, 2.2 * n_rows + 0.5),
                                sharex=True)
        fig.suptitle(
            f"{traj['name']}\n"
            f"motion={traj['motion']}  μ={traj['mu']:.3f}  χ={traj['chi']:.4f}",
            fontsize=10,
        )

        axs[0].plot(Tt, traj['states'][:L, 0], 'k-',  label='Vx (sim)')
        axs[0].plot(Tt, S_pred_np[:, 0],       'r--', label='Vx (pred)')
        axs[0].plot(Tt, traj['states'][:L, 1], 'k:',  label='Vy (sim)')
        axs[0].plot(Tt, S_pred_np[:, 1],       'b--', label='Vy (pred)')
        axs[0].plot(Tt, traj['states'][:L, 2], color='gray',  ls='-',  label=r'$\dot\psi$ (sim)')
        axs[0].plot(Tt, S_pred_np[:, 2],       color='green', ls='--', label=r'$\dot\psi$ (pred)')
        axs[0].set_ylabel('platform vel.')
        axs[0].legend(loc='best', fontsize=7, ncol=3)
        axs[0].grid(True, alpha=0.3)

        for i in range(4):
            line, = axs[1].plot(Tt, F_true_np[:, i],   '-',  alpha=0.5)
            axs[1].plot(Tt, F_pred_np[:, i],          '--', color=line.get_color(),
                        label=f'wheel {i+1}')
        axs[1].set_ylabel('Fx (N)')
        axs[1].legend(loc='best', fontsize=7, ncol=4)
        axs[1].grid(True, alpha=0.3)

        for i in range(4):
            line, = axs[2].plot(Tt, F_true_np[:, 8 + i], '-',  alpha=0.5)
            axs[2].plot(Tt, F_pred_np[:, 8 + i],         '--', color=line.get_color(),
                        label=f'wheel {i+1}')
        axs[2].set_ylabel('Mz (N·m)')
        axs[2].legend(loc='best', fontsize=7, ncol=4)
        axs[2].grid(True, alpha=0.3)

        if show_inverse:
            for i in range(4):
                line, = axs[3].plot(Tt, F_pred_np[:, i], '--', alpha=0.6)
                axs[3].plot(Tt, F_inv_np[:, i],          ':',  color=line.get_color(),
                            label=f'wheel {i+1}')
            axs[3].set_ylabel('Fx: fwd (--) vs inv (:)')
            axs[3].legend(loc='best', fontsize=7, ncol=4)
            axs[3].grid(True, alpha=0.3)

        axs[-1].set_xlabel('time (s)')
        save_figure(label=f"trajpred_{traj['name']}")


# ============================================================
# Per-trajectory mu/chi recovery (least-squares against F_inv)
# ============================================================
def estimate_mu_chi(model: MecanumPINN, traj: Dict, rp: RobotParams,
                    config: Dict) -> Tuple[float, float]:
    """Recover (mu, chi) for a single trajectory by least squares against F_inv."""
    model.eval()
    L = min(traj['states'].shape[0] - 2, 500)
    S  = (traj['states'][:L]   / state_max).astype(np.float32)
    U  = (traj['controls'][:L] / control_max).astype(np.float32)
    Fs = traj['forces'][:L].astype(np.float32)
    Tt = traj['times'][:L].astype(np.float32)

    S_t  = torch.tensor(S, device=config['device']).unsqueeze(0)
    U_t  = torch.tensor(U, device=config['device']).unsqueeze(0)
    S_n  = torch.tensor((traj['states'][1:L+1] / state_max).astype(np.float32),
                        device=config['device']).unsqueeze(0)
    T_t  = torch.tensor(Tt, device=config['device']).unsqueeze(0)
    mu_t = torch.tensor([traj['mu']], dtype=torch.float32, device=config['device'])
    ch_t = torch.tensor([traj['chi']], dtype=torch.float32, device=config['device'])

    with torch.no_grad():
        _, _, H_t = model.forward_path(S_t, U_t, T_t, mu_t, ch_t, rp.N_per_roller)
        H_d = H_t.detach() if model.inverse_model.use_H else None
        F_inv = model.inverse_path(S_curr=S_t, S_next=S_n, U=U_t,
                                   N_per_wheel=rp.N_per_roller, H_detached=H_d)

    F_inv_np = F_inv.squeeze(0).cpu().numpy()
    F_true   = Fs / force_max

    Fxy_inv  = F_inv_np[:, :8].flatten()
    Fxy_true = F_true[:,   :8].flatten()
    mz_inv   = F_inv_np[:, 8:12].flatten()
    mz_true  = F_true[:,   8:12].flatten()

    alpha_F = float(np.dot(Fxy_inv, Fxy_true) / (np.dot(Fxy_inv, Fxy_inv) + 1e-9))
    mu_recovered  = traj['mu'] * alpha_F

    if np.linalg.norm(mz_inv) > 1e-6:
        alpha_M = float(np.dot(mz_inv, mz_true) / (np.dot(mz_inv, mz_inv) + 1e-9))
        chi_recovered = traj['chi'] * math.sqrt(max(alpha_M / max(alpha_F, 1e-6), 1e-6))
    else:
        chi_recovered = traj['chi']

    return mu_recovered, chi_recovered


def estimate_and_plot_mu_chi(model: MecanumPINN, test_trajs, rp: RobotParams,
                             config: Dict, window: int = 100,
                             num_cases: int = 3,
                             seed: Optional[int] = None,
                             tag: str = "muchi"):
    """Per-wheel mu and chi rolling estimates from F_inv across multiple test trajs."""
    rng = np.random.RandomState(seed)
    model.eval()
    n = len(test_trajs)
    if n == 0:
        print("[estimate_and_plot_mu_chi] empty test set")
        return {}

    picks = rng.choice(n, size=min(num_cases, n), replace=False)
    N_cpu = rp.N_per_roller.cpu()

    mu_raw  = {i: [] for i in range(4)}
    chi_raw = {i: [] for i in range(4)}
    true_mu_log, true_chi_log = [], []

    for idx in picks:
        traj = test_trajs[idx]
        S_np  = (traj['states']   / state_max).astype(np.float32)
        U_np  = (traj['controls'] / control_max).astype(np.float32)
        Sn_np = np.roll(S_np, shift=-1, axis=0)
        Sn_np[-1] = S_np[-1]
        Tt_np = traj['times'].astype(np.float32)

        S_t  = torch.tensor(S_np,  device=config['device']).unsqueeze(0)
        U_t  = torch.tensor(U_np,  device=config['device']).unsqueeze(0)
        Sn_t = torch.tensor(Sn_np, device=config['device']).unsqueeze(0)
        T_t  = torch.tensor(Tt_np, device=config['device']).unsqueeze(0)

        mu_val  = float(traj['mu'])
        chi_val = float(traj['chi'])
        true_mu_log.append(mu_val)
        true_chi_log.append(chi_val)

        mu_t  = torch.tensor([mu_val],  dtype=torch.float32, device=config['device'])
        chi_t = torch.tensor([chi_val], dtype=torch.float32, device=config['device'])

        with torch.no_grad():
            _, F_fwd, H_t = model.forward_path(S_t, U_t, T_t, mu_t, chi_t,
                                               rp.N_per_roller)
            H_d = H_t.detach() if model.inverse_model.use_H else None
            F_inv_scaled = model.inverse_path(
                S_curr=S_t, S_next=Sn_t, U=U_t,
                N_per_wheel=rp.N_per_roller, H_detached=H_d,
            )

        F_pred = F_inv_scaled[0].cpu() * torch.tensor(force_max)
        Fx = F_pred[:, 0:4]
        Fy = F_pred[:, 4:8]
        Mz = F_pred[:, 8:12].abs()

        F_mag    = torch.sqrt(Fx**2 + Fy**2)
        mu_step  = F_mag / N_cpu.unsqueeze(0)
        chi_step = torch.sqrt(Mz / (mu_val * N_cpu.unsqueeze(0) + 1e-9))

        for w in range(4):
            mu_raw[w].extend(mu_step[:, w].numpy().tolist())
            chi_raw[w].extend(chi_step[:, w].numpy().tolist())

    fig, (ax_mu, ax_chi) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange']

    for w in range(4):
        s_mu  = pd.Series(mu_raw[w]).rolling(window=window, min_periods=1).mean()
        s_chi = pd.Series(chi_raw[w]).rolling(window=window, min_periods=1).mean()
        ax_mu.plot(s_mu,   color=colors[w], lw=1.6, label=f'wheel {w+1}')
        ax_chi.plot(s_chi, color=colors[w], lw=1.6, label=f'wheel {w+1}')

    true_mu_mean  = float(np.mean(true_mu_log))  if true_mu_log  else 0.0
    true_chi_mean = float(np.mean(true_chi_log)) if true_chi_log else 0.0
    ax_mu .axhline(true_mu_mean,  color='k', lw=0.9, ls=':',
                   label=f'true mu = {true_mu_mean:.3f}')
    ax_chi.axhline(true_chi_mean, color='k', lw=0.9, ls=':',
                   label=f'true chi = {true_chi_mean:.4f}')

    ax_mu.set_ylabel(r'$\hat\mu$ (rolling mean)')
    ax_mu.set_title(f'Per-wheel mu estimation (rolling mean, window={window})')
    ax_mu.grid(True, linestyle='--', alpha=0.4); ax_mu.legend(loc='best', fontsize=8)

    ax_chi.set_ylabel(r'$\hat\chi$ (rolling mean, m)')
    ax_chi.set_xlabel('Cumulative test step')
    ax_chi.set_title(f'Per-wheel chi estimation (rolling mean, window={window})')
    ax_chi.grid(True, linestyle='--', alpha=0.4); ax_chi.legend(loc='best', fontsize=8)

    save_figure(label=tag)

    print(f"\n{'Wheel':<8}{'mu mean':>12}{'mu std':>12}{'chi mean':>12}{'chi std':>12}")
    for w in range(4):
        a, b = np.array(mu_raw[w]), np.array(chi_raw[w])
        print(f"{w+1:<8}{a.mean():>12.4f}{a.std():>12.4f}{b.mean():>12.5f}{b.std():>12.5f}")

    return {'mu': mu_raw, 'chi': chi_raw,
            'true_mu_mean': true_mu_mean, 'true_chi_mean': true_chi_mean}
