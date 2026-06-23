"""Trajectory-level evaluation in the world frame.

Where evaluation.plot_test_trajectory_predictions answers "do the model's
body-frame velocities and per-wheel forces match the sim time-series",
this module answers a different question:

    Starting from an initial condition somewhere in a trajectory and
    integrating forward for a chosen time window, how close does the
    predicted (x, y, psi_pos) path stay to the sim's path? And how do
    the inverse-model forces compare against the forward-model forces
    and the sim ground truth?

Two rollout modes
-----------------

`rollout='teacher_forced'` (default):
    The simulator's S_curr is fed to the model at every step in the
    window. The model produces a one-step-ahead prediction at every
    position, but never uses its own predictions as input. θ comes
    from the sim at every step — kinematic integration not needed.
    This tests "is the one-step Markov predictor accurate?".

`rollout='autoregressive'`:
    The model's own predictions are fed back as inputs. After a short
    warmup of length `seq_len_init` (default = config['seq_len'] = 5)
    seeded from sim, every subsequent step uses:
      (V_next, ω_next) from the model + θ_next from kinematic
                                          integration
    Errors compound in this mode. This tests "is the model useful
    over a multi-step horizon" — the same regime as MPC rollout.

Two functions:

    evaluate_trajectory_window(model, traj, rp, config, *,
                                rollout='teacher_forced',
                                seq_len_init=None, ...) -> dict
        Runs the model over a window of length `window_seconds` starting
        at `start_idx` of `traj`. Integrates body-frame velocities
        (Vx, Vy, psi_dot) into world-frame (x, y, psi) for BOTH the sim
        and the prediction using the same forward-Euler scheme, so any
        integration drift cancels in the comparison. Returns a result
        dict with paired arrays + RMSE summaries + (for autoregressive)
        the warmup-boundary index.

    plot_trajectory_window(result, ...)
        Two figures:
          1. x-y plane figure with predicted vs sim path, start/end
             markers, an error-over-time inset for x, y, psi, and (in
             autoregressive mode) a marker showing where the warmup
             ended and autoregressive rollout began.
          2. Force comparison figure: sim Fx, fwd-model Fx, inv-model Fx
             per wheel, then the same for Mz.

The integration is one-step Euler for psi, then Euler for (x, y) using
the integrated psi sequence. With dt~5ms this is more than accurate
enough; even if it weren't, identical scheme on both paths makes the
COMPARISON exact regardless of integration error.

Note on θ (wheel angles)
------------------------
The forward model's S_pred is 7-dim (V + ω only) — θ is consumed as
input but not predicted. In `teacher_forced` mode every step's θ
comes from sim, so kinematic integration never enters the loop. In
`autoregressive` mode θ is integrated between steps as:

    theta_curr_folded = np.arctan2(np.sin(12*theta_curr),
                                   np.cos(12*theta_curr)) / 12.0
    theta_next        = np.arctan2(np.sin(12*(theta_curr_folded + omega_phys*dt)),
                                   np.cos(12*(theta_curr_folded + omega_phys*dt))) / 12.0

with `omega_phys` in physical rad/s (de-normalized by `state_max[3:7]`).
This is exact for Euler given that the simulator's θ propagation is θ̇ = ω;
the Euler-vs-RK4 gap is O(ω̇·dt²/2), well under a milliradian per step.

Indexing convention
-------------------
Both modes produce length-L `pred_*` arrays where index t aligns to sim
time index `start_idx + t`. In teacher-forced mode this means
`pred_states_full[t]` for `t < L-1` is the model's prediction made from
sim input at time `start_idx + t - 1` for the state at time
`start_idx + t` (one-step shift baked in cleanly). The very first slot
`pred_states_full[0]` is identical to `traj['states'][start_idx]` since
no prediction has been made yet. In autoregressive mode the warmup
slots `[0, seq_len_init)` are also sim, and `[seq_len_init, L)` are
autoregressive predictions of state at the same index.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import control_max, force_max, state_max
from .models import MecanumPINN
from .physics import RobotParams
from .plotting import save_figure


# ============================================================
# World-frame integration
# ============================================================
def _integrate_world_frame(Vx: np.ndarray, Vy: np.ndarray,
                           psi_dot: np.ndarray, dt: np.ndarray,
                           x0: float = 0.0, y0: float = 0.0,
                           psi0: float = 0.0
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate body-frame (Vx, Vy, psi_dot) into world-frame (x, y, psi).

    Forward Euler. Same scheme is used on sim and prediction so any
    integration drift is identical between the two -- the COMPARISON is
    exact regardless of the integrator's absolute accuracy.

    Parameters
    ----------
    Vx, Vy, psi_dot : 1-D arrays of length L
    dt              : 1-D array of step sizes, length L (last entry unused)
    x0, y0, psi0    : initial pose in the world frame

    Returns
    -------
    x, y, psi : 1-D arrays of length L
    """
    L = len(Vx)
    x   = np.empty(L, dtype=np.float64)
    y   = np.empty(L, dtype=np.float64)
    psi = np.empty(L, dtype=np.float64)
    x[0], y[0], psi[0] = x0, y0, psi0

    for k in range(L - 1):
        c, s = np.cos(psi[k]), np.sin(psi[k])
        # body-frame velocity rotated into world frame
        vx_w = Vx[k] * c - Vy[k] * s
        vy_w = Vx[k] * s + Vy[k] * c
        x[k + 1]   = x[k]   + vx_w        * dt[k]
        y[k + 1]   = y[k]   + vy_w        * dt[k]
        psi[k + 1] = psi[k] + psi_dot[k]  * dt[k]
    return x, y, psi


# ============================================================
# Autoregressive helper — fold θ to ±π/12
# ============================================================
def _fold_theta(theta_unfolded: np.ndarray) -> np.ndarray:
    """Wrap θ (rad) to the ±π/12 fundamental domain via atan2(sin12, cos12)/12.

    Idempotent on already-folded values (modulo float roundoff). Used
    inside the autoregressive loop where each kinematic update can push
    θ outside ±π/12 and we want a compact representation matching the
    data loader's convention.
    """
    return np.arctan2(
        np.sin(12.0 * theta_unfolded),
        np.cos(12.0 * theta_unfolded),
    ) / 12.0


# ============================================================
# Window evaluation
# ============================================================
def evaluate_trajectory_window(
    model: MecanumPINN,
    traj: Dict[str, Any],
    rp: RobotParams,
    config: Dict[str, Any],
    *,
    start_idx: int = 0,
    window_seconds: float = 6.0,
    initial_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rollout: str = 'autoregressive',
    seq_len_init: Optional[int] = None,
) -> Dict[str, Any]:
    """Run forward + inverse over a window starting at `start_idx`.

    Parameters
    ----------
    model         : trained MecanumPINN (forward and optionally inverse)
    traj          : a single trajectory dict (from load_all_arrow_trajectories)
    rp            : finalized RobotParams
    config        : run config (provides device + dimensions)
    start_idx     : index in traj['states'] to use as the initial condition
    window_seconds: length of the rollout window in seconds
    initial_pose  : (x0, y0, psi0) world-frame anchor for both paths
    rollout       : 'teacher_forced'  or 'autoregressive' (default).
                    Teacher-forced feeds sim S_curr at every step; θ is
                    always from sim, no kinematic integration. Autoregressive
                    feeds the model's own previous prediction (after a
                    short warmup); θ is kinematically integrated from
                    predicted ω.
    seq_len_init  : Only used when rollout='autoregressive'. Length of
                    the sim-seeded warmup window. Defaults to
                    config.get('seq_len', 5). Must satisfy
                    `seq_len_init <= window_steps - 1`.

    Returns
    -------
    result : dict with the following keys
        # metadata
        'name', 'mu', 'chi', 'motion', 'start_idx', 'window_seconds',
        'window_steps' (= L), 'dt_mean', 'rollout', 'seq_len_init'
        # time + paired body-frame velocity tracks (length L)
        'times'   : (L,) absolute time
        'rel_t'   : (L,) time since start of window
        'true_Vx', 'true_Vy', 'true_psi_dot' : (L,) sim velocities
        'pred_Vx', 'pred_Vy', 'pred_psi_dot' : (L,) model velocities
        # world-frame pose, integrated identically for both
        'true_x', 'true_y', 'true_psi'        : (L,)
        'pred_x', 'pred_y', 'pred_psi'        : (L,)
        # RMSE summaries (scalars)
        'rmse_x', 'rmse_y', 'rmse_psi', 'rmse_xy_2d'
        # cumulative-RMSE-over-time for the time plot (length L)
        'cum_rmse_x_t', 'cum_rmse_y_t', 'cum_rmse_psi_t'
        # forces in physical units (N or N·m)
        'F_true' : (L, 12)
        'F_fwd'  : (L, 12)  forward-model forces. In autoregressive
                            mode, the last entry may be NaN since the
                            very last step's transition isn't predicted.
        'F_inv'  : (L, 12) or None  inverse-model forces (None if model
                            has no usable inverse). Computed in
                            teacher-forced style against sim states
                            regardless of rollout mode — useful as a
                            clean observer reading.
        # autoregressive-specific
        'warmup_end_idx' : int. Only meaningful in autoregressive mode
                            (= seq_len_init). In teacher-forced mode, 0.
    """
    if rollout not in ('teacher_forced', 'autoregressive'):
        raise ValueError(f"rollout must be 'teacher_forced' or 'autoregressive', "
                         f"got {rollout!r}")

    device = config['device']
    model.eval()

    # Resolve window length in steps from the trajectory's median dt.
    T_full = traj['times'].squeeze()
    dt_med = float(np.median(np.diff(T_full)))
    n_window = max(1, int(round(window_seconds / dt_med)))

    n_total = traj['states'].shape[0]
    if start_idx < 0 or start_idx >= n_total - 2:
        raise ValueError(f"start_idx={start_idx} out of range for trajectory "
                         f"of length {n_total}")
    end = min(start_idx + n_window, n_total - 1)
    L = end - start_idx
    if L < 2:
        raise ValueError(f"window too short: L={L}; pick larger window_seconds "
                         f"or smaller start_idx")

    # Slice + scale.
    S_np  = (traj['states'][start_idx:end]    / state_max).astype(np.float32)
    U_np  = (traj['controls'][start_idx:end]  / control_max).astype(np.float32)
    Sn_np = (traj['states'][start_idx + 1:end + 1] / state_max).astype(np.float32)
    T_np  = traj['times'][start_idx:end].astype(np.float32)

    # Build μ, χ tensors once.
    mu_t = torch.tensor([traj['mu']],  dtype=torch.float32, device=device)
    ch_t = torch.tensor([traj['chi']], dtype=torch.float32, device=device)

    # =====================================================================
    # Inverse model forces — computed once from sim states (teacher-forced
    # style, regardless of rollout mode). The inverse is an observer; we
    # always want its reading of the sim transitions.
    # =====================================================================
    S_t_sim  = torch.tensor(S_np,  device=device).unsqueeze(0)
    U_t_sim  = torch.tensor(U_np,  device=device).unsqueeze(0)
    Sn_t_sim = torch.tensor(Sn_np, device=device).unsqueeze(0)
    T_t_sim  = torch.tensor(T_np,  device=device).unsqueeze(0)

    with torch.no_grad():
        # Forward pass on sim states gives H for the inverse model and
        # also gives the teacher-forced F_fwd we'll use if rollout='teacher_forced'.
        S_pred_sim, F_fwd_sim, H_t_sim = model.forward_path(
            S_t_sim, U_t_sim, T_t_sim, mu_t, ch_t, rp.N_per_roller,
        )
        try:
            H_d = H_t_sim.detach() if model.inverse_model.use_H else None
            F_inv = model.inverse_path(
                S_curr=S_t_sim, S_next=Sn_t_sim, U=U_t_sim,
                N_per_wheel=rp.N_per_roller, H_detached=H_d,
            )
            F_inv_np = (F_inv.squeeze(0).cpu().numpy() * force_max)
        except Exception as e:
            print(f"[trajectory_eval] inverse_path unavailable ({e!r}); "
                  f"continuing with F_inv=None")
            F_inv_np = None

    # =====================================================================
    # Branch on rollout mode for the forward path
    # =====================================================================
    # Storage: pred_states_full is 11-dim including θ so the
    # autoregressive loop can re-feed it as input; for teacher-forced
    # we copy sim θ into the relevant slots.
    pred_states_full = np.empty((L, 11), dtype=np.float32)         # normalized
    F_fwd_seq        = np.full((L, 12), np.nan, dtype=np.float32)  # normalized

    # First slot is always the sim state at start_idx (no prediction yet).
    pred_states_full[0] = S_np[0]

    warmup_end_idx = 0

    if rollout == 'teacher_forced':
        # Single forward_path call on the entire window from sim. S_pred
        # at position t is the model's prediction for time index
        # start_idx + t + 1. We store at pred_states_full[t + 1] so that
        # pred_states_full[t] aligns with sim time index start_idx + t.
        S_pred_np_local = S_pred_sim.squeeze(0).cpu().numpy()      # (L, 7)
        F_fwd_np_local  = F_fwd_sim.squeeze(0).cpu().numpy()       # (L, 12)

        # pred_states_full[1:L]: model predictions for sim indices
        # [start_idx+1, ..., start_idx+L-1] from S_pred_np_local[0:L-1].
        # (We discard S_pred_np_local[-1] which would be a prediction
        # for start_idx+L, outside the comparison window.)
        pred_states_full[1:, :7] = S_pred_np_local[:-1]
        # θ in teacher-forced is from sim at the matching index.
        pred_states_full[1:, 7:11] = S_np[1:, 7:11]

        # Forces: F_fwd_np_local[t] is the model's force prediction at
        # input position t = sim index start_idx + t. Same index alignment.
        F_fwd_seq[:] = F_fwd_np_local

    else:  # rollout == 'autoregressive'
        if seq_len_init is None:
            seq_len_init = config.get('seq_len', 5)
        seq_len_init = int(seq_len_init)
        if seq_len_init < 1:
            raise ValueError(f"seq_len_init must be >= 1, got {seq_len_init}")
        if seq_len_init >= L:
            raise ValueError(
                f"seq_len_init ({seq_len_init}) must be < window_steps "
                f"({L}); pick a longer window_seconds or shorter warmup."
            )
        warmup_end_idx = seq_len_init

        # Warmup with sim states (folded θ via the same atan2/12 trick
        # that the data loader uses).
        pred_states_full[:seq_len_init] = S_np[:seq_len_init]
        pred_states_full[:seq_len_init, 7:11] = _fold_theta(
            pred_states_full[:seq_len_init, 7:11]
        )

        # Times array (raw, not normalized — used for dt).
        times_arr = T_np  # shape (L, 1)
        if times_arr.ndim == 1:
            times_arr = times_arr.reshape(-1, 1)
        # Normalized controls (matching what model expects).
        U_norm = U_np

        # state_max as numpy (already a module-level constant).
        sm_omega = state_max[3:7]   # (4,) — to de-normalize ω before kinematic step
        # state_max[7:11] is [1, 1, 1, 1], so θ in 'normalized' space equals
        # θ in physical space — no scaling needed.

        for t in range(seq_len_init, L):
            # Build window over input indices [t - seq_len_init, t).
            ws, we = t - seq_len_init, t
            S_win = pred_states_full[ws:we]                      # (seq_len_init, 11)
            U_win = U_norm[ws:we]                                # (seq_len_init, 4)
            T_win = times_arr[ws:we]                             # (seq_len_init, 1)

            S_t = torch.tensor(S_win, device=device).unsqueeze(0)
            U_t = torch.tensor(U_win, device=device).unsqueeze(0)
            T_t = torch.tensor(T_win, device=device).unsqueeze(0)

            with torch.no_grad():
                S_pred_w, F_fwd_w, _ = model.forward_path(
                    S_t, U_t, T_t, mu_t, ch_t, rp.N_per_roller,
                )
            # S_pred_w shape (1, seq_len_init, 7): last position is the
            # prediction for sim index t.
            V_next_norm     = S_pred_w[0, -1, :3].cpu().numpy()
            omega_next_norm = S_pred_w[0, -1, 3:7].cpu().numpy()
            F_fwd_last_norm = F_fwd_w[0, -1].cpu().numpy()        # (12,)

            # On the FIRST iteration also capture forces for the warmup
            # positions [0, seq_len_init - 1] — these are all clean
            # teacher-forced predictions from sim warmup input.
            if t == seq_len_init:
                F_fwd_warmup = F_fwd_w[0].cpu().numpy()           # (seq_len_init, 12)
                # F_fwd_warmup[k] is the force prediction at sim index
                # start_idx + k (input position k in the warmup window).
                F_fwd_seq[:seq_len_init] = F_fwd_warmup

            # Kinematic θ integration: θ_next = wrap(θ_prev + ω·dt).
            # ω is normalized; de-normalize before applying.
            omega_next_phys = omega_next_norm * sm_omega           # rad/s
            theta_prev = pred_states_full[t - 1, 7:11]             # already physical
            dt_t = float(times_arr[t, 0] - times_arr[t - 1, 0])
            theta_next_unfolded = theta_prev + omega_next_phys * dt_t
            theta_next_folded   = _fold_theta(theta_next_unfolded)

            # Store predicted state at index t (index-aligned with sim).
            pred_states_full[t, :3]    = V_next_norm
            pred_states_full[t, 3:7]   = omega_next_norm
            pred_states_full[t, 7:11]  = theta_next_folded

            # Store force at the same index t-1 + 1 = t... actually:
            # F_fwd_last_norm is the force prediction at the LAST INPUT
            # position of the window, which is sim index t - 1. It
            # corresponds to the forces that drove the transition from
            # state t-1 to state t. We store it at index t-1 (the
            # "during transition" slot), overwriting the warmup value
            # only for t == seq_len_init (which is fine — same value
            # since F_fwd_w[..., -1, :] for first iter equals
            # F_fwd_warmup[-1]).
            #
            # Wait — for t > seq_len_init, F_fwd at index t-1 was
            # already set by the PREVIOUS iteration (as its F_fwd_last).
            # Re-running on a different window (shifted by 1) would give
            # a DIFFERENT prediction at sim index t-1 (since the GRU's
            # hidden-state warmup is now different). For the diagnostic
            # plot we use the FIRST prediction at each sim index (made
            # at the time we autoregressively reached it), so we keep
            # the value already there.
            if np.isnan(F_fwd_seq[t - 1, 0]):
                F_fwd_seq[t - 1] = F_fwd_last_norm

        # Note: F_fwd_seq[L - 1] is NEVER assigned in autoregressive
        # mode (would require a step beyond the window). Leave as NaN.

    # =====================================================================
    # De-normalize predicted states and forces for downstream use
    # =====================================================================
    pred_states_phys = pred_states_full * state_max                # (L, 11)
    F_fwd_np = F_fwd_seq * force_max                                # (L, 12), may have NaN

    # =====================================================================
    # Body-frame velocity tracks for plot + integration
    # =====================================================================
    true_states = traj['states'][start_idx:end]
    true_Vx, true_Vy, true_psi_dot = (true_states[:, 0],
                                      true_states[:, 1],
                                      true_states[:, 2])
    pred_Vx, pred_Vy, pred_psi_dot = (pred_states_phys[:, 0],
                                      pred_states_phys[:, 1],
                                      pred_states_phys[:, 2])

    # Per-step dt for integration; pad final entry to keep length L.
    times_abs = T_np.flatten()
    dt_arr = np.empty(L, dtype=np.float64)
    dt_arr[:-1] = np.diff(times_abs)
    dt_arr[-1]  = dt_arr[-2] if L >= 2 else dt_med

    x0, y0, psi0 = initial_pose
    true_x, true_y, true_psi = _integrate_world_frame(
        true_Vx, true_Vy, true_psi_dot, dt_arr, x0, y0, psi0,
    )
    pred_x, pred_y, pred_psi = _integrate_world_frame(
        pred_Vx, pred_Vy, pred_psi_dot, dt_arr, x0, y0, psi0,
    )

    # Errors and cumulative RMSE for the time-domain plot.
    err_x = pred_x - true_x
    err_y = pred_y - true_y
    err_psi = pred_psi - true_psi

    cum_rmse_x   = np.sqrt(np.cumsum(err_x   ** 2) / np.arange(1, L + 1))
    cum_rmse_y   = np.sqrt(np.cumsum(err_y   ** 2) / np.arange(1, L + 1))
    cum_rmse_psi = np.sqrt(np.cumsum(err_psi ** 2) / np.arange(1, L + 1))

    rmse_x   = float(np.sqrt(np.mean(err_x   ** 2)))
    rmse_y   = float(np.sqrt(np.mean(err_y   ** 2)))
    rmse_psi = float(np.sqrt(np.mean(err_psi ** 2)))
    rmse_xy_2d = float(np.sqrt(np.mean(err_x ** 2 + err_y ** 2)))

    F_true_np = traj['forces'][start_idx:end]

    return {
        # metadata
        'name': traj['name'], 'mu': float(traj['mu']),
        'chi': float(traj['chi']), 'motion': traj['motion'],
        'start_idx': int(start_idx), 'window_seconds': float(window_seconds),
        'window_steps': int(L), 'dt_mean': float(np.mean(dt_arr[:-1])),
        'rollout': rollout, 'seq_len_init': int(warmup_end_idx),
        # time
        'times': times_abs, 'rel_t': times_abs - times_abs[0],
        # body-frame velocities
        'true_Vx': true_Vx, 'true_Vy': true_Vy, 'true_psi_dot': true_psi_dot,
        'pred_Vx': pred_Vx, 'pred_Vy': pred_Vy, 'pred_psi_dot': pred_psi_dot,
        # world-frame pose
        'true_x': true_x, 'true_y': true_y, 'true_psi': true_psi,
        'pred_x': pred_x, 'pred_y': pred_y, 'pred_psi': pred_psi,
        # RMSE summaries
        'rmse_x': rmse_x, 'rmse_y': rmse_y, 'rmse_psi': rmse_psi,
        'rmse_xy_2d': rmse_xy_2d,
        'cum_rmse_x_t':   cum_rmse_x,
        'cum_rmse_y_t':   cum_rmse_y,
        'cum_rmse_psi_t': cum_rmse_psi,
        # forces (physical units)
        'F_true': F_true_np, 'F_fwd': F_fwd_np, 'F_inv': F_inv_np,
        # autoregressive marker — index where autoregressive prediction
        # starts. 0 in teacher-forced mode.
        'warmup_end_idx': int(warmup_end_idx),
    }


# ============================================================
# Plots
# ============================================================
def plot_trajectory_window(result: Dict[str, Any],
                           *,
                           title: Optional[str] = None,
                           save_label: Optional[str] = None,
                           plot_forces: bool = True):
    """Render the x-y plane and time-domain RMSE figure for a window result.

    Two figures are saved (via plotting.save_figure):

    1. x-y + RMSE time:
        - Top-left: predicted vs sim path on the x-y plane.
          In autoregressive mode, an additional marker shows where the
          warmup ended and autoregressive prediction began; before this
          point pred and sim coincide.
        - Bottom-left: heading psi vs time (sim, predicted).
        - Right column (3 stacked): cumulative-RMSE-over-time for x, y, psi.
          Warmup boundary marked with a vertical line in autoregressive mode.

    2. forces (only if `plot_forces` and F_fwd / F_inv are present):
        - Per-wheel Fx track: sim, fwd, inv overlaid.
        - Per-wheel Mz track: sim, fwd, inv overlaid.
        - Warmup boundary marked in autoregressive mode.

    `save_label` defaults to a string built from traj name + rollout
    mode + start index + window length, so multiple calls produce
    distinguishable PNG filenames.
    """
    rollout = result.get('rollout', 'teacher_forced')
    warmup_end_idx = int(result.get('warmup_end_idx', 0))
    is_autoreg = (rollout == 'autoregressive')

    if save_label is None:
        save_label = (f"trajwin_{result['name']}_"
                      f"{rollout}_"
                      f"start{result['start_idx']}_"
                      f"win{result['window_seconds']:.1f}s")

    rel_t = result['rel_t']
    # Time at which autoregressive rollout begins (used for plot markers).
    t_autoreg_start = rel_t[warmup_end_idx] if is_autoreg else None

    # --------------------------------------------------------
    # Figure 1: x-y plane + cumulative RMSE
    # --------------------------------------------------------
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.4, 1.0],
                          hspace=0.35, wspace=0.3)

    ax_xy   = fig.add_subplot(gs[0:2, 0])
    ax_psi  = fig.add_subplot(gs[2,   0])
    ax_rx   = fig.add_subplot(gs[0,   1])
    ax_ry   = fig.add_subplot(gs[1,   1], sharex=ax_rx)
    ax_rp   = fig.add_subplot(gs[2,   1], sharex=ax_rx)

    # x-y plane
    ax_xy.plot(result['true_x'], result['true_y'], 'k-', lw=1.6, label='sim')
    ax_xy.plot(result['pred_x'], result['pred_y'], 'r--', lw=1.4, label='predicted')
    ax_xy.scatter(result['true_x'][0],  result['true_y'][0],
                  s=60, c='green', marker='o',
                  edgecolors='k', zorder=5, label='start')
    ax_xy.scatter(result['true_x'][-1], result['true_y'][-1],
                  s=60, c='black', marker='s',
                  edgecolors='k', zorder=5, label='sim end')
    ax_xy.scatter(result['pred_x'][-1], result['pred_y'][-1],
                  s=60, c='red',  marker='X',
                  edgecolors='k', zorder=5, label='pred end')

    # In autoregressive mode, show where the warmup ended on the x-y plot.
    # Up to this point pred and sim coincide; after this the drift begins.
    if is_autoreg and warmup_end_idx > 0:
        ax_xy.scatter(result['true_x'][warmup_end_idx],
                      result['true_y'][warmup_end_idx],
                      s=80, c='gold', marker='*',
                      edgecolors='k', linewidths=0.8, zorder=6,
                      label=f'autoregressive starts (t={t_autoreg_start:.2f}s)')

    ax_xy.set_xlabel('x (m)'); ax_xy.set_ylabel('y (m)')
    rollout_tag = f"  [{rollout}]"
    if is_autoreg:
        rollout_tag += f" seed={warmup_end_idx}"
    ax_xy.set_title(
        f"x-y trajectory{rollout_tag}  |  μ={result['mu']:.3f} χ={result['chi']:.4f}  "
        f"motion={result['motion']}\n"
        f"RMSE_x={result['rmse_x']:.3f} m  "
        f"RMSE_y={result['rmse_y']:.3f} m  "
        f"RMSE_xy={result['rmse_xy_2d']:.3f} m  "
        f"RMSE_psi={result['rmse_psi']:.3f} rad",
        fontsize=10,
    )
    ax_xy.set_aspect('equal', adjustable='datalim')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc='best', fontsize=8)

    # psi over time
    ax_psi.plot(rel_t, result['true_psi'], 'k-',  lw=1.4, label=r'$\psi$ sim')
    ax_psi.plot(rel_t, result['pred_psi'], 'r--', lw=1.2, label=r'$\psi$ predicted')
    if is_autoreg and t_autoreg_start is not None:
        ax_psi.axvline(t_autoreg_start, color='gold', lw=1.2, ls=':',
                       alpha=0.9, label='autoregressive starts')
    ax_psi.set_xlabel('time since window start (s)')
    ax_psi.set_ylabel(r'$\psi$ (rad)')
    ax_psi.grid(True, alpha=0.3)
    ax_psi.legend(loc='best', fontsize=8)

    # cumulative RMSE over time
    for ax, key, color, ylabel in (
        (ax_rx, 'cum_rmse_x_t',   'b-', 'cum. RMSE x (m)'),
        (ax_ry, 'cum_rmse_y_t',   'g-', 'cum. RMSE y (m)'),
        (ax_rp, 'cum_rmse_psi_t', 'm-', 'cum. RMSE ψ (rad)'),
    ):
        ax.plot(rel_t, result[key], color, lw=1.4)
        if is_autoreg and t_autoreg_start is not None:
            ax.axvline(t_autoreg_start, color='gold', lw=1.0, ls=':', alpha=0.9)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    ax_rx.set_title('Cumulative RMSE over time', fontsize=10)
    ax_rp.set_xlabel('time since window start (s)')

    if title:
        fig.suptitle(title, fontsize=11)
    save_figure(label=f"{save_label}_xyrmse")

    # --------------------------------------------------------
    # Figure 2: forces (sim vs fwd-model vs inv-model)
    # --------------------------------------------------------
    if not plot_forces:
        return
    F_true = result['F_true']
    F_fwd  = result['F_fwd']
    F_inv  = result['F_inv']
    if F_fwd is None and F_inv is None:
        return

    fig, axs = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    fig.suptitle(
        f"Forces — {result['name']}{rollout_tag}\n"
        f"sim (—)   fwd model (--)   inv model (:)",
        fontsize=10,
    )

    for i in range(4):
        line, = axs[0].plot(rel_t, F_true[:, i],
                            '-',  alpha=0.55, label=f'wheel {i+1}')
        if F_fwd is not None:
            axs[0].plot(rel_t, F_fwd[:, i],  '--',
                        color=line.get_color(), alpha=0.95)
        if F_inv is not None:
            axs[0].plot(rel_t, F_inv[:, i],  ':',
                        color=line.get_color(), alpha=0.95)
    if is_autoreg and t_autoreg_start is not None:
        axs[0].axvline(t_autoreg_start, color='gold', lw=1.0, ls=':', alpha=0.9,
                       label='autoregressive starts')
    axs[0].set_ylabel('Fx (N)')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc='best', fontsize=7, ncol=5)

    for i in range(4):
        line, = axs[1].plot(rel_t, F_true[:, 8 + i],
                            '-',  alpha=0.55, label=f'wheel {i+1}')
        if F_fwd is not None:
            axs[1].plot(rel_t, F_fwd[:, 8 + i], '--',
                        color=line.get_color(), alpha=0.95)
        if F_inv is not None:
            axs[1].plot(rel_t, F_inv[:, 8 + i], ':',
                        color=line.get_color(), alpha=0.95)
    if is_autoreg and t_autoreg_start is not None:
        axs[1].axvline(t_autoreg_start, color='gold', lw=1.0, ls=':', alpha=0.9)
    axs[1].set_ylabel('Mz (N·m)')
    axs[1].set_xlabel('time since window start (s)')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc='best', fontsize=7, ncol=4)

    save_figure(label=f"{save_label}_forces")


# ============================================================
# Convenience: run + plot together
# ============================================================
def evaluate_and_plot_trajectory_window(
    model: MecanumPINN,
    traj: Dict[str, Any],
    rp: RobotParams,
    config: Dict[str, Any],
    *,
    start_idx: int = 0,
    window_seconds: float = 6.0,
    initial_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rollout: str = 'teacher_forced',
    seq_len_init: Optional[int] = None,
    plot_forces: bool = True,
    title: Optional[str] = None,
    save_label: Optional[str] = None,
) -> Dict[str, Any]:
    """One-call wrapper: evaluate + plot, return the result dict.

    See `evaluate_trajectory_window` for the meaning of `rollout` and
    `seq_len_init`. Same defaults as the underlying function.
    """
    result = evaluate_trajectory_window(
        model, traj, rp, config,
        start_idx=start_idx, window_seconds=window_seconds,
        initial_pose=initial_pose,
        rollout=rollout, seq_len_init=seq_len_init,
    )
    plot_trajectory_window(result, title=title, save_label=save_label,
                           plot_forces=plot_forces)
    return result
