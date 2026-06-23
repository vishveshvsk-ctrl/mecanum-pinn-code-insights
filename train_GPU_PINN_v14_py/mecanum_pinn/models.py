"""Neural-network model definitions for the Mecanum PINN.

v12.5 — factored forward model with three decoder heads (no learned θ head),
factored inverse model with per-wheel processing, and zero-initialized
per-wheel learnable embeddings on every per-wheel head in both models.
The embedding lets us keep weight-sharing at sim-time (sample-efficient,
matches the simulator's wheel symmetry) while leaving a clean adaptation
path for sim-to-real fine-tuning, where wheels degrade differently.

ARCHITECTURE NOTE (v14 θ-rework)
----------------------------------
θ is consumed by the encoder GRU and dec1 as periodic features
(sin(12·θ), cos(12·θ)) so the model can predict the 12-fold force
oscillations that real Mecanum wheels exhibit, but **θ is not part of
the model output**. The simulator's θ propagation is exactly θ̇ = ω, so
no learned head can do better than kinematic integration outside the
model. Reasons for this split:

  (1) The previous learned θ head fought the atan2 wrap's periodic loss
      landscape — random init scatters the predicted Δθ across multiple
      equivalent minima separated by π/6, and per-sample gradients
      conflicted, leaving θ MSE stuck at the random baseline ≈ 2·Var
      while V/ω components converged 3-4 decades lower.
  (2) The physics that depends on θ (force oscillation via ΔY_i and the
      moment-arm correction in the platform yaw equation) is entirely
      captured by the friction-prediction path, which still receives θ
      via sin/cos features.
  (3) For autoregressive use (MPC, multi-step eval), the caller does
      θ_next = wrap(θ_curr_folded + ω·dt) externally — exact for Euler,
      with the Euler-vs-RK4 gap O(ω̇·dt²/2) bounded to a few mrad/step.

Forward model output: S_pred ∈ ℝ^(B, L, 7), layout [V_X, V_Y, Ω, ω_1..4].
The state vector in the data and the inverse model's S_curr / S_next
inputs remain 11-dim (V + ω + θ); only the forward output drops θ.

Forward model
-------------

Encoder: SHARED per-wheel GRU, called four times with tied weights.
  per-wheel input  : [embed_i, sin(12·θ_i), cos(12·θ_i), ω_i, U_i,
                      V_X, V_Y, Ω, T]
                     (embed_dim + 8 dim total)
  per-wheel hidden : H_wheel_i ∈ ℝ^hidden_dim_wheel  (default 4)
  compound H       : stack(H_wheel_i) flattened to ℝ^(n_wheels · h_wheel)

dec1: SHARED per-wheel friction-factor head.
  input  : [embed_i, H_wheel_i, U_i, V]
           (embed_dim + h_wheel + 1 + 3 dim)
  layers : Linear(in → 32) → SiLU → Linear(32 → 32) → SiLU
                       → Linear(32 → 32) → SiLU → Linear(32 → 5)
  output : 5 factors per wheel → algebraic reconstruction → F ∈ ℝ^12

dec2: platform-velocity head, three purely STRUCTURAL additive paths.
H_compound is NOT passed here — platform dynamics is rigid M⁻¹ kinematics
and should not absorb per-wheel state. Total dec2 params: 12·3 + 3·3 +
3·3 = 54 (Linear with no bias).
  V_next = V + linear_F(F)                  # force-summing
             + coriolis([Ω·V_X, Ω·V_Y, Ω²])   # Coriolis modes
             + u_mode([U_X, U_Y, U_τ])        # torque modes

dec3: SHARED per-wheel ω-velocity head. Small MLP (sim-to-real hedge).
  input  : [embed_i, ω_i, U_i, F_x,i, T]   (embed_dim + 4 dim)
  layers : Linear(in → 8) → SiLU → Linear(8 → 1, bias=False)
  output : Δω_i;  ω_next,i = ω_i + Δω_i

(No dec_theta — θ_next is not predicted; see ARCHITECTURE NOTE above.)

wheel_embed: nn.Embedding(n_wheels, embed_dim), zero-initialized.
The embedding is shared by encoder/dec1/dec3 so "wheel i" means the
same thing across the three heads. At sim-time the data is symmetric
across wheels and these gradients average out, so the embeddings stay
near zero — behavior is identical to a fully shared network. At
sim-to-real fine-tuning, the embeddings can grow to encode per-wheel
wear / mounting / manufacturing differences without requiring any
architectural surgery.

Inverse model
-------------

Per-wheel SHARED inverse network, called four times with tied weights.

  per-wheel input  : [embed_i, H_wheel_i (if use_H),
                      U_i, V, ΔV, ω_i, ω_next,i,
                      Vpx_i, Vpy_i,                       # slip
                      sin(12·θ_i), cos(12·θ_i)]
                     dim = embed_dim + (h_wheel if use_H else 0) + 13
  layers           : Linear(in → 32) → SiLU → Linear(32 → 32) → SiLU
                       → Linear(32 → 3)
  output           : (Fx_raw, Fy_raw, Mz_raw)_i

The inverse model still reads θ from S_curr (for the slip approximation
and sin/cos features) but doesn't predict θ either. Disk and Mz caps
are unchanged — the disk uses |F| ≤ N (NOT μ·N) because the inverse is
a force *observer* and must remain free to report any force consistent
with the absolute physical upper bound. Conditioning the disk on μ
would defeat the observer's role (it could no longer report sudden μ
changes as sudden |F| changes). The Mz cap stays at
mz_bound_factor · N for the same reason.

Total params (with default knobs)
---------------------------------
forward: ~3.1k (dec_theta removed; was ~3.2k)
inverse: ~1.9k
Combined: ~5.0k vs v12's ~249k — ~50× reduction.

JVP behavior
------------
T enters the per-wheel GRU as a feature AND directly as input to dec3.
Every output of forward_path therefore depends on T through H_wheel
(and downstream via F) or directly. JVP w.r.t. T flows through all
three heads, and the v12-style continuous forward physics residual
works on the 7-dim S_pred (no θ-identity term — see ARCHITECTURE NOTE).

Public surface (unchanged in interface)
---------------------------------------
    PurePyTorchGRU         JVP-compatible hand-unrolled GRU cell
    MecanumForwardModel    factored forward model (S_pred is 7-dim)
    MecanumInverseModel    factored inverse model (per-wheel)
    MecanumPINN            unchanged coordinator
    set_grad               toggle requires_grad
    maybe_compile_pinn     wraps forward_path / inverse_path with torch.compile
    build_empty_pinn       construct an untrained PINN from a config dict
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .physics import Geometry, RobotParams, make_geometry, sawtooth_approx


# ============================================================
# JVP-compatible GRU cell — unchanged from v12
# ============================================================
class PurePyTorchGRU(nn.Module):
    """Hand-unrolled GRU compatible with torch.func.jvp."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_ir = nn.Linear(input_size,  hidden_size)
        self.W_iz = nn.Linear(input_size,  hidden_size)
        self.W_in = nn.Linear(input_size,  hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size)
        self.W_hz = nn.Linear(hidden_size, hidden_size)
        self.W_hn = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None):
        B, L, _ = x.shape
        if hx is None:
            hx = torch.zeros(1, B, self.hidden_size, device=x.device, dtype=x.dtype)
        h_t = hx[0]
        outs = []
        for t in range(L):
            x_t = x[:, t, :]
            r_t = torch.sigmoid(self.W_ir(x_t) + self.W_hr(h_t))
            z_t = torch.sigmoid(self.W_iz(x_t) + self.W_hz(h_t))
            n_t = torch.tanh   (self.W_in(x_t) + r_t * self.W_hn(h_t))
            h_t = (1 - z_t) * n_t + z_t * h_t
            outs.append(h_t.unsqueeze(1))
        out = torch.cat(outs, dim=1)
        return out, h_t.unsqueeze(0)


# ============================================================
# Forward model
# ============================================================
class MecanumForwardModel(nn.Module):
    """Factored encoder + three decoder heads with per-wheel embeddings.

    S_pred is 7-dim: [V_X, V_Y, Ω, ω_1..4]. θ is consumed as input
    (for sin/cos features feeding the GRU and dec1) but not predicted.
    For autoregressive rollout, integrate θ kinematically outside the
    model: θ_next = wrap(θ_curr_folded + ω · dt). See top-of-file
    ARCHITECTURE NOTE for rationale.

    v12.5 kwargs:

        hidden_dim_wheel : per-wheel GRU hidden dim. If None, derived as
                           hidden_dim // n_wheels. Default config sets
                           hidden_dim=16 with n_wheels=4 → wheel hidden = 4.
        dec1_hidden      : width of the per-wheel friction head's three
                           hidden layers. Default 32.
        dec3_hidden      : width of the per-wheel ω head's hidden layer.
                           Default 8.
        embed_dim        : dimensionality of the per-wheel learnable
                           embedding (zero-initialized). Default 4.
                           Pass embed_dim=0 to disable.
    """

    N_DEC1_PER_WHEEL = 5

    # Base per-wheel GRU input layout (before the wheel embedding is prepended):
    #   [sin(12·θ_i), cos(12·θ_i), ω_i, U_i, V_X, V_Y, Ω, T]
    WHEEL_GRU_INPUT_DIM_BASE = 8

    def __init__(self,
                 raw_state_dim: int = 11,
                 net_state_dim: int = 15,           # ignored; kept for compat
                 ctrl_dim: int = 4,
                 force_dim: int = 12,
                 hidden_dim: int = 16,
                 n_wheels: int = 4,
                 hidden_dim_wheel: Optional[int] = None,
                 dec1_hidden: int = 32,
                 dec3_hidden: int = 8,
                 embed_dim: int = 4,
                 F_max: Optional[torch.Tensor] = None,
                 Mz_max: Optional[torch.Tensor] = None,
                 **_legacy_kwargs):
        """
        _legacy_kwargs absorbs deprecated args like `dec_theta_hidden`
        from older configs / checkpoints so old training scripts don't
        crash on instantiation. The values are silently ignored.
        """
        super().__init__()
        if _legacy_kwargs:
            ignored = sorted(_legacy_kwargs.keys())
            print(f"[MecanumForwardModel] ignoring deprecated kwargs: {ignored} "
                  f"(removed when dec_theta was dropped — θ_next is no longer "
                  f"a learned output; integrate kinematically outside the model)")

        self.raw_state_dim = raw_state_dim
        self.ctrl_dim      = ctrl_dim
        self.force_dim     = force_dim
        self.n_wheels      = n_wheels
        self.embed_dim     = embed_dim

        # ---- Force-magnitude normalization buffers ------------
        # `_reconstruct_forces` produces physical Newtons via the algebraic
        # friction reconstruction; dividing by F_max / Mz_max here puts the
        # output in the same normalized space as `F_sim` from the dataset.
        # Stored as non-persistent buffers (not saved to state_dict) since
        # they're derived from RobotParams, not learned.
        if F_max is None:
            from .physics import F_MAX as _F_MAX_DEFAULT
            F_max = torch.tensor(_F_MAX_DEFAULT, dtype=torch.float32)
        if Mz_max is None:
            from .physics import MZ_MAX as _MZ_MAX_DEFAULT
            Mz_max = torch.tensor(_MZ_MAX_DEFAULT, dtype=torch.float32)
        if not torch.is_tensor(F_max):
            F_max = torch.tensor(F_max, dtype=torch.float32)
        if not torch.is_tensor(Mz_max):
            Mz_max = torch.tensor(Mz_max, dtype=torch.float32)
        # Defensive clone() to prevent the model's buffer from aliasing
        # the caller's RobotParams instance.
        self.register_buffer('_F_max',  F_max.float().detach().clone(),  persistent=False)
        self.register_buffer('_Mz_max', Mz_max.float().detach().clone(), persistent=False)

        # ---- Resolve per-wheel hidden dim --------------------
        if hidden_dim_wheel is None:
            if hidden_dim % n_wheels != 0:
                raise ValueError(
                    f"hidden_dim ({hidden_dim}) must be divisible by n_wheels "
                    f"({n_wheels}) when hidden_dim_wheel is not given."
                )
            hidden_dim_wheel = hidden_dim // n_wheels
        self.hidden_dim_wheel = hidden_dim_wheel
        self.hidden_dim = hidden_dim_wheel * n_wheels

        # ---- Per-wheel learnable embedding (zero-init) -------
        # Shared across encoder / dec1 / dec3 so "wheel i" means the same
        # thing to every per-wheel head. zero_() initialization makes
        # sim-time training identical to a fully-shared network; the
        # embedding receives gradient signal only when per-wheel asymmetric
        # data is encountered (real-world fine-tuning).
        if embed_dim > 0:
            self.wheel_embed = nn.Embedding(n_wheels, embed_dim)
            nn.init.zeros_(self.wheel_embed.weight)
        else:
            self.wheel_embed = None
        # Cache the index tensor; persistent=False so it doesn't bloat the
        # state-dict.
        self.register_buffer('_wheel_idx',
                             torch.arange(n_wheels, dtype=torch.long),
                             persistent=False)

        # ---- Encoder: per-wheel GRU (shared across wheels) ----
        gru_input_dim = self.WHEEL_GRU_INPUT_DIM_BASE + embed_dim
        self.gru_wheel = PurePyTorchGRU(gru_input_dim, hidden_dim_wheel)

        # ---- dec1: shared per-wheel friction-factor head ------
        # Input: [embed_i, H_wheel_i, U_i, V] = embed_dim + h_wheel + 1 + 3
        dec1_in_dim = embed_dim + hidden_dim_wheel + 1 + 3
        self.dec1 = nn.Sequential(
            nn.Linear(dec1_in_dim, dec1_hidden), nn.SiLU(),
            nn.Linear(dec1_hidden, dec1_hidden), nn.SiLU(),
            nn.Linear(dec1_hidden, dec1_hidden), nn.SiLU(),
            nn.Linear(dec1_hidden, self.N_DEC1_PER_WHEEL),
        )

        # ---- dec2: platform-velocity head (three additive paths) ----
        # All three paths are STRUCTURAL: each is a single Linear with no
        # bias and no nonlinearity. The point is that platform dynamics in
        # body frame is a rigid M⁻¹ map from (forces, Coriolis poly-features,
        # torque modes) to (dV_X, dV_Y, dΩ). The network only has to learn
        # 12+3+3 columns of constants, nothing more.
        self.dec2_linear_F = nn.Linear(force_dim, 3, bias=False)
        self.dec2_coriolis = nn.Linear(3, 3, bias=False)
        self.dec2_u_mode   = nn.Linear(3, 3, bias=False)

        # ---- dec3: shared per-wheel ω-velocity head ----
        # Input: [embed_i, ω_i, U_i, F_x,i, T] = embed_dim + 4
        dec3_in_dim = embed_dim + 4
        self.dec3 = nn.Sequential(
            nn.Linear(dec3_in_dim, dec3_hidden), nn.SiLU(),
            nn.Linear(dec3_hidden, 1, bias=False),
        )

        # NOTE: dec_theta has been removed. θ_next is not predicted —
        # callers integrate kinematically when needed (see top-of-file).

    # =========================================================
    # Static helpers — fixed projections (no parameters)
    # =========================================================
    @staticmethod
    def _coriolis_features(V: torch.Tensor) -> torch.Tensor:
        """Polynomial features [Ω·V_X, Ω·V_Y, Ω²] for platform Coriolis."""
        Vx = V[..., 0:1]
        Vy = V[..., 1:2]
        Om = V[..., 2:3]
        return torch.cat([Om * Vx, Om * Vy, Om * Om], dim=-1)

    @staticmethod
    def _torque_modes(U: torch.Tensor) -> torch.Tensor:
        """Fixed kinematic projection of wheel torques onto platform-DoF basis."""
        U1 = U[..., 0:1]; U2 = U[..., 1:2]
        U3 = U[..., 2:3]; U4 = U[..., 3:4]
        U_X =  U1 + U2 + U3 + U4
        U_Y = -U1 + U2 + U3 - U4
        U_t = -U1 + U2 - U3 + U4
        return torch.cat([U_X, U_Y, U_t], dim=-1)

    # =========================================================
    # Force reconstruction — algebraic; outputs normalized forces
    # =========================================================
    def _reconstruct_forces(self, dec1_out: torch.Tensor,
                            mu_b: torch.Tensor, chi_b: torch.Tensor,
                            N_per_wheel: torch.Tensor,
                            eps: float = 1e-8) -> torch.Tensor:
        """Map dec1 output (B, L, n_wheels, 5) → forces (B, L, 12).

        Output forces are in NORMALIZED space (matching F_sim from the
        dataset, i.e. physical forces divided by F_max / Mz_max).
        """
        smoother_raw = dec1_out[..., 0]
        sin_a_raw    = dec1_out[..., 1]
        cos_a_raw    = dec1_out[..., 2]
        log_a1       = dec1_out[..., 3]
        m_spin       = dec1_out[..., 4]

        smoother = torch.sigmoid(smoother_raw)
        norm_a = torch.sqrt(sin_a_raw**2 + cos_a_raw**2 + eps).clamp(min=1e-4)
        sin_a = sin_a_raw / norm_a
        cos_a = cos_a_raw / norm_a

        if mu_b.dim() == 1:
            mu_b = mu_b.view(-1, 1, 1)
        elif mu_b.dim() == 2:
            mu_b = mu_b.unsqueeze(-1)
        if chi_b.dim() == 1:
            chi_b = chi_b.view(-1, 1, 1)
        elif chi_b.dim() == 2:
            chi_b = chi_b.unsqueeze(-1)

        a1 = torch.exp(log_a1)
        a2 = 0.4 * a1
        S_force  = 1.0 / (a1 * chi_b + 1.0)
        S_moment = 1.0 / (a2 * chi_b + 1.0)
        N_b = N_per_wheel.view(1, 1, -1)

        # Physical magnitudes
        force_mag_phys = -mu_b * N_b * smoother * S_force
        Fx_phys = force_mag_phys * cos_a
        Fy_phys = force_mag_phys * sin_a
        Mz_phys = (-mu_b * N_b * (chi_b ** 2) * torch.tanh(m_spin * 3.0)
                   * a1 * 3.0 * torch.pi / 40.0 * S_moment * smoother)

        # Normalize to match F_sim's units.
        Fx = Fx_phys / self._F_max
        Fy = Fy_phys / self._F_max
        Mz = Mz_phys / self._Mz_max

        return torch.cat([Fx, Fy, Mz], dim=-1)

    # =========================================================
    # Embedding helper
    # =========================================================
    def _all_embeds(self) -> Optional[torch.Tensor]:
        """Return (n_wheels, embed_dim) embedding table, or None if disabled."""
        if self.wheel_embed is None:
            return None
        return self.wheel_embed(self._wheel_idx)

    # =========================================================
    # Forward path — vectorized over wheels
    # =========================================================
    def forward_path(self, S_hist: torch.Tensor, U_curr: torch.Tensor,
                     T_curr: torch.Tensor, mu_batch: torch.Tensor,
                     chi_batch: torch.Tensor, N_per_wheel: torch.Tensor):
        """Predict next state and forces from the current window.

        S_pred dimensionality is 7 (V + ω). θ is consumed as input via
        the sin/cos features feeding the GRU and dec1, but no θ_next
        is produced — callers integrate θ_next = wrap(θ_curr + ω·dt)
        externally when needed (see top-of-file ARCHITECTURE NOTE).

        Parameters
        ----------
        S_hist        : (B, L, 11)  [V_X, V_Y, Ω, ω₁..₄, θ₁..₄]
        U_curr        : (B, L, 4)
        T_curr        : (B, L) or (B, L, 1)
        mu_batch, chi_batch : (B,) or (B, 1)
        N_per_wheel   : (n_wheels,) physical normal force per wheel

        Returns
        -------
        S_pred        : (B, L, 7)   [V_X_next, V_Y_next, Ω_next, ω₁..₄_next]
        F_fwd         : (B, L, 12)
        H_compound    : (B, L, n_wheels · hidden_dim_wheel)
        """
        B, L, _ = S_hist.shape
        if T_curr.dim() == 2:
            T_curr = T_curr.unsqueeze(-1)
        nw = self.n_wheels

        V     = S_hist[..., 0:3]    # (B, L, 3)
        omega = S_hist[..., 3:7]    # (B, L, nw)
        theta = S_hist[..., 7:11]   # (B, L, nw)  -- input only; not predicted

        # ---- Per-wheel feature broadcast: (B, L, nw, *) -----
        # Embeddings broadcast from (nw, embed_dim) to (B, L, nw, embed_dim)
        if self.wheel_embed is not None:
            embeds = self.wheel_embed(self._wheel_idx)              # (nw, embed_dim)
            embeds_BLN = embeds.view(1, 1, nw, self.embed_dim).expand(B, L, -1, -1)
        else:
            embeds_BLN = S_hist.new_empty(B, L, nw, 0)

        # Platform-level features broadcast to per-wheel
        V_BLN     = V.unsqueeze(2).expand(-1, -1, nw, -1)            # (B, L, nw, 3)
        T_BLN     = T_curr.unsqueeze(2).expand(-1, -1, nw, -1)       # (B, L, nw, 1)
        omega_BLN = omega.unsqueeze(-1)                              # (B, L, nw, 1)
        U_BLN     = U_curr.unsqueeze(-1)                             # (B, L, nw, 1)
        theta_BLN = theta.unsqueeze(-1)                              # (B, L, nw, 1)
        sin12_BLN = torch.sin(12.0 * theta_BLN)                      # (B, L, nw, 1)
        cos12_BLN = torch.cos(12.0 * theta_BLN)                      # (B, L, nw, 1)

        # ============================================================
        # Encoder: ONE GRU call processing all 4 wheels in parallel
        # ============================================================
        # The sin/cos features are how θ informs the model's predictions
        # of forces (12-fold oscillation) and ω (drag torques) — without
        # θ ever appearing on the output.
        gru_in_BLN = torch.cat(
            [embeds_BLN, sin12_BLN, cos12_BLN, omega_BLN, U_BLN,
             V_BLN, T_BLN], dim=-1)
        # Reshape so wheels join the batch dim:
        #   (B, L, nw, in_dim) → (B, nw, L, in_dim) → (B·nw, L, in_dim)
        gru_in_flat = (gru_in_BLN.permute(0, 2, 1, 3)
                                  .reshape(B * nw, L, -1))
        H_flat, _ = self.gru_wheel(gru_in_flat)                     # (B·nw, L, h_wheel)
        # Reshape back: (B·nw, L, h_wheel) → (B, nw, L, h_wheel) → (B, L, nw, h_wheel)
        H_wheels = (H_flat.view(B, nw, L, self.hidden_dim_wheel)
                          .permute(0, 2, 1, 3)
                          .contiguous())
        H_compound = H_wheels.reshape(B, L, self.hidden_dim)

        # ============================================================
        # dec1: friction-factor head — single call across all wheels
        # ============================================================
        dec1_in = torch.cat(
            [embeds_BLN, H_wheels, U_BLN, V_BLN], dim=-1)           # (B, L, nw, in)
        dec1_per_wheel = self.dec1(dec1_in)                          # (B, L, nw, 5)

        # Algebraic reconstruction → 12-D force
        F_fwd = self._reconstruct_forces(dec1_per_wheel, mu_batch,
                                         chi_batch, N_per_wheel)

        # ============================================================
        # dec2: platform-velocity head (three structural paths, no H)
        # ============================================================
        dV_F   = self.dec2_linear_F(F_fwd)
        dV_cor = self.dec2_coriolis(self._coriolis_features(V))
        dV_U   = self.dec2_u_mode(self._torque_modes(U_curr))
        V_next = V + dV_F + dV_cor + dV_U

        # ============================================================
        # dec3: ω-velocity head — single call
        # ============================================================
        Fx_BLN = F_fwd[..., 0:nw].unsqueeze(-1)                      # (B, L, nw, 1)
        dec3_in = torch.cat(
            [embeds_BLN, omega_BLN, U_BLN, Fx_BLN, T_BLN], dim=-1)
        dW_per_wheel = self.dec3(dec3_in).squeeze(-1)                # (B, L, nw)
        omega_next = omega + dW_per_wheel

        # θ_next is NOT computed — see top-of-file ARCHITECTURE NOTE.
        # For autoregressive rollout, do this externally:
        #   theta_curr_folded = atan2(sin(12·θ_curr), cos(12·θ_curr)) / 12
        #   theta_next = atan2(sin(12·(theta_curr_folded + ω_phys·dt)),
        #                      cos(12·(theta_curr_folded + ω_phys·dt))) / 12
        # with ω_phys in PHYSICAL rad/s (de-normalized by state_max[3:7]).
        S_pred = torch.cat([V_next, omega_next], dim=-1)             # (B, L, 7)
        return S_pred, F_fwd, H_compound

    # =========================================================
    # Diagnostics
    # =========================================================
    def physical_diagnostics(self) -> Dict[str, torch.Tensor]:
        """Snapshot of structurally-meaningful trained linear weights."""
        out: Dict[str, torch.Tensor] = {}
        out['linear_F_weight']  = self.dec2_linear_F.weight.detach().cpu()
        out['coriolis_weight']  = self.dec2_coriolis.weight.detach().cpu()
        out['u_mode_weight']    = self.dec2_u_mode.weight.detach().cpu()
        out['dec3_w1']          = self.dec3[0].weight.detach().cpu()
        out['dec3_b1']          = self.dec3[0].bias.detach().cpu()
        out['dec3_w2']          = self.dec3[2].weight.detach().cpu()
        if self.wheel_embed is not None:
            out['wheel_embed']  = self.wheel_embed.weight.detach().cpu()
            # Per-wheel embedding norms — a quick sanity check that
            # the embeddings stay near zero at sim-time.
            out['wheel_embed_norms'] = self.wheel_embed.weight.detach().norm(dim=1).cpu()
        return out


# ============================================================
# Inverse model — unchanged from prior v12.5
# ============================================================
class MecanumInverseModel(nn.Module):
    """Per-wheel factored inverse force/moment observer.

    The model is called four times with tied weights, producing per-wheel
    (Fx, Fy, Mz) predictions. A per-wheel zero-initialized embedding allows
    the network to specialize per wheel at sim-to-real fine-tuning time
    without disturbing the simulator fit.

    The inverse model still reads θ from S_curr (for the slip
    approximation and sin/cos features) but does not predict θ. Its
    S_curr / S_next inputs remain 11-dim (V + ω + θ) — only the
    forward model's S_pred drops θ.

    Output bounds are intentionally μ-agnostic:
      |F_i| ≤ N_i (per-wheel normal force — hard physical upper bound)
      |Mz_i| ≤ mz_bound_factor · N_i

    use_H toggles whether the per-wheel hidden state from the forward
    encoder is passed in.
    """

    # Per-wheel base input features (excluding embedding + optional H):
    #   U_i           1
    #   V             3
    #   ΔV            3
    #   ω_i           1
    #   ω_next,i      1
    #   Vpx_i         1
    #   Vpy_i         1
    #   sin(12·θ_i)   1
    #   cos(12·θ_i)   1
    #   ---total---  13
    PER_WHEEL_BASE_DIM = 13

    def __init__(self, geom: Geometry,
                 raw_state_dim: int = 11, net_state_dim: int = 15,
                 ctrl_dim: int = 4, force_dim: int = 12,
                 hidden_dim: int = 16, n_wheels: int = 4,
                 hidden_dim_wheel: Optional[int] = None,
                 inv_hidden: int = 32,
                 embed_dim: int = 4,
                 use_H: bool = True, mz_bound_factor: float = 0.01):
        super().__init__()
        self.use_H = use_H
        self.force_dim = force_dim
        self.n_wheels = n_wheels
        self.mz_bound_factor = mz_bound_factor
        self.geom = geom
        self.raw_state_dim = raw_state_dim
        self.net_state_dim = net_state_dim       # kept for backward compat
        self.ctrl_dim = ctrl_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # Resolve per-wheel hidden dim from compound H if not given
        if hidden_dim_wheel is None:
            if hidden_dim % n_wheels != 0:
                raise ValueError(
                    f"hidden_dim ({hidden_dim}) must be divisible by n_wheels "
                    f"({n_wheels}) when hidden_dim_wheel is not given."
                )
            hidden_dim_wheel = hidden_dim // n_wheels
        self.hidden_dim_wheel = hidden_dim_wheel

        # ---- Per-wheel learnable embedding (zero-init) ---------
        if embed_dim > 0:
            self.wheel_embed = nn.Embedding(n_wheels, embed_dim)
            nn.init.zeros_(self.wheel_embed.weight)
        else:
            self.wheel_embed = None
        self.register_buffer('_wheel_idx',
                             torch.arange(n_wheels, dtype=torch.long),
                             persistent=False)

        # ---- Per-wheel shared inverse network -------------------
        h_in = (hidden_dim_wheel if use_H else 0)
        per_wheel_in_dim = embed_dim + h_in + self.PER_WHEEL_BASE_DIM
        self.inverse_net = nn.Sequential(
            nn.Linear(per_wheel_in_dim, inv_hidden), nn.SiLU(),
            nn.Linear(inv_hidden, inv_hidden),       nn.SiLU(),
            nn.Linear(inv_hidden, 3),                            # (Fx, Fy, Mz) per wheel
        )

    def _approximate_slip(self, S: torch.Tensor) -> torch.Tensor:
        """Per-wheel observable slip; γ_i (roller spin) terms zeroed.

        Output layout: (B, L, 2*n_wheels) with [Vpx_1..4, Vpy_1..4].
        """
        Vx      = S[..., 0:1]
        Vy      = S[..., 1:2]
        psi_dot = S[..., 2:3]
        w       = S[..., 3:7]
        theta   = S[..., 7:11]
        tan_saw = torch.tan(sawtooth_approx(theta))
        DY = self.geom.Ra * self.geom.tan_delta * tan_saw

        Vpx = Vx - psi_dot * (self.geom.wc_y + DY) - w * self.geom.R
        Vpy = Vy + psi_dot * self.geom.wc_x
        return torch.cat([Vpx, Vpy], dim=-1)

    def inverse_path(self, S_curr: torch.Tensor, S_next: torch.Tensor,
                     U: torch.Tensor, N_per_wheel: torch.Tensor,
                     H_detached: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict per-wheel forces from current and next state + control.

        S_curr and S_next remain 11-dim (full state with θ). The inverse
        model uses θ from S_curr for the slip approximation and sin/cos
        features; from S_next it only reads V and ω (for ΔV and ω_next).

        Returns (B, L, 12) with layout [Fx_1..4, Fy_1..4, Mz_1..4].
        """
        if self.use_H and H_detached is None:
            raise ValueError("use_H=True requires H_detached")
        if not self.use_H:
            H_detached = None

        B, L, _ = S_curr.shape
        nw = self.n_wheels

        # ---- State decomposition --------------------------------
        V        = S_curr[..., 0:3]
        V_next   = S_next[..., 0:3]
        delta_V  = V_next - V
        omega    = S_curr[..., 3:7]
        omega_n  = S_next[..., 3:7]
        theta    = S_curr[..., 7:11]

        # ---- Slip (B, L, 8) split into x, y components ----------
        slip = self._approximate_slip(S_curr)
        slip_x = slip[..., 0:nw]    # (B, L, nw)
        slip_y = slip[..., nw:2*nw] # (B, L, nw)

        # ---- Per-wheel broadcast: (B, L, nw, *) ----------------
        if self.wheel_embed is not None:
            embeds = self.wheel_embed(self._wheel_idx)               # (nw, embed_dim)
            embeds_BLN = embeds.view(1, 1, nw, self.embed_dim).expand(B, L, -1, -1)
        else:
            embeds_BLN = S_curr.new_empty(B, L, nw, 0)

        V_BLN       = V.unsqueeze(2).expand(-1, -1, nw, -1)          # (B, L, nw, 3)
        dV_BLN      = delta_V.unsqueeze(2).expand(-1, -1, nw, -1)    # (B, L, nw, 3)
        omega_BLN   = omega.unsqueeze(-1)                            # (B, L, nw, 1)
        omega_n_BLN = omega_n.unsqueeze(-1)                          # (B, L, nw, 1)
        U_BLN       = U.unsqueeze(-1)                                # (B, L, nw, 1)
        slip_x_BLN  = slip_x.unsqueeze(-1)                           # (B, L, nw, 1)
        slip_y_BLN  = slip_y.unsqueeze(-1)                           # (B, L, nw, 1)
        theta_BLN   = theta.unsqueeze(-1)                            # (B, L, nw, 1)
        sin12_BLN   = torch.sin(12.0 * theta_BLN)
        cos12_BLN   = torch.cos(12.0 * theta_BLN)

        # ---- Optional H from forward encoder -------------------
        feats = [embeds_BLN]
        if self.use_H:
            H_wheels = H_detached.reshape(B, L, nw, self.hidden_dim_wheel)
            feats.append(H_wheels)
        feats.extend([
            U_BLN, V_BLN, dV_BLN,
            omega_BLN, omega_n_BLN,
            slip_x_BLN, slip_y_BLN,
            sin12_BLN, cos12_BLN,
        ])
        per_wheel_in = torch.cat(feats, dim=-1)                      # (B, L, nw, in_dim)

        # ---- SINGLE forward pass through shared inverse net ----
        forces = self.inverse_net(per_wheel_in)                      # (B, L, nw, 3)
        Fx_raw = forces[..., 0]                                      # (B, L, nw)
        Fy_raw = forces[..., 1]
        Mz_raw = forces[..., 2]

        # ============================================================
        # Output bounds — μ-agnostic, observer style (unchanged)
        # ============================================================
        N_b = N_per_wheel.view(1, 1, -1)
        Mz_max = self.mz_bound_factor * N_b

        r2      = (Fx_raw**2 + Fy_raw**2) / (N_b**2 + 1e-12)
        scale_F = torch.rsqrt(1.0 + r2)
        Fx = Fx_raw * scale_F
        Fy = Fy_raw * scale_F
        Mz = Mz_raw / torch.sqrt(1.0 + (Mz_raw / (Mz_max + 1e-12))**2)

        return torch.cat([Fx, Fy, Mz], dim=-1)


# ============================================================
# Coordinator — UNCHANGED
# ============================================================
class MecanumPINN(nn.Module):
    """Lightweight wrapper holding a forward + inverse pair."""
    def __init__(self, forward_model: MecanumForwardModel,
                 inverse_model: MecanumInverseModel):
        super().__init__()
        self.forward_model = forward_model
        self.inverse_model = inverse_model

    def forward_path(self, *args, **kwargs):
        return self.forward_model.forward_path(*args, **kwargs)

    def inverse_path(self, *args, **kwargs):
        return self.inverse_model.inverse_path(*args, **kwargs)


# ============================================================
# Helpers
# ============================================================
def set_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def maybe_compile_pinn(model: MecanumPINN,
                       config: Dict[str, Any]) -> MecanumPINN:
    """Wrap forward_path and inverse_path with torch.compile when enabled."""
    if not config.get('compile_enabled', True):
        print("[compile] disabled by config")
        return model
    if config['device'].type != 'cuda':
        print("[compile] disabled (device is not CUDA)")
        return model

    fwd_mode = config.get('compile_mode_forward', 'default')
    inv_mode = config.get('compile_mode_inverse', 'reduce-overhead')
    print(f"[compile] forward_path mode='{fwd_mode}', "
          f"inverse_path mode='{inv_mode}' "
          f"(first batch will trigger tracing -- expect 30-90 s)")

    try:
        model.forward_model.forward_path = torch.compile(
            model.forward_model.forward_path,
            mode=fwd_mode, dynamic=False, fullgraph=False,
        )
    except Exception as e:
        print(f"[compile] forward_path compile failed: {e!r} -- using eager")

    try:
        model.inverse_model.inverse_path = torch.compile(
            model.inverse_model.inverse_path,
            mode=inv_mode, dynamic=False, fullgraph=False,
        )
    except Exception as e:
        print(f"[compile] inverse_path compile failed: {e!r} -- using eager")

    return model


def build_empty_pinn(config, geom, use_H=True):
    fwd = MecanumForwardModel(
        raw_state_dim=config['raw_state_dim'],
        net_state_dim=config['net_state_dim'],
        ctrl_dim=config['ctrl_dim'], force_dim=config['force_dim'],
        hidden_dim=config['hidden_dim'], n_wheels=config['n_wheels'],
    ).to(config['device'])
    inv = MecanumInverseModel(
        geom=geom, raw_state_dim=config['raw_state_dim'],
        net_state_dim=config['net_state_dim'],
        ctrl_dim=config['ctrl_dim'], force_dim=config['force_dim'],
        hidden_dim=config['hidden_dim'], n_wheels=config['n_wheels'],
        use_H=use_H,
    ).to(config['device'])
    return MecanumPINN(fwd, inv).to(config['device'])
