"""Models: selective-SSM encoder, structured force head, forward + inverse models.

Design (Approach 1):
  - Measurable-only inputs; hidden roller/bristle/contact-spin states are
    reconstructed implicitly by a small diagonal SELECTIVE SSM ("Mamba-lite",
    plain-PyTorch unrolled — supports full autodiff, deterministic, no CUDA build).
  - 4-term deck force law (roller frame, Mz dropped):
        F_i = mu*N_i*softcircle( g_slip*(A + chi*g_spin*C) ) + N_i*g_slip*B + N_i*D
    mu multiplicative; chi modulates inside (held out from ID for now); whole friction
    bracket slip-gated, C also spin-gated; B mu-indep slip-gated; D bristle (ungated).
    softcircle = friction-circle cap. mu-ID is a TEST-TIME residual readout, not trained.
  - Forward model returns PHYSICAL forces (the analytical NE integrator in
    physics.forward_integrate consumes them). Inverse model reconstructs forces
    from a short causal Delta-state window (mu-agnostic, bounded by ||F||<=N_i)
    and a linear (a=mu, b=mu*chi) readout recovers (mu_hat, chi_hat).

Tensor conventions: S (B,L,11) normalized [Vx,Vy,psi_dot, w1..4, theta1..4];
U (B,L,4) normalized Msat; mu,chi (B,). Force order: [Fpar_1..4, Fperp_1..4].
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .physics import RobotParams


# ============================================================
# Measurable per-wheel feature construction
# ============================================================
def build_wheel_features(S: torch.Tensor, U: torch.Tensor,
                         wheel_embed: torch.Tensor) -> torch.Tensor:
    """(B,L,11),(B,L,4),(4,e) -> (B,L,4,F_in) measurable-only per-wheel features.

    Per wheel i: [w_i, sin(12 th_i), cos(12 th_i), Msat_i, Vx, Vy, psi_dot] ++ embed_i.
    Linear slip precursors (e.g. Vx - psi_dot*py_i - w_i*R) are formed by the
    encoder's linear input projection, so we feed the raw measurable components.
    """
    B, L, _ = S.shape
    Vx, Vy, pd = S[..., 0:1], S[..., 1:2], S[..., 2:3]      # (B,L,1)
    w  = S[..., 3:7]                                         # (B,L,4)
    th = S[..., 7:11]                                        # (B,L,4)
    body = torch.cat([Vx, Vy, pd], dim=-1)                   # (B,L,3)

    feats = []
    for i in range(4):
        fi = torch.stack([
            w[..., i],
            torch.sin(12.0 * th[..., i]),
            torch.cos(12.0 * th[..., i]),
            U[..., i],
        ], dim=-1)                                          # (B,L,4)
        fi = torch.cat([fi, body], dim=-1)                  # (B,L,7)
        e = wheel_embed[i].view(1, 1, -1).expand(B, L, -1)  # (B,L,e)
        feats.append(torch.cat([fi, e], dim=-1))
    return torch.stack(feats, dim=2)                        # (B,L,4,F_in)


# ============================================================
# Diagonal selective SSM ("Mamba-lite"), plain-PyTorch unrolled
# ============================================================
class SelectiveSSM(nn.Module):
    """Mamba-S6 selective state-space encoder (lean core), per wheel.

    Lifts the measurable per-wheel features into a D-dim latent (`in_proj`), then
    runs a diagonal selective scan with an N-dim state PER channel:

        h_t[d,:] = exp(dt_t[d] * A[d,:]) (.) h_{t-1}[d,:] + (dt_t[d] * B_t) (.) u_t[d]
        y_t[d]   = <C_t, h_t[d,:]> + D_skip[d] * u_t[d]

    SELECTIVITY (input-dependence) lives in dt_t (per channel), B_t and C_t (the
    N-vectors, shared across the D channels) -- all linear in the lifted u. A is
    FIXED and not selective. dt_t is ANCHORED to the sampling interval T_s: it
    swings log-uniformly in [dt_min, dt_max] (~[T_s/4, 4*T_s]; nominal = geomean
    ~ T_s), so A = -exp(A_log) is a physical relaxation-rate spectrum (tau = -1/A)
    and exp(dt*A) is the physical per-step decay.

    LEAN CORE: no causal conv1d, no SiLU gate / out_proj -- the downstream ForceHead
    MLP is the nonlinear readout. The instantaneous features still reach the head
    linearly via the D_skip * u term. Returns the latent y sequence (B,L,W,D).
    """
    def __init__(self, in_dim: int, d_model: int, d_state: int,
                 dt_min: float, dt_max: float, selective_dt: bool = True):
        super().__init__()
        self.D, self.N = int(d_model), int(d_state)
        self.selective_dt = selective_dt
        self.dt_min, self.dt_max = float(dt_min), float(dt_max)
        self.log_dt_min, self.log_dt_max = math.log(self.dt_min), math.log(self.dt_max)
        self.in_proj = nn.Linear(in_dim, self.D)            # lift  f_in -> D
        self.B_proj  = nn.Linear(self.D, self.N)            # selective B (shared across D)
        self.C_proj  = nn.Linear(self.D, self.N)            # selective C (shared across D)
        self.dt_proj = nn.Linear(self.D, self.D)            # selective dt(x), per channel
        self.D_skip  = nn.Parameter(torch.ones(self.D))     # per-channel input skip
        # Fixed physical A = -exp(A_log), shape (D,N): a log-spaced relaxation-rate
        # ladder over the anchored band, identical across channels (S4D-real init).
        # dt_nom = geomean(dt_min,dt_max) ~ T_s; rates span ~[0.05,5]/dt_nom so
        # tau in ~[dt_nom/5, 20*dt_nom] (= [T_s/5, 20*T_s] ~ [0.4, 40] ms).
        dt_nom = (self.dt_min * self.dt_max) ** 0.5
        a = torch.linspace(math.log(0.05 / dt_nom), math.log(5.0 / dt_nom), self.N)
        self.A_log = nn.Parameter(a.unsqueeze(0).repeat(self.D, 1).clone())  # (D,N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B,L,W,in_dim) -> Y (B,L,W,D). W = n_wheels (folded into batch)."""
        B, L, W, _ = x.shape
        xf = x.reshape(B * W, L, -1)
        u = self.in_proj(xf)                                 # (BW,L,D) the lift
        Bsel = self.B_proj(u)                                # (BW,L,N) selective B
        Csel = self.C_proj(u)                                # (BW,L,N) selective C
        # selective per-channel step dt(x), log-uniform in [dt_min, dt_max] (nominal ~ T_s)
        if self.selective_dt:
            s = torch.sigmoid(self.dt_proj(u))               # (BW,L,D) in (0,1)
        else:
            s = torch.full((B * W, L, self.D), 0.5, device=x.device, dtype=x.dtype)
        dt = torch.exp(self.log_dt_min + (self.log_dt_max - self.log_dt_min) * s)
        A = -torch.exp(self.A_log)                           # (D,N) physical rates < 0

        h = torch.zeros(B * W, self.D, self.N, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            dt_t = dt[:, t].unsqueeze(-1)                    # (BW,D,1)
            dA  = torch.exp(dt_t * A.unsqueeze(0))           # (BW,D,N)
            dBu = (dt_t * Bsel[:, t].unsqueeze(1)) * u[:, t].unsqueeze(-1)  # (BW,D,N)
            h = dA * h + dBu
            y = (h * Csel[:, t].unsqueeze(1)).sum(-1) + self.D_skip * u[:, t]  # (BW,D)
            outs.append(y)
        Y = torch.stack(outs, dim=1)                         # (BW,L,D)
        return Y.reshape(B, W, L, self.D).permute(0, 2, 1, 3).contiguous()


def soft_circle(v: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    """Smooth friction-circle clamp: scale a vector so ||v_out|| < 1, identity
    for small ||v|| (sub-cap). v_out = v / sqrt(1 + ||v||^2)."""
    r2 = (v * v).sum(dim=dim, keepdim=True)
    return v * torch.rsqrt(1.0 + r2 + eps)


# ============================================================
# Structured force head
# ============================================================
class ForceHead(nn.Module):
    """SSM latent (D-dim) -> dimensionless shapes -> physical forces.

    4-TERM deck force law (HANDOFF_identifiability_results.md + flowchart):

        F_i = mu*N_i * softcircle( g_slip*(Phi_A + chi*g_spin*Phi_C) )   # mu(A + C*chi), capped
              + N_i * g_slip * Phi_B                                     # B: mu-indep, slip-gated
              + N_i * Phi_D                                              # D: bristle, NOT slip-gated

    mu enters multiplicatively, chi modulates inside it (chi-ID deferred; chi is a held
    input). Phi_* tanh-bounded (par,perp) directions; g_slip,g_spin >= 0 are the
    co-directional-slip and contact-spin gates; softcircle = friction-circle cap. mu-ID is
    a TEST-TIME residual readout (mu_readout_residual) -- never a training objective.
    """
    def __init__(self, in_dim: int, hidden: int = 32, four_term: bool = True):
        super().__init__()
        self.four_term = four_term
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            # [A_par,A_perp, C_par,C_perp, g_spin, B_par,B_perp, g_slip, D_par,D_perp]
            nn.Linear(hidden, 10),
        )

    def shapes(self, enc: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.net(enc)                                     # (B,L,4,10)
        sp = torch.nn.functional.softplus
        return {
            'Phi_A':  torch.tanh(out[..., 0:2]),                # mu-scaled Coulomb (slip-gated)
            'Phi_C':  torch.tanh(out[..., 2:4]),                # chi-coupling (slip+spin-gated)
            'g_spin': sp(out[..., 4]),                          # contact-spin gate (chi)
            'Phi_B':  torch.tanh(out[..., 5:7]),                # mu-indep, slip-gated
            'g_slip': sp(out[..., 7]),                          # co-directional-slip gate (mu)
            'Phi_D':  torch.tanh(out[..., 8:10]),               # bristle (mu/chi-indep, ungated)
        }

    def assemble(self, shapes: Dict[str, torch.Tensor],
                 mu: torch.Tensor, chi: torch.Tensor,
                 N_per_wheel: torch.Tensor) -> torch.Tensor:
        """-> physical forces F (B,L,8) = [Fpar_1..4, Fperp_1..4]."""
        Phi_A, Phi_C = shapes['Phi_A'], shapes['Phi_C']
        g_spin, Phi_B, g_slip, Phi_D = (shapes['g_spin'], shapes['Phi_B'],
                                        shapes['g_slip'], shapes['Phi_D'])
        mu_b  = mu.view(-1, 1, 1, 1)
        chi_b = chi.view(-1, 1, 1, 1)
        N = N_per_wheel.view(1, 1, -1, 1)                       # (1,1,4,1)
        gsl = g_slip.unsqueeze(-1); gsp = g_spin.unsqueeze(-1)
        coulomb = mu_b * N * soft_circle(gsl * (Phi_A + chi_b * gsp * Phi_C), dim=-1)
        F = coulomb
        if self.four_term:
            F = F + N * gsl * Phi_B + N * Phi_D                 # B (slip-gated) + D (bristle)
        return torch.cat([F[..., 0], F[..., 1]], dim=-1)       # (B,L,8)


# ============================================================
# Forward model
# ============================================================
class MecanumForwardModel(nn.Module):
    def __init__(self, cfg: Dict, rp: RobotParams):
        super().__init__()
        e = cfg['embed_dim']
        self.n_wheels = cfg['n_wheels']
        self.wheel_embed = nn.Parameter(torch.zeros(self.n_wheels, e))
        f_in = 7 + e                                         # see build_wheel_features
        self.encoder = SelectiveSSM(f_in, cfg['ssm_d_model'], cfg['ssm_d_state'],
                                    dt_min=cfg['ssm_dt_min'], dt_max=cfg['ssm_dt_max'],
                                    selective_dt=cfg['ssm_selective_dt'])
        # ForceHead reads the SSM latent ONLY (no raw-feat concat; the instantaneous
        # channels reach it via the encoder's D_skip * u term).
        self.head = ForceHead(cfg['ssm_d_model'], cfg['shape_hidden'],
                              four_term=cfg.get('force_four_term', True))
        self.register_buffer('N_per_wheel', rp.N_per_roller.clone())

    def forward(self, S: torch.Tensor, U: torch.Tensor,
                mu: torch.Tensor, chi: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feat = build_wheel_features(S, U, self.wheel_embed)  # (B,L,4,f_in)
        h = self.encoder(feat)                               # (B,L,4,D)
        shapes = self.head.shapes(h)
        F_phys = self.head.assemble(shapes, mu, chi, self.N_per_wheel)
        return F_phys, shapes


# ============================================================
# Inverse model + (mu_hat, chi_hat) readout
# ============================================================
class MecanumInverseModel(nn.Module):
    """Reconstructs forces from a short causal Delta-state window (mu-agnostic).

    Output at step t uses features [feat_t, feat_t-feat_{t-1}, ... ] over
    `window` steps -> per-wheel MLP -> F_inv, soft-bounded by ||F_i|| <= N_i.
    """
    def __init__(self, cfg: Dict, rp: RobotParams):
        super().__init__()
        e = cfg['embed_dim']
        self.window = max(2, int(cfg['inv_window']))
        self.n_wheels = cfg['n_wheels']
        self.wheel_embed = nn.Parameter(torch.zeros(self.n_wheels, e))
        f_in = 7 + e
        self.net = nn.Sequential(
            nn.Linear(self.window * f_in, cfg['inv_hidden']), nn.SiLU(),
            nn.Linear(cfg['inv_hidden'], cfg['inv_hidden']), nn.SiLU(),
            nn.Linear(cfg['inv_hidden'], 2),                 # (Fpar, Fperp) per wheel
        )
        self.register_buffer('N_per_wheel', rp.N_per_roller.clone())

    def forward(self, S: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """-> F_inv (B, L-window+1, 8) physical, aligned to steps [window-1 : L]."""
        feat = build_wheel_features(S, U, self.wheel_embed)  # (B,L,4,f_in)
        win = [feat[:, k:feat.shape[1] - (self.window - 1) + k]    # window slices
               for k in range(self.window)]
        # stack as [feat_t, feat_{t-1}, ...] then difference all but the newest
        newest = win[-1]
        diffs = [win[j + 1] - win[j] for j in range(self.window - 1)]
        x = torch.cat([newest] + diffs, dim=-1)              # (B,L',4, window*f_in)
        F = self.net(x)                                      # (B,L',4,2)
        F = soft_circle(F, dim=-1) * self.N_per_wheel.view(1, 1, -1, 1)  # ||F_i||<=N_i
        return torch.cat([F[..., 0], F[..., 1]], dim=-1)     # (B,L',8)


def mu_readout_residual(F: torch.Tensor, shapes: Dict[str, torch.Tensor],
                        chi: torch.Tensor, N_per_wheel: torch.Tensor,
                        eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """TEST-TIME mu identification by residual inversion (NEVER trained):

        mu_hat = proj of ( F - N*Phi_D - N*g_slip*Phi_B )  onto
                          ( N*g_slip*(Phi_A + chi*g_spin*Phi_C) )

    i.e. remove the mu-independent bristle (D) and slip-gated (B) parts, then divide by
    the mu-scaled (A + C*chi) basis -- aggregated as a slip-energy-weighted least-squares
    projection, so it is well-conditioned only where co-directional slip is high
    (mu_conf = that energy). chi is the held input. Returns (mu_hat (B,), mu_conf (B,)).
    Apply to BOTH F_inv (identification) and F_fwd (self-consistency: should recover the
    conditioning mu); the F_inv-vs-F_fwd gap is the parameter-change signal.
    """
    L = F.shape[1]
    Phi_A = shapes['Phi_A'][:, :L]; Phi_C = shapes['Phi_C'][:, :L]
    Phi_B = shapes['Phi_B'][:, :L]; Phi_D = shapes['Phi_D'][:, :L]
    g_sl  = shapes['g_slip'][:, :L].unsqueeze(-1)
    g_sp  = shapes['g_spin'][:, :L].unsqueeze(-1)
    chi_b = chi.view(-1, 1, 1, 1)
    N = N_per_wheel.view(1, 1, -1, 1)
    Fv = torch.stack([F[..., :4], F[..., 4:]], dim=-1)          # (B,L,4,2)
    y = (Fv - N * Phi_D - N * g_sl * Phi_B).flatten(1)          # remove mu-indep B, D
    X = (N * g_sl * (Phi_A + chi_b * g_sp * Phi_C)).flatten(1)  # mu-scaled basis (slip-gated)
    Sxx = (X * X).sum(1); Sxy = (X * y).sum(1)
    return Sxy / (Sxx + eps), Sxx                              # mu_hat, mu_conf (slip energy)


class MecanumPINN(nn.Module):
    """Coordinator holding the forward + inverse models."""
    def __init__(self, cfg: Dict, rp: RobotParams):
        super().__init__()
        self.forward_model = MecanumForwardModel(cfg, rp)
        self.inverse_model = MecanumInverseModel(cfg, rp)


# ============================================================
# Training utilities
# ============================================================
def set_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


def maybe_compile_pinn(model: MecanumPINN, config: Dict) -> MecanumPINN:
    """torch.compile the forward + inverse submodules (CUDA only). Checkpoints
    strip the resulting `_orig_mod.` prefix on save, so they load into an
    uncompiled skeleton. Build -> (load) -> compile is the intended order."""
    if config.get('compile_enabled') and torch.cuda.is_available():
        try:
            model.forward_model = torch.compile(
                model.forward_model, mode=config.get('compile_mode_forward', 'default'))
            model.inverse_model = torch.compile(
                model.inverse_model, mode=config.get('compile_mode_inverse', 'default'))
            print('[compile] forward + inverse submodules compiled')
        except Exception as e:                       # pragma: no cover
            print(f'[compile] disabled ({e!r})')
    return model
