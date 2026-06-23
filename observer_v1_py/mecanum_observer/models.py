#!/usr/bin/env python
# =============================================================================
# models.py — wheel-shared causal encoders + per-state heads.
#
# Two interchangeable encoders behind one interface:
#   * MambaLiteSSM  — diagonal *selective* state-space scan, hand-unrolled in
#     plain PyTorch (full forward/double-backward AD, deterministic, no CUDA
#     kernel; handoff §4.2). Selectivity = input-dependent decay, mirroring the
#     LuGre bristle relaxation sigma0|v|/g(v).
#   * GRUBaseline   — nn.GRU fallback (Approach-1-validated hidden=16 spirit).
#
# Both are CAUSAL: the representation at the final timestep (past-only) drives
# the heads -> a deployable filter, not a smoother (decision: sim-to-real).
# The encoder is shared across the 4 wheels; a wheel embedding breaks symmetry.
# =============================================================================
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from .config import ObserverConfig, N_GLOBAL, N_PERWHEEL, N_STATES, N_WHEELS


class MambaLiteSSM(nn.Module):
    """Minimal selective diagonal SSM (one block).

    State recurrence per channel d, timestep t (input x_t = projected feature),
    with Δ and A anchored to PHYSICAL time (T_s = 1/rate):
        dt_t  = dt_min * (dt_max/dt_min)^sigmoid(Linear_dt(u_t))   (selective, in [dt_min,dt_max] s)
        a     = -exp(A_log)  = -1/tau                             (<0, physical rate 1/s)
        decay = exp(dt_t * a)                                     in (0,1)
        h_t   = decay * h_{t-1} + dt_t * B_t * x_t
        y_t   = sum_n C_t * h_t   (+ D * x_t)
    Selectivity stays in Δ (log-uniform in the bounds; nominal = geomean = T_s)
    AND in B_t, C_t (input-dependent). A is a fixed-init physical spectrum (modes
    span tau in [tau_min, tau_max]). Scan is an explicit Python loop over the
    (short) window -> AD-safe and deterministic."""

    def __init__(self, d_model: int, state_dim: int,
                 dt_min: float, dt_max: float, tau_min: float, tau_max: float):
        super().__init__()
        self.d_model = d_model
        self.n = state_dim
        self.in_proj = nn.Linear(d_model, d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.dt_proj.bias)                    # nominal Δ = geomean = T_s
        self.B_proj = nn.Linear(d_model, state_dim)
        self.C_proj = nn.Linear(d_model, state_dim)
        self._dt_min = dt_min
        self._log_dt_ratio = math.log(dt_max / dt_min)
        # A init: tau log-spaced in [tau_min, tau_max] (s); A = -1/tau = -exp(A_log)
        tau = torch.logspace(math.log10(tau_min), math.log10(tau_max), state_dim)
        self.A_log = nn.Parameter(-torch.log(tau))           # A_log = log(1/tau)
        self.D = nn.Parameter(torch.zeros(d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: [B, W, d_model] -> returns final-step representation [B, d_model]
        B, W, d = u.shape
        x = self.in_proj(u)                                  # [B, W, d]
        # selective Δ, log-uniform in [dt_min, dt_max] (physical seconds)
        dt = self._dt_min * torch.exp(self._log_dt_ratio * torch.sigmoid(self.dt_proj(u)))
        a = -torch.exp(self.A_log)                           # [n] = -1/tau < 0
        Bt = self.B_proj(u)                                  # [B, W, n]
        Ct = self.C_proj(u)                                  # [B, W, n]
        h = u.new_zeros(B, d, self.n)                        # [B, d, n]
        y_last = None
        for t in range(W):
            decay = torch.exp(dt[:, t, :].unsqueeze(-1) * a.view(1, 1, -1))  # [B,d,n]
            h = decay * h + (dt[:, t, :].unsqueeze(-1)
                             * x[:, t, :].unsqueeze(-1)
                             * Bt[:, t, :].unsqueeze(1))      # [B,d,n]
            y_last = (h * Ct[:, t, :].unsqueeze(1)).sum(-1) + self.D * x[:, t, :]
        return self.norm(y_last)                              # [B, d]


class GRUBaseline(nn.Module):
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.gru = nn.GRU(in_dim, d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)                                  # [B, W, d]
        return self.norm(out[:, -1, :])                      # last step (causal)


class WheelObserver(nn.Module):
    """measurable window -> {gamma, zx, zy, zs} per wheel.

    Per-wheel input at each step = [globals(3) ‖ perwheel(4) ‖ wheel_emb(e)].
    All 4 wheels run through the SAME encoder (batched on the wheel axis) with a
    wheel embedding for identity (zero/frozen here). Heads are per-state MLPs."""

    def __init__(self, cfg: ObserverConfig):
        super().__init__()
        self.cfg = cfg
        self.wheel_emb = nn.Embedding(N_WHEELS, cfg.emb_dim)
        # Identical sim wheels -> zero-init the embedding; freeze it for this run so
        # the model stays fully wheel-symmetric. It's a structural hook: unfreeze
        # later for wheel-asymmetry / sim-to-real to absorb per-wheel differences.
        nn.init.zeros_(self.wheel_emb.weight)
        if cfg.freeze_wheel_emb:
            self.wheel_emb.weight.requires_grad_(False)
        raw_in = N_GLOBAL + N_PERWHEEL + cfg.emb_dim
        if cfg.model == "ssm":
            self.feat = nn.Linear(raw_in, cfg.d_model)
            self.encoder = MambaLiteSSM(cfg.d_model, cfg.state_dim,
                                        cfg.ssm_dt_min, cfg.ssm_dt_max,
                                        cfg.ssm_tau_min, cfg.ssm_tau_max)
            self._ssm = True
        elif cfg.model == "gru":
            self.encoder = GRUBaseline(raw_in, cfg.d_model)
            self._ssm = False
        else:
            raise ValueError(f"unknown model {cfg.model!r}")
        self.heads = nn.ModuleList([                          # SiLU: matches Mamba
            nn.Sequential(nn.Linear(cfg.d_model, cfg.head_hidden), nn.SiLU(),
                          nn.Linear(cfg.head_hidden, 1))
            for _ in range(N_STATES)
        ])

    def forward(self, Gw: torch.Tensor, Pw: torch.Tensor) -> torch.Tensor:
        # Gw: [B, W, 3]   Pw: [B, W, 4, N_PERWHEEL]  ->  pred [B, 4, N_STATES]
        B, W, _ = Gw.shape
        G = Gw.unsqueeze(2).expand(B, W, N_WHEELS, N_GLOBAL)  # [B,W,4,3]
        emb = self.wheel_emb(torch.arange(N_WHEELS, device=Gw.device))  # [4,e]
        emb = emb.view(1, 1, N_WHEELS, -1).expand(B, W, N_WHEELS, -1)
        x = torch.cat([G, Pw, emb], dim=-1)                  # [B,W,4,raw_in]
        x = x.permute(0, 2, 1, 3).reshape(B * N_WHEELS, W, -1)  # [B*4,W,raw_in]
        if self._ssm:
            rep = self.encoder(self.feat(x))                 # [B*4, d]
        else:
            rep = self.encoder(x)                            # [B*4, d]
        outs = [head(rep) for head in self.heads]            # 4 x [B*4,1]
        pred = torch.cat(outs, dim=-1)                       # [B*4, 4]
        return pred.view(B, N_WHEELS, N_STATES)              # [B,4,4]


def build_model(cfg: ObserverConfig) -> WheelObserver:
    return WheelObserver(cfg)
