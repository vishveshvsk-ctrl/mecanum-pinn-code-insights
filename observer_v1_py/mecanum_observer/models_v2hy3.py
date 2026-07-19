#!/usr/bin/env python
# =============================================================================
# models_v2hy3.py — γ + ΔV causal observer model for Observer v2-Hy3.
#
# Shared per-wheel encoder (v1 MambaLiteSSM / GRUBaseline, imported unchanged)
# feeds two separate readout heads:
#   * head_gamma: per-wheel roller-spin γ̂ [B,4]
#   * head_dv:    concat of the 4 wheel latents → global ΔV̂ [B,2]
#
# V_true is NEVER an input; the two heads interact only through the physics loss.
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from .config import N_PERWHEEL, N_WHEELS
from .config_v2hy3 import ObserverConfigV2Hy3
from .models import GRUBaseline, MambaLiteSSM


class WheelObserverV2Hy3(nn.Module):
    """measurable window -> (γ̂ [B,4], ΔV̂ [B,2]).

    Per-wheel input at each step = [globals(5) ‖ perwheel(4) ‖ wheel_emb(e)].
    The shared encoder is batched on the wheel axis; a wheel embedding breaks
    symmetry.  head_gamma is per-wheel; head_dv concatenates the 4 wheel
    latents so wheel identity (fixed O-configuration) is preserved."""

    def __init__(self, cfg: ObserverConfigV2Hy3):
        super().__init__()
        self.cfg = cfg
        self.wheel_emb = nn.Embedding(N_WHEELS, cfg.emb_dim)
        nn.init.zeros_(self.wheel_emb.weight)
        if cfg.freeze_wheel_emb:
            self.wheel_emb.weight.requires_grad_(False)

        raw_in = cfg.in_dim
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

        # γ head: per-wheel readout.
        self.head_gamma = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden), nn.SiLU(),
            nn.Linear(cfg.head_hidden, 1)
        )
        # ΔV head: concat 4 wheel latents, preserve asymmetric O-configuration.
        self.head_dv = nn.Sequential(
            nn.Linear(N_WHEELS * cfg.d_model, cfg.head_hidden), nn.SiLU(),
            nn.Linear(cfg.head_hidden, 2)
        )

    def forward(self, Gw: torch.Tensor, Pw: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            Gw: [B, W, 5]  sensor-real globals (V̂x, V̂y, ψ̇_gyro, a_x, a_y), normalised
            Pw: [B, W, 4, N_PERWHEEL]  per-wheel measurables, normalised
        Returns:
            gamma_hat: [B, 4]  per-wheel roller spin, normalised
            dv_hat:    [B, 2]  body-velocity correction, normalised by dv_scale
        """
        B, W, _ = Gw.shape
        G = Gw.unsqueeze(2).expand(B, W, N_WHEELS, 5)
        emb = self.wheel_emb(torch.arange(N_WHEELS, device=Gw.device))
        emb = emb.view(1, 1, N_WHEELS, -1).expand(B, W, N_WHEELS, -1)
        x = torch.cat([G, Pw, emb], dim=-1)
        x = x.permute(0, 2, 1, 3).reshape(B * N_WHEELS, W, -1)
        if self._ssm:
            rep = self.encoder(self.feat(x))
        else:
            rep = self.encoder(x)
        # rep: [B*4, d_model]
        gamma_hat = self.head_gamma(rep).view(B, N_WHEELS)  # [B,4]

        # Reshape to [B, 4, d_model] and concat for the ΔV head.
        H = rep.view(B, N_WHEELS, -1)
        H_cat = H.reshape(B, -1)
        dv_hat = self.head_dv(H_cat)
        return gamma_hat, dv_hat


def build_model_v2hy3(cfg: ObserverConfigV2Hy3) -> WheelObserverV2Hy3:
    return WheelObserverV2Hy3(cfg)


def load_warm_start_v2hy3(model: WheelObserverV2Hy3, ckpt_path: str) -> List[str]:
    """Weights-only load from a v1 3-state checkpoint.

    Transfers encoder core + wheel_emb; skips the v1 head bank, the v1 feat
    projection (raw_in mismatch 11 vs 13), and the new head_dv (always fresh).
    """
    ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    src = ckpt.get("model", ckpt)
    dst = model.state_dict()
    skipped: List[str] = []
    loaded: List[str] = []
    for k, v in src.items():
        if k.startswith("heads."):
            skipped.append(k)
            continue
        if k == "feat.weight" or k == "feat.bias":
            skipped.append(f"{k} (raw_in mismatch)")
            continue
        if k in dst:
            if dst[k].shape == v.shape:
                dst[k].copy_(v)
                loaded.append(k)
            else:
                skipped.append(f"{k} (shape mismatch {tuple(v.shape)} vs {tuple(dst[k].shape)})")
        else:
            skipped.append(f"{k} (not in v2hy3 model)")
    model.load_state_dict(dst, strict=False)
    print(f"[warm-start-v2hy3] loaded {len(loaded)} tensors, skipped {len(skipped)}")
    if skipped:
        for k in skipped[:20]:
            print(f"  skipped: {k}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")
    return skipped
