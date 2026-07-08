#!/usr/bin/env python
# =============================================================================
# models_v2.py — γ-only causal observer model.
#
# Reuses the v1 MambaLiteSSM / GRUBaseline encoders unchanged and replaces the
# per-state head bank with a single γ MLP.  Warm-start from a v1 3-state
# checkpoint transfers the encoder + wheel embedding; the γ head is always
# initialised fresh.
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from .config import N_GLOBAL, N_PERWHEEL, N_WHEELS
from .config_v2 import ObserverConfigV2
# Re-use v1 encoder implementations (Triton-free, plain PyTorch).
from .models import GRUBaseline, MambaLiteSSM


class WheelObserverV2(nn.Module):
    """measurable window -> γ̂ per wheel.

    Per-wheel input at each step = [globals(3) ‖ perwheel(4) ‖ wheel_emb(e)].
    All 4 wheels run through the SAME encoder (batched on the wheel axis) with a
    wheel embedding for identity (zero/frozen here). A single γ head predicts
    the roller spin rate."""

    def __init__(self, cfg: ObserverConfigV2):
        super().__init__()
        self.cfg = cfg
        self.wheel_emb = nn.Embedding(N_WHEELS, cfg.emb_dim)
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
        # Single γ head.
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden), nn.SiLU(),
            nn.Linear(cfg.head_hidden, 1)
        )

    def forward(self, Gw: torch.Tensor, Pw: torch.Tensor) -> torch.Tensor:
        """Args:
            Gw: [B, W, 3]  global measurables, normalised
            Pw: [B, W, 4, N_PERWHEEL]  per-wheel measurables, normalised
        Returns:
            gamma_hat: [B, 4]  per-wheel roller spin, normalised
        """
        B, W, _ = Gw.shape
        G = Gw.unsqueeze(2).expand(B, W, N_WHEELS, N_GLOBAL)
        emb = self.wheel_emb(torch.arange(N_WHEELS, device=Gw.device))
        emb = emb.view(1, 1, N_WHEELS, -1).expand(B, W, N_WHEELS, -1)
        x = torch.cat([G, Pw, emb], dim=-1)
        x = x.permute(0, 2, 1, 3).reshape(B * N_WHEELS, W, -1)
        if self._ssm:
            rep = self.encoder(self.feat(x))
        else:
            rep = self.encoder(x)
        out = self.head(rep).view(B, N_WHEELS)  # [B,4]
        return out


def build_model_v2(cfg: ObserverConfigV2) -> WheelObserverV2:
    return WheelObserverV2(cfg)


def load_warm_start_v2(model: WheelObserverV2, ckpt_path: str) -> List[str]:
    """Weights-only load from a v1 3-state checkpoint.

    Transfers encoder + wheel_emb + feature projection.  All v1 head-bank keys
    are skipped and returned for logging.  The v2 γ head stays fresh.
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
        if k in dst:
            if dst[k].shape == v.shape:
                dst[k].copy_(v)
                loaded.append(k)
            else:
                skipped.append(f"{k} (shape mismatch {tuple(v.shape)} vs {tuple(dst[k].shape)})")
        else:
            skipped.append(f"{k} (not in v2 model)")
    model.load_state_dict(dst, strict=False)
    print(f"[warm-start-v2] loaded {len(loaded)} tensors, skipped {len(skipped)}")
    if skipped:
        for k in skipped[:20]:
            print(f"  skipped: {k}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")
    return skipped
