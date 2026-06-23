"""Mecanum_PINN_Mamba_ForceRecon_v1 — forward-inverse force-reconstruction PINN.

Successor to train_GPU_PINN_v14_py with:
  - roller-frame F_par / F_perp targets (Mz dropped),
  - a small selective-SSM ("Mamba-lite") measurable-only encoder,
  - an analytical Newton-Euler integrator (forces are the sole interface),
  - 4-term force law  F = mu(A + chi*C) + B + D  with a TEST-TIME residual
    mu readout (mu-ID is not a training objective; chi-ID deferred).

Modules land incrementally; only the foundation is exported until the rest
(models, losses, stages, training, evaluation) are added.
"""
from __future__ import annotations

from . import (config, data, evaluation, losses, manifest, models, physics,
               plotting, regime_split, stages, training)

__all__ = ["config", "data", "evaluation", "losses", "manifest", "models",
           "physics", "plotting", "regime_split", "stages", "training"]
