#!/usr/bin/env python
# =============================================================================
# config_v2hy3.py — Observer v2-Hy3 run configuration.
#
# Extends ObserverConfigV2 (base γ-only brief) with the Hy3-specific knobs:
#   * learned ΔV head and its supervised weight w_dv
#   * dv_scale frozen-p95 normalization
#   * regime-split physics gate parameters
#   * Mindlin Picard iterations for the steady-state slip-fraction
# Hy3 does NOT use the base brief's physics_variant selector; it always uses
# the regime-split steady-state loss.
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from .config_v2 import ObserverConfigV2
from . import config as C


@dataclass
class ObserverConfigV2Hy3(ObserverConfigV2):
    # ΔV head
    w_dv: float = 1.0
    dv_scale: Tuple[float, float] = (C.PRED_P95["Vx"], C.PRED_P95["Vy"])

    # Derived-slip consistency loss: pull the MODEL-DERIVED per-wheel contact
    # slip |Vp| = |contact_from_gamma(gamma_hat_phys, Vpx0_hat+dV, ...)| (the SAME
    # quantity the physics gate consumes) toward the true vpm label. No new head —
    # v_slip is a function of the (gamma_hat, dV_hat) the model already predicts.
    # 0 = off. Its MSE is normalized by vpm_scale (frozen p95 of |Vp|).
    w_slip: float = 0.0
    vpm_scale: float = 0.17538320049643497

    # Regime-split gate (smooth sigmoid over |Vp|)
    gate_center: float = C.LG_V_STR          # 0.01 m/s
    gate_width: float = C.LG_V_STR           # 0.01 m/s -> one Stribeck velocity

    # Steady-state force law
    mindlin_iters: int = 2

    # Disable base-brief variant selector; Hy3 has its own physics loss.
    physics_variant: str = "hy3_regime_split"

    # Sensor front-end knob (mirrors tools_accel/frontend_audit.py selection)
    vel_filter_crossover_hz: float = 1.0
    vel_filter_integrator: str = "rot_ab2"
    noise_stage: str = "none"                # {"none", "real"}

    # --- GradNorm: gradient-balanced adaptive weighting between the two TARGETS
    # (gamma, dV). Weights are trained to equalize per-task gradient magnitudes at
    # the shared layer, scaled by relative training rate r^alpha (slower task ->
    # up-weighted). Not gameable like uncertainty weighting (the weights minimize a
    # gradnorm target, not the task loss). The slip term stays a FIXED regularizer,
    # OUTSIDE this pool. When off, weights are fixed (1, w_dv).
    use_gradnorm: bool = False
    gradnorm_alpha: float = 1.5              # restoring force toward equal rates
    gradnorm_lr: float = 0.025               # LR for the 2 weight params

    # --- cosine-conflict diagnostic (measurement only; never changes the loss) ---
    # GradNorm reacts to gradient MAGNITUDE, so it cannot tell "gamma has a big
    # gradient" apart from "gamma and dV fight over the shared encoder". Only the
    # gradient DIRECTION cosine at the shared trunk distinguishes them:
    #   cos < 0  -> the heads pull the representation opposite ways: a split /
    #               per-wheel adapter encoder is justified.
    #   cos >= 0 -> the trunk serves both; any imbalance is a pure weighting issue.
    # Costs 2 extra autograd.grad per MEASURED batch, so it is gated to a heartbeat
    # epoch and a capped batch count -> negligible run cost.
    cosine_diag: bool = False
    cosine_diag_every: int = 10          # heartbeat: measure when epoch % N == 0
    cosine_diag_batches: int = 20        # max batches measured per heartbeat epoch

    # --- run-2 fixes: grounding early-stop + LR reset at the physics hand-off ---
    # Grounding supervised loss plateaus well before the 80-epoch budget; stop it
    # early (on the SUPERVISED val metric, NOT the physics-dominated terminal one)
    # to save compute and avoid over-collapsing LR before physics starts.
    grounding_early_stop: bool = True
    grounding_min_epochs: int = 20           # floor so BOTH gamma+dV heads ground
    grounding_patience: int = 8              # epochs w/o supervised-val improvement
    grounding_rel_tol: float = 1e-3          # rel improvement to reset patience
    # Reset optimizer LR to cfg.lr AND re-instantiate the scheduler at the first
    # non-grounding epoch, so the physics curriculum starts at full LR regardless
    # of any collapse during grounding (the run-1 bug: LR dead by ep77).
    lr_reset_at_physics: bool = True

    def resolved(self) -> "ObserverConfigV2Hy3":
        # Run the v2 resolver (Path normalization, cap checks, max-norm pin).
        super().resolved()
        if self.gate_width <= 0.0:
            raise ValueError("gate_width must be positive")
        if self.mindlin_iters < 1:
            raise ValueError("mindlin_iters must be >= 1")
        if self.vel_filter_integrator not in ("euler", "rot_ab2"):
            raise ValueError("vel_filter_integrator must be 'euler' or 'rot_ab2'")
        if self.noise_stage not in ("none", "real"):
            raise ValueError("noise_stage must be 'none' or 'real'")
        return self

    @property
    def in_dim(self) -> int:
        # 5 sensor-real globals + 4 per-wheel feats + wheel embedding
        from .config import N_PERWHEEL
        return 5 + N_PERWHEEL + self.emb_dim

    @property
    def run_tag(self) -> str:
        if self.run_tag_override:
            return self.run_tag_override
        fold = self.train_fold or "S0"
        tag = f"{fold}_train_w{self.window}_gamma_dv_v2hy3_phys_max_norm"
        if self.noise_stage == "real":
            tag += "_noisy"
        return tag
