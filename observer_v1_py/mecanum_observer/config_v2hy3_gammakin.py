#!/usr/bin/env python
# =============================================================================
# config_v2hy3_gammakin.py — Observer v2-Hy3 "gamma-kin residual" VARIANT config.
#
# SEPARATE FILE ON PURPOSE. ObserverConfigV2Hy3 (config_v2hy3.py) is left byte-for-
# byte alone so every run up to now stays reproducible; this subclass carries only
# the new parametrization. Pick the variant explicitly via
# train_observer_v2hy3_gammakin.py — nothing on the v2hy3 path changes.
#
# What differs from the v2hy3 baseline, and why:
#
# 1. gamma RESIDUAL parametrization (nd711 §5.1 "Model 1", nonholonomic/no-slip).
#    Model 1 splits the two contact conditions: VPiX = 0 fixes the WHEEL spin,
#    VPiY = 0 fixes the ROLLER spin, giving the closed form
#        gamma_noslip_i = (V_Y + Omega*rho_Xi) / ((Rd - R) * cos delta_i)
#    -> gamma depends on V_Y and Omega ONLY; not V_X, not the measured wheel speed.
#    Measured over 9.94M sim samples, gamma_noslip predicts gamma to
#        RMS 0.285 rad/s in STICK   (vs the trained v2hy3 head's 5.56 rad/s => ~19x)
#        RMS 3.617 in slip, 8.171 in high-slip;  overall RMS 3.241, p95 6.491
#    So the network should learn only the deviation:
#        gamma_hat = gamma_noslip(V_y_used) + dgamma_hat * dgamma_scale
#    In stick that collapses to "predict ~0" instead of "predict a p95=84 rad/s
#    signal to 1%", which is what the |Vp| gate needs (gate width = 1 cm/s, while
#    the v2hy3 head's 5.56 rad/s stick error => 7.8 cm/s of |Vp| error, 7.8x over).
#
# 2. dgamma_scale = gamma_p95/10 (discretionary). Reusing gamma_p95 = 82.8 would
#    normalize dgamma (RMS 3.24) to std 0.039 — the same mis-normalization that
#    starved dV_x. gamma_p95/10 = 8.28 sits 1.28x from the measured p95|dgamma|
#    (6.49) and needs NO new percentile row / no front-end coupling.
#
# 3. UNIFORM dv_scale = mean(p95 Vx, p95 Vy) = 1.2967 m/s. Isotropic on purpose:
#    d|Vp|/ddV ~ 1 on BOTH axes, so the dV loss should track the physical vector
#    error |dV_err|^2 in m/s. The baseline's per-axis (1.920, 0.673) normalizes a
#    ~(5.3, 13.8) cm/s RMS correction by the VELOCITY p95 -> dV_x std 0.028 vs
#    dV_y 0.205 (7.4x). At init that is a 98.2% / 1.8% y/x gradient split; measured
#    consequence: dV_x captured 45% (corr 0.853) vs dV_y 90.7% (corr 0.996).
#    Uniform keeps the physical 2.6x (RMS_y/RMS_x) and lifts x to ~13%.
#
# 4. vy_label lambda — the sim-to-real phase-out. gamma_noslip needs V_y:
#        V_y_used = V_hat_y + (lam*dV_true + (1-lam)*dV_hat)_y * dv_scale
#    lam=1 bootstraps on the TRUE V_y label (grounding); lam=0 is the deployable
#    endpoint (base from V_hat + the model's own dV_hat). Ramped like w_phys so the
#    label is gone once physics engages. NOTE at lam=0 the base and Vpy0_u share
#    V_y, so Vpy = Vpy0 + gamma_noslip*dVpy_dg == 0 cancels identically.
#    dgamma/dV_y ~ 97.5 rad/s per m/s, so at lam<1 the gradient path from gamma back
#    into dv_hat carries MORE leverage than the slip term's 70:1 that was measured
#    corrupting dV (4.2x at w_slip=0.1). The detach policy is deliberately deferred:
#    it does not arise while lam=1 (grounding trains on the label).
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from . import config as C
from .config_v2 import GAMMA_P95_DEFAULT
from .config_v2hy3 import ObserverConfigV2Hy3

_DV_UNIFORM = 0.5 * (C.PRED_P95["Vx"] + C.PRED_P95["Vy"])      # 1.2967 m/s


@dataclass
class ObserverConfigV2Hy3GammaKin(ObserverConfigV2Hy3):
    # --- (3) uniform dV normalization ---
    dv_scale: Tuple[float, float] = (_DV_UNIFORM, _DV_UNIFORM)

    # --- (1)+(2) gamma residual ---
    gamma_residual: bool = True
    dgamma_scale: float = GAMMA_P95_DEFAULT / 10.0                # 8.2806 rad/s

    # --- (4) label phase-out for the gamma_noslip base ---
    vy_label_start: float = 1.0      # lambda during grounding (TRUE V_y label)
    vy_label_end: float = 0.0        # lambda once physics is fully engaged
    # If > 0, lambda ramps LINEARLY start->end over this many GROUNDING epochs, then
    # holds at end (decoupled from the physics ramp). Used by the gamma-only phase-out
    # fine-tune: ramp 1->0 over N epochs so the gamma head tracks the base shift
    # gradually instead of taking a sudden jump when the label is withdrawn.
    # 0 = grounding lambda is constant at vy_label_start (original behavior).
    vy_label_ramp_epochs: int = 0

    # Cut the gradient path from gamma back into dv_hat through the base. MOOT while
    # lam = 1 (the base is then the TRUE V_y label and carries no grad to dv_hat), so
    # this only bites at the phase-out — where dgamma_noslip/dV_y ~ 97.5 rad/s per m/s
    # is MORE leverage onto dV than the slip term's ~70:1 that was measured corrupting
    # dV 4.2x at w_slip=0.1. Defaults to the safe side; revisit when lam first drops.
    gamma_base_detach: bool = True

    # The baseline's high-slip upweight fights the gate: it puts 3x weight on the
    # 1.5-2.1% high-slip tail while gamma is WORST in the 20.8-28.6% stick bin,
    # which is exactly where the gate decides. Under the residual it also matters
    # less (dgamma really is larger in slip), so default it off.
    gamma_high_slip_upweight: float = 1.0

    # --- slip-loss shape (log-domain redesign) ------------------------------
    # "mse": linear normalized MSE, mean((|Vp|-vpm)^2)/vpm_scale^2  (REPRODUCES
    #        _noslip/_slip02/_wslip1 exactly; leave as the default).
    # "log": half-square log-ratio, mean((0.5*log((|Vp|^2+e^2)/(vpm^2+e^2)))^2).
    # Motivation: under a constant-relative-error model the LINEAR form spends
    # 73.5% of its gradient on the >p99 tail (1.0% of samples) and 0.0% on the
    # stick band -- the population the gate actually decides. The log form cuts
    # the tail to ~1.5% and moves 23% into the gate band. It is a REDISTRIBUTION,
    # not a rescale: a p50-vs-p95 normalizer change cannot do this.
    slip_loss_kind: str = "mse"
    # eps is NOT a numerical guard -- it is the turnover scale of the implied
    # per-sample weight w(v) = (v^2/(v^2+e^2))^2, which peaks at v = e/sqrt(3).
    # Set to the GATE WIDTH. Larger eps (0.0173 to put the peak exactly at the
    # gate, or the vpm p50) monotonically drains gate-band gradient share
    # (23.0% -> 13.3% -> 7.3%); peak LOCATION is not the figure of merit, band
    # SHARE is. Do not go below 0.005: worst-case gradient scales as 1/eps and
    # the loss starts fitting unidentifiable near-zero noise (vpm has exact 0s).
    slip_log_eps: float = 0.01

    # Use the component-consistent vpm label from the `.vpcomp.npz` sidecars
    # (observer_v1_py/build_vp_components.py) instead of the cached vpm.
    # The cache stores decimate(hypot(Vpx,Vpy)); hypot is convex, so by Jensen
    # that is biased HIGH vs hypot(decimate(Vpx),decimate(Vpy)) -- which is what
    # the model path produces. Measured over the full test splits: stick_frac
    # +10.5% (S1) / +7.7% (S2) relative, all crossings one-directional, p95/p99
    # unchanged (<0.25%, so vpm_scale needs no revision). Negligible for "mse"
    # (0.0% of its gradient sits where the bias lives) but MATERIAL for "log".
    use_vp_components: bool = False

    # --- gamma-only fine-tune of the label phase-out ------------------------
    # Warm-start the FULL model from a trained gammakin checkpoint, then (with
    # freeze_encoder_dv) freeze encoder + feat + wheel_emb + head_dv and train ONLY
    # head_gamma. Combined with the vy_label ramp (start=1, end=0, ramp_epochs=40)
    # this runs the sim-to-real phase-out as a STABLE fine-tune: because dv_hat is
    # frozen it is a fixed deterministic function of the window, so the
    # gamma_noslip(V̂+ΔV̂) base stops moving and the gamma head is not chasing a
    # drifting target. It re-fits dgamma to the DEPLOYABLE (model-prediction) base
    # instead of the true-V_y label it was trained on, removing the lam=1->lam=0
    # train/eval mismatch WITHOUT risking the ~1 cm/s ΔV (kept frozen, accepted as
    # good enough). head_gamma is NOT reinitialized — it keeps its trained weights.
    finetune_gamma_from: str = ""     # checkpoint path; "" = normal training
    freeze_encoder_dv: bool = False   # freeze all but head_gamma

    def resolved(self) -> "ObserverConfigV2Hy3GammaKin":
        super().resolved()
        if self.dgamma_scale <= 0.0:
            raise ValueError("dgamma_scale must be positive")
        for nm, v in (("vy_label_start", self.vy_label_start),
                      ("vy_label_end", self.vy_label_end)):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{nm} must be in [0,1]; got {v}")
        if self.slip_loss_kind not in ("mse", "log"):
            raise ValueError(f"slip_loss_kind must be 'mse' or 'log'; "
                             f"got {self.slip_loss_kind!r}")
        if self.slip_log_eps <= 0.0:
            raise ValueError("slip_log_eps must be positive")
        return self

    @property
    def run_tag(self) -> str:
        if self.run_tag_override:
            return self.run_tag_override
        fold = self.train_fold or "S0"
        tag = f"{fold}_train_w{self.window}_gamma_dv_v2hy3_gammakin"
        if self.noise_stage == "real":
            tag += "_noisy"
        return tag
