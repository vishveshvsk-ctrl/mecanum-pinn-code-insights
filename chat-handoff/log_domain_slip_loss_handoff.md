# Handoff: log-domain (relative) slip-consistency loss for the hy3 gamma-kin observer

**Continues:** the hy3 gamma-kin observer session (slip-weight ablation → loss-shape redesign).
**Frame:** PINN digital twin (KUKA youBot, 4 Mecanum wheels). Observer v2-Hy3 gamma-kin predicts
per-wheel roller spin **γ** (as a residual off a closed-form no-slip base) + body-velocity
correction **ΔV**; a regime-split physics loss gates stick vs slip on a model-derived contact
slip **|Vp|**. This task redesigns the *slip consistency* loss to serve the gate better.

## Context the task depends on
- **Env:** `C:\Users\vishv\miniforge3\envs\myenv\python.exe`. Run from `code_insights/`.
- **Cache:** `C:/Users/vishv/mecanum_cache_decim` (warm, full μ/χ grid, ~5,345 files).
- **Files:**
  - `observer_v1_py/mecanum_observer/losses_v2hy3.py` — `slip_consistency_loss(v_slip, vpm_true,
    vpm_scale)` **← the function to change.** Currently `mean((v_slip−vpm)²)/vpm_scale²` (linear
    MSE, normalized by frozen p95). Also `derived_contact_slip` builds `|Vp|` from (γ̂,ΔV̂).
  - `observer_v1_py/mecanum_observer/training_v2hy3_gammakin.py` — computes/adds the slip term
    (search `l_slip`); logs `slip_rmse`. `w_slip=0` path computes it under `no_grad` for logging.
  - `observer_v1_py/mecanum_observer/config_v2hy3_gammakin.py` — `w_slip`, `vpm_scale=0.17538`,
    `gate_center=gate_width=0.01`, `gamma_high_slip_upweight` (default 1.0 in gammakin).
  - `observer_v1_py/regime_split_attrib.py` — attribution probe; `--vy-label {0,1}`,
    `--regime <toml>` for multisine. Reports stick_frac + per-regime γ/ΔV/|Vp|.
- **Scales (frozen, from `variable_scaler_percentiles.csv`):** vpm median **0.0247 m/s**, p95
  **0.17538**; γ median **9.176 rad/s**, p95 **82.806**; dv_scale uniform **1.2967**.
- **Gate:** `g = σ((|Vp|−0.01)/0.01)`. Decision at |Vp| = 1 cm/s ≈ the vpm *median*.
- **Baselines to compare against (all λ=1-trained, 80 grounding ep, S1/S2, checkpoints intact):**
  `runs/S{1,2}_..._gammakin_grnd80_{noslip, slip02, wslip1}` (w_slip = 0 / 0.02 / 1.0). Each has
  `regime_attrib_lam{0,1}.json` + `..._eval_multisine_lam{0,1}.json`.

## Purpose
Replace the linear-MSE slip loss with a **relative / log-domain** loss so it optimizes accuracy
**near the gate** (small |Vp|) instead of the large-slip tail. Success = a modest-weight log-slip
run that **matches w_slip=1.0's gate calibration** (S2 stick_frac ≈ 0.21 vs true 0.216, ~4% error)
**without** w_slip=1.0's costs: keep γ near the noslip/slip02 level and **do not** regress OOD
(multisine) γ. If achieved, the slip term earns a permanent, cheap place; if not, keep w_slip=0.02.

## Key design decisions (already made)
1. **Loss shape, not scale, is the lever.** A global normalizer (p50 vs p95) is a per-head reweight
   and scales all samples equally — it does NOT redistribute attention toward small/typical slip.
   Only a relative/log/robust loss (or gate-boundary per-sample weighting) does. Median-normalizing
   is a legit *cross-head balance* choice but does not fix the skew.
2. **Log form:** `L = mean( (log(|Vp|_hat+ε) − log(vpm+ε))² )` — squared log-ratio; scale-invariant;
   weight ∝ 1/|Vp| peaks at the gate. Use **ε ≈ gate width (0.01)** so the loss cares about relative
   accuracy down to the gate scale and goes flat below it (below gate = "stick", differences moot).
3. **|Vp| is positive → plain log(+ε).** If ever applied to signed components Vpx/Vpy, use arcsinh.
4. **Keep the slip term redundant-at-optimum** (a fn of γ,ΔV; no new head) — decision from prior
   session, unchanged. `gamma_base_detach=True` stays.
5. **Why not just raise linear w_slip:** w_slip=1.0 fixes the S2 gate (err 23%→4%) but overfits the
   large-slip tail → worst γ and worst OOD (multisine γ 0.66→0.91). γ/ΔV differences across the
   *linear* w_slip sweep are negligible (≤0.5% of p95) — the tail-overfit is the only real cost, and
   it is exactly what a log/relative loss removes.
6. **Do NOT touch inputs.** This is a loss-target change only. (Input skew is a separate, un-started
   conditioning question — p95-normalized inputs are conservative, fine.)

## Open decisions / hand-back
- **ε value** (0.01 vs a smaller floor) and **log vs a boundary-weight** (weight ∝ ∂g/∂|Vp|) —
  try log first (simplest, principled); boundary-weight is the fallback if log under-serves.
- **Weight for the log term** — start where its early-epoch gradient ≈ γ's (measure, ~like the old
  `w_slip≈0.02` balance exercise); the log loss's gradient magnitude differs from linear MSE.
- **Hand-back:** a `regime_attrib_lam0.json` (+ multisine) for the log-slip run vs the 3 linear
  baselines; conclusion on whether log-slip beats the w_slip=0.02-vs-1.0 tradeoff. That decides the
  production slip loss.

## Deliverables
1. Modified `slip_consistency_loss` (log/relative form, ε param) — surgical, keep signature stable;
   add a config flag (e.g. `slip_loss_kind: {"mse","log"}`) so linear stays reproducible.
2. One S1/S2 gammakin grounding-80 run at λ=1 with the log-slip loss, tag `_grnd80_slipLOG`
   (do NOT overwrite `_slip02/_noslip/_wslip1`).
3. `regime_attrib_lam{0,1}.json` + `..._eval_multisine_lam{0,1}.json` for both folds; a short
   comparison table (stick_frac err%, γ, OOD γ) vs the 3 linear baselines.

## Conventions to respect
- **Surgical edits, explain diffs before applying.** Variant files stay separate from baseline
  (`*_gammakin*` are the variant; never mutate `config_v2hy3.py`/`training_v2hy3.py` defaults).
- **Never overwrite existing run dirs** — new tag per experiment; `metrics.json` is the sweep's
  done-marker. **Preserve `_slip02/_noslip/_wslip1` checkpoints + JSONs.**
- **Compare spreads to the quantity's own scale** (median AND p95), not runs to each other.
- **Freeze LR scheduler + ES + best-ckpt during any ramp** (already implemented in the gammakin
  trainer via the `ramping` flag) — a plateau watcher misreads a non-stationary objective.
- **Laptop RTX 3060 6 GB:** ≤2 concurrent torch procs; `keep_awake.py` for long runs; batch 4096
  × stride 16 is the tested footprint (~3.2 GB). Grad-tracked logging tensors OOM — use `no_grad`.
- **No LaTeX in chat** (Unicode + code blocks). Verify before concluding; the user reasons hard
  from physics and catches premature trend calls.
